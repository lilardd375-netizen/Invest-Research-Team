"""数据工具层：财务指标 Provider。

契约：provider.get_financials(ticker) 返回 **可 JSON 序列化的 dict**，
且取到的必须是**最新报告期**的数据——下游要把它直接塞进 LLM prompt。
"""

import math
from datetime import date, datetime
from typing import Protocol


class DataProvider(Protocol):
    """财务数据源协议。实现方需保证 get_financials 返回可 JSON 序列化的 dict。"""

    def get_financials(self, ticker: str) -> dict:  # pragma: no cover - 协议声明
        ...


class FakeProvider:
    """测试 / 离线用：构造时注入 ticker -> data 映射。"""

    def __init__(self, mapping: dict = None):
        self.mapping = mapping or {}
        self.calls: list[str] = []

    def get_financials(self, ticker: str) -> dict:
        self.calls.append(ticker)
        if ticker not in self.mapping:
            raise KeyError(f"未配置 ticker 的假数据: {ticker}")
        return self.mapping[ticker]


def _clean_value(value):
    """numpy / NaN 清洗：把数字（含数字字符串）归一成纯 Python float，NaN、inf 返回 None。

    必要性：
    1. akshare 返回 numpy 标量，NaN 经 json.dumps(default=str) 会变成字符串
       "nan"——那是**非法 JSON**，会让 LLM 拿到脏上下文并解析失败；
    2. 有的接口把数字给成字符串（如 "63.6525"），统一成 float 才能比较与计算。
    清洗不掉的非数字文本按原样返回（例如 "--"），由上游决定怎么用。
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _period_key(value) -> str:
    """把报告期归一化成可比较的 YYYYMMDD 字符串（空值 -> ""）。"""
    if isinstance(value, datetime | date):
        return value.strftime("%Y%m%d")
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


class AkshareProvider:
    """akshare 真实数据源（延迟导入：import akshare 放在方法内）。

    需要 akshare 属于可选依赖，未安装时不影响其它功能。
    """

    # 中文指标名 -> 输出字段名
    METRIC_MAP = {
        "净资产收益率(%)": "roe",
        "销售毛利率(%)": "gross_margin",
        "销售净利率(%)": "net_margin",
        "主营业务收入增长率(%)": "revenue_growth",
        "净利润增长率(%)": "net_profit_growth",
        "资产负债率(%)": "debt_ratio",
        "每股经营性现金流(元)": "operating_cashflow_per_share",
        "总资产利润率(%)": "roa",
    }
    RAW_FIELDS = {
        "主营业务利润(元)": "main_business_profit",
        "总资产(元)": "total_assets",
    }
    # 取不到就算数据不可用的关键指标（三个都缺 -> 报错，避免把空壳喂给 LLM）
    REQUIRED_ANY = ("roe", "gross_margin", "net_margin")
    SOURCE = "akshare:stock_financial_analysis_indicator"

    def get_financials(self, ticker: str) -> dict:
        import akshare as ak  # 延迟导入

        df = ak.stock_financial_analysis_indicator(symbol=ticker)
        if df is None or df.empty:
            raise RuntimeError(f"akshare 未返回数据: {ticker}")
        if "日期" not in df.columns:
            raise RuntimeError(f"akshare 返回结构与预期不符，列: {list(df.columns)[:20]}")

        # 该接口按报告期**升序**返回（首行是最老的年报），必须自己挑最新一期。
        periods = df["日期"].map(_period_key)
        if periods.eq("").all():
            raise RuntimeError(f"akshare 报告期列无法解析: {list(df['日期'].head(5))}")
        latest = df.loc[periods.idxmax()]

        date_raw = str(latest.get("日期")).strip()
        as_of = _period_key(latest.get("日期"))[:6]
        as_of = f"{as_of[:4]}-{as_of[4:]}" if len(as_of) == 6 else date_raw

        out = {"ticker": ticker, "source": self.SOURCE}
        for cn_name, field in self.METRIC_MAP.items():
            value = _clean_value(latest.get(cn_name))
            if value is not None:
                out[field] = {"value": value, "as_of": as_of}
        for cn_name, field in self.RAW_FIELDS.items():
            value = _clean_value(latest.get(cn_name))
            if value is not None:
                out[field] = value
        out["report_period"] = date_raw
        out["as_of"] = as_of
        out["missing_metrics"] = sorted(set(self.METRIC_MAP.values()) - set(out))

        if all(f not in out for f in self.REQUIRED_ANY):
            raise RuntimeError(
                f"akshare 最新报告期({date_raw}) 缺失全部关键指标，实际列: "
                f"{list(df.columns)[:20]}"
            )
        return out
