"""数据工具层：财务指标 Provider。

契约：provider.get_financials(ticker) 返回 **可 JSON 序列化的 dict**，
且取到的必须是**最新报告期**的数据——下游要把它直接塞进 LLM prompt。
"""

from typing import Protocol

from src.tools.utils import (
    clean_metrics,
    clean_value,
    fetch_with_retry,
    period_key,
    pick_latest_row,
)

# 兼容旧导入路径：历史测试与脚本按私有名从本模块导入这两个清洗函数。
_clean_value = clean_value
_period_key = period_key


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


# 数值清洗与最新期选取已抽到 src/tools/utils.py（clean_value / period_key /
# pick_latest_row / clean_metrics），各 Provider 共用。
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

        df = fetch_with_retry(
            lambda: ak.stock_financial_analysis_indicator(symbol=ticker),
            label="stock_financial_analysis_indicator",
        )
        if df is None or df.empty:
            raise RuntimeError(f"akshare 未返回数据: {ticker}")
        if "日期" not in df.columns:
            raise RuntimeError(f"akshare 返回结构与预期不符，列: {list(df.columns)[:20]}")

        # 该接口按报告期**升序**返回（首行是最老的年报），必须自己挑最新一期。
        try:
            latest, date_raw = pick_latest_row(df)
        except ValueError as exc:
            raise RuntimeError(f"akshare 报告期列无法解析: {list(df['日期'].head(5))}") from exc

        as_of = period_key(latest.get("日期"))[:6]
        as_of = f"{as_of[:4]}-{as_of[4:]}" if len(as_of) == 6 else date_raw

        metric_values, missing_metrics = clean_metrics(latest, self.METRIC_MAP)
        raw_values, _ = clean_metrics(latest, self.RAW_FIELDS)

        out = {"ticker": ticker, "source": self.SOURCE}
        for field, value in metric_values.items():
            out[field] = {"value": value, "as_of": as_of}
        out.update(raw_values)
        out["report_period"] = date_raw
        out["as_of"] = as_of
        out["missing_metrics"] = sorted(missing_metrics)

        if all(f not in out for f in self.REQUIRED_ANY):
            raise RuntimeError(
                f"akshare 最新报告期({date_raw}) 缺失全部关键指标，实际列: "
                f"{list(df.columns)[:20]}"
            )
        return out
