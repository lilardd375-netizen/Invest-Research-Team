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

# 报告期 -> 累计口径说明。接口给的比率都是"期内累计值"，不是年化值。
PERIOD_SCALE = {
    "03": "一季报累计（非年化）",
    "06": "半年报累计（非年化）",
    "09": "三季报累计（非年化）",
    "12": "年报累计（即为年度值）",
}


def period_scale(report_period: str) -> str:
    """由报告期推出累计口径说明（比率是期内累计值，不是年化值）。"""
    month = str(report_period or "")[5:7]
    return PERIOD_SCALE.get(month, "报告期累计值（非年化）")


def dupont_context(values: dict) -> dict:
    """由同一行指标算出杜邦上下文与恒等式自检；缺字段就留空，不猜。"""
    roe = values.get("roe")
    roa = values.get("roa")
    debt_ratio = values.get("debt_ratio")
    equity_ratio = values.get("equity_ratio")

    equity_multiplier = None
    if debt_ratio is not None and 0 <= debt_ratio < 100:
        equity_multiplier = round(1 / (1 - debt_ratio / 100), 4)
    elif equity_ratio:
        equity_multiplier = round(100 / equity_ratio, 4)

    roe_implied = None
    gap_pct = None
    if roa is not None and equity_multiplier is not None:
        roe_implied = round(roa * equity_multiplier, 4)
        if roe:
            gap_pct = round(abs(roe_implied - roe) / abs(roe) * 100, 2)

    out = {}
    for key in (
        "roe",
        "roe_weighted",
        "roa",
        "roa_net",
        "net_margin",
        "total_asset_turnover",
        "equity_ratio",
        "debt_ratio",
        "equity_multiplier",
        "roe_implied_roa_x_em",
        "identity_gap_pct",
    ):
        value = equity_multiplier if key == "equity_multiplier" else values.get(key)
        if key == "roe_implied_roa_x_em":
            value = roe_implied
        if key == "identity_gap_pct":
            value = gap_pct
        if value is not None:
            out[key] = value
    return out


def caliber_notes(values: dict, report_period: str) -> list:
    """口径标注：哪些数字同源同期间、哪些口径不可混用。直接进 LLM prompt。"""
    notes = [
        f"报告期 {report_period}：{period_scale(report_period)}；"
        "所有比率都是该期累计值而不是年化值（例如总资产周转率 0.26 次是半年度累计，"
        "年化约 0.52 次），不要拿年化阈值直接判断高低。",
    ]
    if values.get("roa") is not None:
        notes.append(
            "总资产利润率(%) 是利润总额口径，与总资产净利润率(%)（净利润口径）"
            "不是同一个 ROA，不可互换使用。"
        )
    roe = values.get("roe")
    roe_weighted = values.get("roe_weighted")
    if roe is not None and roe_weighted is not None:
        notes.append(
            f"净资产收益率(%) {roe} 为摊薄（期末）口径，加权净资产收益率(%) "
            f"{roe_weighted} 为加权口径，二者本就不同，不构成矛盾。"
        )
    notes.append(
        "本接口不提供归母 / 少数股东权益拆分：归母类比率与总资产类（全部权益）"
        "比率混用会带来约 7% 量级的偏差，比较时必须保持同一口径族。"
    )
    notes.append(
        "恒等式（同期间、同口径下可自检）：ROE(摊薄) ≈ 总资产利润率 × 权益乘数；"
        "总资产净利润率 ≈ 销售净利率 × 总资产周转率。"
    )
    return notes


def trend_series(df, date_col: str = "日期", limit: int = 5) -> list:
    """最近 limit 期的关键指标序列（升序，最后一项为最新期）。"""
    if date_col not in df.columns:
        return []
    keyed = df.assign(__pk=df[date_col].map(period_key))
    keyed = keyed[keyed["__pk"].astype(str).str.len() == 8].sort_values("__pk").tail(limit)
    series = []
    for _, row in keyed.iterrows():
        item = {"period": str(row.get(date_col))}
        for column, field in AkshareProvider.TREND_FIELDS.items():
            value = clean_value(row.get(column))
            if value is not None:
                item[field] = value
        series.append(item)
    return series


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
        "扣除非经常性损益后的净利润(元)": "net_profit_deducted",
    }
    # 杜邦上下文（与主指标同源同一行，不额外请求数据源）
    DUPONT_FIELDS = {
        "总资产净利润率(%)": "roa_net",
        "总资产周转率(次)": "total_asset_turnover",
        "股东权益比率(%)": "equity_ratio",
        "加权净资产收益率(%)": "roe_weighted",
    }
    # 近 N 期趋势（同一份 DataFrame，零额外请求）
    TREND_FIELDS = {
        "净资产收益率(%)": "roe",
        "总资产利润率(%)": "roa",
        "销售净利率(%)": "net_margin",
        "资产负债率(%)": "debt_ratio",
        "主营业务收入增长率(%)": "revenue_growth",
    }
    TREND_PERIODS = 5
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
        dupont_values, _ = clean_metrics(latest, self.DUPONT_FIELDS)
        all_values = {**metric_values, **dupont_values}

        out = {"ticker": ticker, "source": self.SOURCE}
        for field, value in metric_values.items():
            out[field] = {"value": value, "as_of": as_of}
        out.update(raw_values)
        out["report_period"] = date_raw
        out["as_of"] = as_of
        out["period_scale"] = period_scale(date_raw)
        out["dupont"] = dupont_context(all_values)
        out["trend"] = trend_series(df, limit=self.TREND_PERIODS)
        out["caliber_notes"] = caliber_notes(all_values, date_raw)
        out["missing_metrics"] = sorted(missing_metrics)

        if all(f not in out for f in self.REQUIRED_ANY):
            raise RuntimeError(
                f"akshare 最新报告期({date_raw}) 缺失全部关键指标，实际列: "
                f"{list(df.columns)[:20]}"
            )
        return out
