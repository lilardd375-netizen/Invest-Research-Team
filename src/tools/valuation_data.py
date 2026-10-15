"""数据工具层：估值 Provider。

契约：provider.get_valuation(ticker) 返回 **可 JSON 序列化的 dict**，只给可验算的
计算事实（当前值 / 窗口 / 样本数 / 历史分位数 / 行业对照），**不做「贵/便宜」的定性
判断**——定性交给分析师 LLM（第十步任务约束）。

实测口径（akshare 1.19.1 + pandas 3.0.6，2026-10-15 实测 300750）：
- 首选的 ak.stock_a_indicator_lg 在本机 akshare 1.19.1 **不存在**（AttributeError），
  改用 ak.stock_zh_valuation_baidu(symbol, indicator, period)：
  signature (symbol='002044', indicator='总市值', period='近一年')，
  列 ['date','value']，period='近三年' 返回日频 1097 行（2023-10-15 ~ 2026-10-15），
  indicator 取 '市盈率(TTM)' / '市净率' / '总市值'，单次约 1.8-2.0s；
  period='全部' 只有 763 行且非日频（2018 起稀疏），故分位数一律基于「近三年」窗口。
- 行业对照 ak.stock_industry_pe_ratio_cninfo(symbol='证监会行业分类', date='YYYYMMDD')：
  列 ['变动日期','行业分类','行业层级','行业编码','行业名称','公司数量',
  '纳入计算公司数量','总市值-静态','净利润-静态','静态市盈率-加权平均',
  '静态市盈率-中位数','静态市盈率-算术平均']，shape (120, 12)，约 0.4s；
  行业名称取自 ak.stock_profile_cninfo(symbol) 的 '所属行业'（证监会口径，
  300750 = '电气机械和器材制造业'，编码 C38）——**注意该表命中行为「中上协」标准**，
  名称与证监会口径一致故可直接匹配；
  **日期陷阱**：周末/未发布日会抛 `ValueError: Length mismatch: Expected axis has 0
  elements, new values have 12 elements`，需按天向前回退（实测 2026-10-11 失败、
  10-12/13/14 成功），故实现里带 RETRY_DAYS 天回退。
- **口径提示**：行业对照是「静态市盈率」(LYR)，个股是 PE(TTM)，两者口径不同，
  只作量级对照，输出里显式写入 caliber_notes。
"""

import statistics
from datetime import date, timedelta
from typing import Protocol

from src.tools.utils import clean_value, fetch_with_retry, normalize_ticker

PERIOD = "近三年"
PE_INDICATOR = "市盈率(TTM)"
PB_INDICATOR = "市净率"
MIN_SAMPLES = 120
INDUSTRY_SYMBOL = "证监会行业分类"
RETRY_DAYS = 8
SOURCE = "akshare:stock_zh_valuation_baidu+akshare:stock_industry_pe_ratio_cninfo"


class ValuationProvider(Protocol):
    """估值数据源协议。实现方需保证 get_valuation 返回可 JSON 序列化的 dict。"""

    def get_valuation(self, ticker: str) -> dict:  # pragma: no cover - 协议声明
        ...


class FakeValuationProvider:
    """测试 / 离线用：构造时注入 ticker -> data 映射。"""

    def __init__(self, mapping: dict = None):
        self.mapping = mapping or {}
        self.calls: list[str] = []

    def get_valuation(self, ticker: str) -> dict:
        self.calls.append(ticker)
        if ticker not in self.mapping:
            raise KeyError(f"未配置 ticker 的假数据: {ticker}")
        return self.mapping[ticker]


def _describe(points: list[tuple[str, float]], *, label: str, caliber: str) -> dict:
    """由 (日期, 值) 序列算出可验算的中间量：当前值 / 区间 / 中位数 / 分位数 / 样本数。"""
    values = [value for _, value in points]
    current = values[-1]
    at_or_below = sum(1 for value in values if value <= current)
    return {
        "label": label,
        "caliber": caliber,
        "current": round(current, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "median": round(statistics.median(values), 4),
        "percentile": round(100.0 * at_or_below / len(values), 2),
        "samples": len(values),
        "start": points[0][0],
        "end": points[-1][0],
    }


def _valuation_series(ticker: str, indicator: str) -> list[tuple[str, float]]:
    import akshare as ak

    frame = fetch_with_retry(
        lambda: ak.stock_zh_valuation_baidu(symbol=ticker, indicator=indicator, period=PERIOD),
        label=f"stock_zh_valuation_baidu({indicator})",
    )
    if frame is None or len(frame) == 0:
        raise RuntimeError(f"akshare 未返回估值序列: {indicator}")
    if "date" not in frame.columns or "value" not in frame.columns:
        raise RuntimeError(f"估值序列结构不符: {list(frame.columns)}")
    points: list[tuple[str, float]] = []
    for raw_date, raw_value in zip(frame["date"], frame["value"]):
        value = clean_value(raw_value)
        if not isinstance(value, (int, float)):
            continue
        points.append((str(raw_date).strip(), float(value)))
    points.sort(key=lambda item: item[0])
    if len(points) < MIN_SAMPLES:
        raise RuntimeError(f"估值序列样本不足: {indicator} 仅 {len(points)} 条（要求 ≥ {MIN_SAMPLES}）")
    return points


def _industry_name(ticker: str, missing: list[str]) -> str | None:
    import akshare as ak

    try:
        frame = fetch_with_retry(
            lambda: ak.stock_profile_cninfo(symbol=ticker), label="stock_profile_cninfo"
        )
    except Exception:  # noqa: BLE001 - 行业对照缺失不应让整个估值数据不可用
        missing.append("industry_name")
        return None
    if frame is None or len(frame) == 0 or "所属行业" not in frame.columns:
        missing.append("industry_name")
        return None
    name = str(frame["所属行业"].iloc[0]).strip()
    if not name:
        missing.append("industry_name")
        return None
    return name


def _industry_context(ticker: str, missing: list[str]) -> dict | None:
    """行业静态 PE 对照：按天向前回退，命中该股所属行业那一行。"""
    import akshare as ak

    name = _industry_name(ticker, missing)
    if name is None:
        return None
    today = date.today()
    last_error: Exception | None = None
    for offset in range(RETRY_DAYS):
        stamp = (today - timedelta(days=offset)).strftime("%Y%m%d")
        try:
            frame = fetch_with_retry(
                lambda stamp=stamp: ak.stock_industry_pe_ratio_cninfo(
                    symbol=INDUSTRY_SYMBOL, date=stamp
                ),
                label=f"stock_industry_pe_ratio_cninfo({stamp})",
            )
        except Exception as exc:  # noqa: BLE001 - 逐日回退
            last_error = exc
            continue
        if frame is None or len(frame) == 0 or "行业名称" not in frame.columns:
            last_error = RuntimeError("行业市盈率表结构不符")
            continue
        hit = frame[frame["行业名称"].astype(str).str.strip() == name]
        if len(hit) == 0:
            last_error = RuntimeError(f"行业市盈率表未包含行业: {name}")
            continue
        row = hit.iloc[0]
        return {
            "name": name,
            "code": str(row.get("行业编码", "")).strip(),
            "standard": str(row.get("行业分类", "")).strip(),
            "as_of": str(row.get("变动日期", "")).strip(),
            "companies": clean_value(row.get("公司数量")),
            "companies_in_calc": clean_value(row.get("纳入计算公司数量")),
            "pe_static_weighted": clean_value(row.get("静态市盈率-加权平均")),
            "pe_static_median": clean_value(row.get("静态市盈率-中位数")),
            "pe_static_mean": clean_value(row.get("静态市盈率-算术平均")),
        }
    missing.append("industry")
    if last_error is not None:
        missing.append(f"industry_last_error: {type(last_error).__name__}")
    return None


class AkshareValuationProvider:
    """真实实现：PE(TTM) / PB 近三年序列 + 中上协行业静态 PE 对照。"""

    source = SOURCE

    def get_valuation(self, ticker: str) -> dict:
        code = normalize_ticker(ticker) or ticker
        missing: list[str] = []

        pe = _describe(
            _valuation_series(code, PE_INDICATOR),
            label=PE_INDICATOR,
            caliber="TTM 口径，百度股市通近三年日频序列",
        )
        pb = _describe(
            _valuation_series(code, PB_INDICATOR),
            label=PB_INDICATOR,
            caliber="市净率口径，百度股市通近三年日频序列",
        )
        industry = _industry_context(code, missing)

        result = {
            "ticker": code,
            "source": SOURCE,
            "as_of": pe["end"],
            "window": {
                "label": PERIOD,
                "start": pe["start"],
                "end": pe["end"],
                "trading_days": pe["samples"],
                "definition": "窗口 = 近三年日频交易日；分位数 = 窗口内取值 ≤ 当前值的样本占比",
            },
            "metrics": {"pe_ttm": pe, "pb": pb},
            "industry": industry,
            "industry_caliber_note": (
                "行业 PE 为静态(LYR)口径、个股 PE 为 TTM 口径，两者口径不同，仅作量级对照"
            ),
            "missing_fields": sorted(missing),
        }
        return result
