"""数据工具层：技术面 Provider（由日线历史算统计指标）。

契约：provider.get_technical(ticker) 返回 **可 JSON 序列化的 dict**，指标全部由真实
日线推导：MA5 / MA20 / MA60、近 20 日涨跌幅、20 日区间高低点、量比（近 5 日均量 / 近 60 日均量）。

双源容灾（第九步）：
- 主源 ak.stock_zh_a_daily（**新浪**，qfq）。实测（akshare 1.19.1，2026-10-16 内省 + 实测 300750）：
  签名 (symbol='sh603843', start_date='19900101', end_date='21000118', adjust='')，
  **symbol 必须带市场前缀**（'sz300750' 成功；'300750' 直接 KeyError: 'date'）；
  返回列是英文 ['date','open','high','low','close','volume','amount','outstanding_share','turnover']，
  shape (187, 9)，date 是字符串 'YYYY-MM-DD'、行序升序。
- fallback ak.stock_zh_a_hist（**东财**，qfq）：参数名是 start_date / end_date（写成 start= 会
  TypeError），列是中文 ['日期','股票代码','开盘','收盘','最高','最低','成交量',...]，
  实测会连抛 ConnectionError（RemoteDisconnected），是第七步 technical 四次冒烟都拿不到数据的根因。
- 两源复权因子口径有细微差异；对 MA / 涨跌幅 / 量比这类相对指标可接受，本模块**不做交叉校正**。
- 主源失败（含超时/重试耗尽/列不符）→ 自动走同等的超时与退避序列切 fallback；两源都失败才抛
  RuntimeError，由节点按既有路径降级 unavailable + analyst_errors。输出用 source / fallback_used /
  primary_error 显式标注实际口径，便于事后判断结论来自哪个源。
"""

from datetime import date, timedelta
from typing import Protocol

import pandas as pd

from src.tools.utils import clean_value, fetch_with_retry, normalize_ticker

HISTORY_ROWS = 120  # 参与统计的交易日数（"近 120 日"）
FETCH_CALENDAR_DAYS = 300  # 向前取的自然日跨度，保证覆盖 120 个交易日
MIN_ROWS = 60  # 少于 60 个交易日就算不出 MA60 / 量比，直接判数据不可用
HIST_ATTEMPTS = 5  # 实测东财该接口会连抛 RemoteDisconnected：同一次探测里 3 连失败、第 4 次才成功
HIST_RETRY_DELAYS = (1.5, 3.0, 6.0, 12.0)  # 递增退避，总耐心 ~23s：实测对端突发限流会持续数十秒
MA_WINDOWS = (5, 20, 60)
CHANGE_DAYS = 20
RANGE_DAYS = 20
VOLUME_SHORT = 5
VOLUME_LONG = 60
SINA_SOURCE = "akshare:stock_zh_a_daily(新浪,qfq)"
EM_SOURCE = "akshare:stock_zh_a_hist(东财,qfq)"
SOURCE = SINA_SOURCE  # 主源（向后兼容旧引用）
# 列映射：中文列名（东财） / 英文列名（新浪） -> 本模块统一列名
SINA_COLUMNS = {"date": "date", "close": "close", "high": "high", "low": "low", "volume": "volume"}
EM_COLUMNS = {"日期": "date", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume"}


class TechnicalProvider(Protocol):
    """技术面数据源协议。实现方需保证 get_technical 返回可 JSON 序列化的 dict。"""

    def get_technical(self, ticker: str) -> dict:  # pragma: no cover - 协议声明
        ...


class FakeTechnicalProvider:
    """测试 / 离线用：构造时注入 ticker -> data 映射。"""

    def __init__(self, mapping: dict = None):
        self.mapping = mapping or {}
        self.calls: list[str] = []

    def get_technical(self, ticker: str) -> dict:
        self.calls.append(ticker)
        if ticker not in self.mapping:
            raise KeyError(f"未配置 ticker 的假数据: {ticker}")
        return self.mapping[ticker]


def sina_symbol(code: str) -> str:
    """6 位代码 -> 新浪格式（沪市 sh / 深市 sz）；已带前缀的原样返回。"""
    text = str(code).strip().lower()
    if text.startswith(("sh", "sz", "bj")):
        return text
    return ("sh" if text.startswith(("6", "9")) else "sz") + text


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    """列 -> float 序列（NaN/inf 置为 NaN，稍后统一 dropna）。"""
    return pd.to_numeric(frame[column].map(clean_value), errors="coerce")


def _bars_from(frame: pd.DataFrame, mapping: dict, label: str, code: str) -> pd.DataFrame:
    """把任一源的原始 DataFrame 规范化成 [date, close, high, low, volume] 并排序。"""
    if frame is None or frame.empty:
        raise RuntimeError(f"{label} 未返回 {code} 的日线数据")
    absent = sorted(set(mapping) - set(frame.columns))
    if absent:
        raise RuntimeError(f"{label} 结构与预期不符，缺少列: {absent}")

    bars = pd.DataFrame(
        {
            canonical: (frame[column].astype(str).str.strip() if canonical == "date" else _series(frame, column))
            for column, canonical in mapping.items()
        }
    )
    bars = bars.dropna(subset=["close", "volume"])
    bars = bars[bars["date"] != ""].sort_values("date")  # 行序不保证 -> 显式排序
    return bars.reset_index(drop=True)


class AkshareTechnicalProvider:
    """真实数据源：新浪前复权日线为主，东财同口径日线为自动 fallback。"""

    SOURCE = SOURCE
    FALLBACK_SOURCE = EM_SOURCE

    def _fetch_sina(self, ak, code: str, start: str, end: str):
        fn = getattr(ak, "stock_zh_a_daily", None)
        if fn is None:
            raise RuntimeError("akshare 未提供 stock_zh_a_daily（新浪主源不可用）")
        return fetch_with_retry(
            lambda: fn(symbol=sina_symbol(code), start_date=start, end_date=end, adjust="qfq"),
            label="stock_zh_a_daily",
            attempts=HIST_ATTEMPTS,
            delays=HIST_RETRY_DELAYS,
        )

    def _fetch_eastmoney(self, ak, code: str, start: str, end: str):
        fn = getattr(ak, "stock_zh_a_hist", None)
        if fn is None:
            raise RuntimeError("akshare 未提供 stock_zh_a_hist（东财 fallback 不可用）")
        return fetch_with_retry(
            lambda: fn(
                symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq"
            ),
            label="stock_zh_a_hist",
            attempts=HIST_ATTEMPTS,
            delays=HIST_RETRY_DELAYS,
        )

    def get_technical(self, ticker: str) -> dict:
        code = normalize_ticker(ticker)
        import akshare as ak  # 延迟导入：测试环境不装 akshare 也能 import 本模块

        end = date.today()
        start = end - timedelta(days=FETCH_CALENDAR_DAYS)
        start_text = start.strftime("%Y%m%d")
        end_text = end.strftime("%Y%m%d")

        # 主源 -> fallback：任一环节失败（含重试耗尽、超时、列不符）都切下一源
        primary_error = None
        source = SINA_SOURCE
        try:
            raw = self._fetch_sina(ak, code, start_text, end_text)
            bars = _bars_from(raw, SINA_COLUMNS, "stock_zh_a_daily", code)
        except Exception as exc:  # noqa: BLE001 - 主源任何失败都要让位给 fallback
            primary_error = f"{type(exc).__name__}: {exc}"
            source = EM_SOURCE
            try:
                raw = self._fetch_eastmoney(ak, code, start_text, end_text)
                bars = _bars_from(raw, EM_COLUMNS, "stock_zh_a_hist", code)
            except Exception as fallback_exc:  # noqa: BLE001 - 双源都失败才降级
                raise RuntimeError(
                    f"日线数据双源均失败：主源 {SINA_SOURCE} -> {primary_error}；"
                    f"fallback {EM_SOURCE} -> {type(fallback_exc).__name__}: {fallback_exc}"
                ) from fallback_exc

        if len(bars) < MIN_ROWS:
            raise RuntimeError(
                f"日线数据不足 {MIN_ROWS} 个交易日（实际 {len(bars)}），无法计算 MA60 / 量比"
            )
        bars = bars.tail(HISTORY_ROWS).reset_index(drop=True)

        last = bars.iloc[-1]
        close = float(last["close"])
        metrics = {"close": close}
        missing: list[str] = []
        for window in MA_WINDOWS:
            value = clean_value(bars["close"].tail(window).mean())
            metrics[f"ma{window}"] = value
            if value is None:
                missing.append(f"ma{window}")

        if len(bars) > CHANGE_DAYS:
            base = float(bars["close"].iloc[-(CHANGE_DAYS + 1)])
            value = clean_value((close / base - 1) * 100) if base else None
        else:
            value = None
        metrics[f"change_{CHANGE_DAYS}d_pct"] = value
        if value is None:
            missing.append(f"change_{CHANGE_DAYS}d_pct")

        metrics["high_20d"] = clean_value(bars["high"].tail(RANGE_DAYS).max())
        metrics["low_20d"] = clean_value(bars["low"].tail(RANGE_DAYS).min())
        for field in ("high_20d", "low_20d"):
            if metrics[field] is None:
                missing.append(field)

        long_volume = float(bars["volume"].tail(VOLUME_LONG).mean())
        ratio = clean_value(float(bars["volume"].tail(VOLUME_SHORT).mean()) / long_volume) if long_volume else None
        metrics["volume_ratio_5_60"] = ratio
        if ratio is None:
            missing.append("volume_ratio_5_60")

        return {
            "ticker": code,
            "source": source,
            "fallback_used": source != SINA_SOURCE,
            "primary_error": primary_error,
            "adjust": "qfq",
            "as_of": str(last["date"])[:10],
            "window": {
                "rows": len(bars),
                "start": str(bars["date"].iloc[0])[:10],
                "end": str(last["date"])[:10],
                "metrics": (
                    f"MA{MA_WINDOWS[0]}/{MA_WINDOWS[1]}/{MA_WINDOWS[2]}、"
                    f"近 {CHANGE_DAYS} 日涨跌幅、{RANGE_DAYS} 日区间高低点、"
                    f"量比（近 {VOLUME_SHORT} 日均量 / 近 {VOLUME_LONG} 日均量）"
                ),
            },
            **metrics,
            "missing_fields": sorted(set(missing)),
        }
