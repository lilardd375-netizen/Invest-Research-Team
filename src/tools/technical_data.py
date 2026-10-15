"""数据工具层：技术面 Provider（由日线历史算统计指标）。

契约：provider.get_technical(ticker) 返回 **可 JSON 序列化的 dict**，指标全部由真实
日线推导：MA5 / MA20 / MA60、近 20 日涨跌幅、20 日区间高低点、量比（近 5 日均量 / 近 60 日均量）。

实测口径（akshare 1.19.1 + pandas 3.0.6，2026-10-15 实测 300750）：
- ak.stock_zh_a_hist(symbol, period="daily", start_date, end_date, adjust="qfq")
  参数名就是 start_date / end_date（写成 start= 会 TypeError）；
  列 ['日期','股票代码','开盘','收盘','最高','最低','成交量','成交额','振幅','涨跌幅','涨跌额','换手率']，
  实测 shape (187, 12)，日期列是**字符串** 'YYYY-MM-DD'（非 datetime），行序为日期升序。
- 稳定性陷阱：同一调用会连抛 ConnectionError（对端断连），实测第 3 次才成功，必须带重试。
- 行序不保证，本模块一律按日期重新排序后再算指标（沿用第四步的教训）。
"""

from datetime import date, timedelta
from typing import Protocol

import pandas as pd

from src.tools.utils import clean_value, fetch_with_retry, normalize_ticker

HISTORY_ROWS = 120  # 参与统计的交易日数（"近 120 日"）
FETCH_CALENDAR_DAYS = 300  # 向前取的自然日跨度，保证覆盖 120 个交易日
MIN_ROWS = 60  # 少于 60 个交易日就算不出 MA60 / 量比，直接判数据不可用
HIST_ATTEMPTS = 5  # 实测该接口会连抛 RemoteDisconnected：同一次探测里 3 连失败、第 4 次才成功
HIST_RETRY_DELAYS = (1.5, 3.0, 6.0, 12.0)  # 递增退避，总耐心 ~23s：实测对端突发限流会持续数十秒
MA_WINDOWS = (5, 20, 60)
CHANGE_DAYS = 20
RANGE_DAYS = 20
VOLUME_SHORT = 5
VOLUME_LONG = 60
SOURCE = "akshare:stock_zh_a_hist"


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


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    """列 -> float 序列（NaN/inf 置为 NaN，稍后统一 dropna）。"""
    return pd.to_numeric(frame[column].map(clean_value), errors="coerce")


class AkshareTechnicalProvider:
    """真实数据源：东财前复权日线 -> 均线 / 区间涨跌 / 量比。"""

    SOURCE = SOURCE

    def get_technical(self, ticker: str) -> dict:
        code = normalize_ticker(ticker)
        import akshare as ak  # 延迟导入：测试环境不装 akshare 也能 import 本模块

        end = date.today()
        start = end - timedelta(days=FETCH_CALENDAR_DAYS)
        frame = fetch_with_retry(
            lambda: ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust="qfq",
            ),
            label="stock_zh_a_hist",
            attempts=HIST_ATTEMPTS,
            delays=HIST_RETRY_DELAYS,
        )
        if frame is None or frame.empty:
            raise RuntimeError(f"stock_zh_a_hist 未返回 {code} 的日线数据")

        required = {"日期", "收盘", "最高", "最低", "成交量"}
        absent = sorted(required - set(frame.columns))
        if absent:
            raise RuntimeError(f"stock_zh_a_hist 结构与预期不符，缺少列: {absent}")

        bars = pd.DataFrame(
            {
                "date": frame["日期"].astype(str).str.strip(),
                "close": _series(frame, "收盘"),
                "high": _series(frame, "最高"),
                "low": _series(frame, "最低"),
                "volume": _series(frame, "成交量"),
            }
        ).dropna(subset=["close", "volume"])
        bars = bars[bars["date"] != ""].sort_values("date")  # 行序不保证 -> 显式排序
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
            "source": SOURCE,
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
