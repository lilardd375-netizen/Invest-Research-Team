"""数据工具层：舆情（新闻）Provider。

契约：provider.get_sentiment(ticker) 返回 **可 JSON 序列化的 dict**：
最近 NEWS_LIMIT 条新闻，每条含标题、摘要（截断）、发布时间。

实测口径（akshare 1.19.1 + pandas 3.0.6，2026-10-15 实测 300750）：
- ak.stock_news_em(symbol="300750")，signature (symbol: str = '603777')，
  列 ['关键词','新闻标题','新闻内容','发布时间','文章来源','新闻链接']，shape (10, 6)，全 str dtype、0 NaN；
- **行序不是时间序**（实测最新一条落在 index 8，is_monotonic_desc=False、asc=False），
  必须自己按发布时间排序再截断，否则会把最旧的一条当最新；
- 发布时间是字符串 'YYYY-MM-DD HH:MM:SS'（不是 datetime），字典序即时间序。
"""

import math
from datetime import date
from typing import Protocol

from src.tools.utils import fetch_with_retry, normalize_ticker

NEWS_LIMIT = 10
TITLE_MAX_CHARS = 80
SUMMARY_MAX_CHARS = 120
SOURCE = "akshare:stock_news_em"


class SentimentProvider(Protocol):
    """舆情数据源协议。实现方需保证 get_sentiment 返回可 JSON 序列化的 dict。"""

    def get_sentiment(self, ticker: str) -> dict:  # pragma: no cover - 协议声明
        ...


class FakeSentimentProvider:
    """测试 / 离线用：构造时注入 ticker -> data 映射。"""

    def __init__(self, mapping: dict = None):
        self.mapping = mapping or {}
        self.calls: list[str] = []

    def get_sentiment(self, ticker: str) -> dict:
        self.calls.append(ticker)
        if ticker not in self.mapping:
            raise KeyError(f"未配置 ticker 的假数据: {ticker}")
        return self.mapping[ticker]


def _text(value) -> str:
    """表格单元格 -> 去空白字符串（None / NaN -> ""）。"""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _truncate(text: str, limit: int) -> str:
    """按字符截断并加省略号；空文本原样返回。"""
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


class AkshareSentimentProvider:
    """真实数据源：eastmoney 个股新闻搜索（最近 10 条）。"""

    SOURCE = SOURCE

    def get_sentiment(self, ticker: str) -> dict:
        code = normalize_ticker(ticker)
        import akshare as ak  # 延迟导入：测试环境不装 akshare 也能 import 本模块

        frame = fetch_with_retry(lambda: ak.stock_news_em(symbol=code), label="stock_news_em")

        if frame is None or frame.empty:
            return {
                "ticker": code,
                "source": SOURCE,
                "as_of": date.today().isoformat(),
                "count": 0,
                "news": [],
                "missing_fields": ["news"],
            }

        required = {"新闻标题", "新闻内容", "发布时间"}
        absent = sorted(required - set(frame.columns))
        if absent:
            raise RuntimeError(f"stock_news_em 结构与预期不符，缺少列: {absent}")

        records = [
            {
                "title": _truncate(_text(row.get("新闻标题")), TITLE_MAX_CHARS),
                "summary": _truncate(_text(row.get("新闻内容")), SUMMARY_MAX_CHARS),
                "published_at": _text(row.get("发布时间")),
            }
            for row in frame.to_dict("records")
        ]
        # 行序不是时间序：先按发布时间倒序，再取最近 NEWS_LIMIT 条
        records.sort(key=lambda item: item["published_at"], reverse=True)
        news = records[:NEWS_LIMIT]

        missing = []
        if not news:
            missing.append("news")
        if any(not item["title"] for item in news):
            missing.append("news.title")

        return {
            "ticker": code,
            "source": SOURCE,
            "as_of": date.today().isoformat(),
            "latest_published_at": news[0]["published_at"] if news else None,
            "count": len(news),
            "news": news,
            "missing_fields": missing,
        }
