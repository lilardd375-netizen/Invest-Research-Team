"""离线回归测试：第七步三个新 Provider 的清洗与口径（假 akshare 模块，不联网）。

锁定的都是真实探测里踩到的坑：
- stock_news_em 行序不是时间序（最新一条落在中间）→ 必须自己按发布时间倒序；
- stock_zh_a_hist 行序不保证、日期是字符串、收盘 NaN → 必须排序 + dropna；
- 字段名/结构变动 → 必须显式报错而不是静默产出错结论；
- 板块表现优先同花顺口径，同业取板块成分股（剔除自己）；
- 日线双源容灾（第九步）：新浪 stock_zh_a_daily 为主（symbol 要带 sz/sh 前缀、英文列名），
  东财 stock_zh_a_hist 为自动 fallback，两源都失败才降级 unavailable。
"""

import json
import os
import sys
import threading
import types
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
import pytest

from src.graph.builder import build_graph
from src.tools import industry_data, utils
from src.tools.industry_data import AkshareIndustryProvider, _pick_industry, _secucode
from src.tools.sentiment_data import AkshareSentimentProvider
from src.tools.technical_data import AkshareTechnicalProvider


def _fake_akshare(monkeypatch, **attrs):
    """把假 akshare 模块塞进 sys.modules（Provider 内部是延迟导入）。"""
    module = types.SimpleNamespace(**attrs)
    monkeypatch.setitem(sys.modules, "akshare", module)
    return module


# --------------------------------------------------------------------------- 舆情


def _news_frame(rows):
    return pd.DataFrame(
        rows, columns=["关键词", "新闻标题", "新闻内容", "发布时间", "文章来源", "新闻链接"]
    )


def test_sentiment_sorts_desc_and_limits(monkeypatch):
    rows = [
        [f"300750", f"标题{i}", "内容" * 3, f"2026-10-{day:02d} 09:00:00", "财联社", "u"]
        for i, day in enumerate([1, 11, 3, 12, 5, 13, 7, 14, 9, 15, 2, 4])
    ]
    _fake_akshare(monkeypatch, stock_news_em=lambda symbol: _news_frame(rows))

    data = AkshareSentimentProvider().get_sentiment("300750")

    assert data["count"] == 10
    stamps = [item["published_at"] for item in data["news"]]
    assert stamps == sorted(stamps, reverse=True)
    assert stamps[0] == "2026-10-15 09:00:00"
    assert data["latest_published_at"] == "2026-10-15 09:00:00"
    assert json.dumps(data, ensure_ascii=False)


def test_sentiment_truncates_and_flags_missing_title(monkeypatch):
    rows = [
        ["300750", "标" * 100, "内" * 300, "2026-10-15 09:00:00", "财联社", "u"],
        ["300750", np.nan, "内容", "2026-10-14 09:00:00", "财联社", "u"],
    ]
    _fake_akshare(monkeypatch, stock_news_em=lambda symbol: _news_frame(rows))

    data = AkshareSentimentProvider().get_sentiment("300750")

    assert data["news"][0]["title"] == "标" * 80 + "…"
    assert data["news"][0]["summary"] == "内" * 120 + "…"
    assert data["news"][1]["title"] == ""
    assert "news.title" in data["missing_fields"]


def test_sentiment_missing_columns_raises(monkeypatch):
    frame = pd.DataFrame([{"新闻标题": "标题"}])
    _fake_akshare(monkeypatch, stock_news_em=lambda symbol: frame)

    with pytest.raises(RuntimeError, match="结构"):
        AkshareSentimentProvider().get_sentiment("300750")


def test_sentiment_empty_frame(monkeypatch):
    _fake_akshare(monkeypatch, stock_news_em=lambda symbol: _news_frame([]))

    data = AkshareSentimentProvider().get_sentiment("300750")

    assert data["count"] == 0
    assert data["news"] == []
    assert data["missing_fields"] == ["news"]


# --------------------------------------------------------------------------- 技术面


def _hist_frame(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "日期",
            "股票代码",
            "开盘",
            "收盘",
            "最高",
            "最低",
            "成交量",
            "成交额",
            "振幅",
            "涨跌幅",
            "涨跌额",
            "换手率",
        ],
    )


def _bars(count=130, start_day=1):
    rows = []
    for index in range(count):
        day = start_day + index
        close = 10.0 + index * 0.1
        rows.append(
            [
                f"2026-01-{day:02d}" if day <= 31 else f"2026-02-{day - 31:02d}",
                "300750",
                close,
                close,
                close + 1.0,
                close - 1.0,
                1000.0 + index,
                1.0,
                1.0,
                1.0,
                0.0,
                1.0,
            ]
        )
    return rows


def test_technical_sorts_and_computes(monkeypatch):
    rows = _bars()
    shuffled = rows[60:] + rows[:60]  # 行序乱掉：必须自己排序
    _fake_akshare(monkeypatch, stock_zh_a_hist=lambda **kwargs: _hist_frame(shuffled))

    data = AkshareTechnicalProvider().get_technical("300750")

    closes = [row[3] for row in rows][-5:]
    volumes = [row[6] for row in rows]
    assert data["window"]["rows"] == 120
    assert data["as_of"] == rows[-1][0]
    assert data["ma5"] == pytest.approx(sum(closes) / 5)
    assert data["ma60"] == pytest.approx(sum(row[3] for row in rows[-60:]) / 60)
    assert data["high_20d"] == pytest.approx(rows[-1][4])
    assert data["low_20d"] == pytest.approx(rows[-20][5])
    assert data["volume_ratio_5_60"] == pytest.approx(
        sum(volumes[-5:]) / 5 / (sum(volumes[-60:]) / 60)
    )
    assert data["missing_fields"] == []
    assert json.dumps(data, ensure_ascii=False)


def test_technical_drops_nan_close(monkeypatch):
    rows = _bars()
    rows[10][3] = np.nan
    _fake_akshare(monkeypatch, stock_zh_a_hist=lambda **kwargs: _hist_frame(rows))

    data = AkshareTechnicalProvider().get_technical("300750")

    assert data["window"]["rows"] == 120
    assert data["ma5"] is not None


def test_technical_too_few_rows_raises(monkeypatch):
    _fake_akshare(monkeypatch, stock_zh_a_hist=lambda **kwargs: _hist_frame(_bars(30)))

    with pytest.raises(RuntimeError, match="不足"):
        AkshareTechnicalProvider().get_technical("300750")


def test_technical_missing_columns_raises(monkeypatch):
    frame = _hist_frame(_bars()).drop(columns=["成交量"])
    _fake_akshare(monkeypatch, stock_zh_a_hist=lambda **kwargs: frame)

    with pytest.raises(RuntimeError, match="结构"):
        AkshareTechnicalProvider().get_technical("300750")


def _sina_frame(rows):
    """新浪日线帧：英文列名（实测 9 列），symbol 需要 sz/sh 前缀。"""
    return pd.DataFrame(
        [
            {
                "date": row[0],
                "open": row[2],
                "high": row[4],
                "low": row[5],
                "close": row[3],
                "volume": row[6],
                "amount": 1.0,
                "outstanding_share": 1.0,
                "turnover": 0.01,
            }
            for row in rows
        ]
    )


def test_technical_prefers_sina_primary(monkeypatch):
    """主源（新浪）可用时不碰东财，并在输出里标注实际口径。"""
    rows = _bars()
    symbols: list[str] = []

    def sina_daily(**kwargs):
        symbols.append(kwargs["symbol"])
        return _sina_frame(rows)

    _fake_akshare(
        monkeypatch,
        stock_zh_a_daily=sina_daily,
        stock_zh_a_hist=lambda **kwargs: pytest.fail("主源可用时不应调用东财 fallback"),
    )

    data = AkshareTechnicalProvider().get_technical("300750")

    assert symbols == ["sz300750"]  # 新浪要求带市场前缀（实测 '300750' 会 KeyError）
    assert data["source"] == "akshare:stock_zh_a_daily(新浪,qfq)"
    assert data["fallback_used"] is False
    assert data["primary_error"] is None
    assert data["window"]["rows"] == 120
    assert data["as_of"] == rows[-1][0]
    assert data["ma5"] == pytest.approx(sum(row[3] for row in rows[-5:]) / 5)
    assert json.dumps(data, ensure_ascii=False)


def test_technical_falls_back_to_eastmoney(monkeypatch):
    """主源连抛断连（第七步实测就是这样）→ 自动切东财，并记录主源错误。"""
    monkeypatch.setattr(utils.time, "sleep", lambda _seconds: None)  # 跳过主源 23s 退避，别真等
    rows = _bars()

    def sina_down(**kwargs):
        raise ConnectionError("Remote end closed connection without response")

    _fake_akshare(
        monkeypatch,
        stock_zh_a_daily=sina_down,
        stock_zh_a_hist=lambda **kwargs: _hist_frame(rows),
    )

    data = AkshareTechnicalProvider().get_technical("300750")

    assert data["source"] == "akshare:stock_zh_a_hist(东财,qfq)"
    assert data["fallback_used"] is True
    assert "ConnectionError" in data["primary_error"]
    assert data["window"]["rows"] == 120
    assert data["ma60"] == pytest.approx(sum(row[3] for row in rows[-60:]) / 60)


def test_technical_both_sources_fail_degrades_to_unavailable(monkeypatch):
    """双源都失败 → 抛 RuntimeError；节点按既有路径降级，且不把空数据喂给 LLM。"""
    monkeypatch.setattr(utils.time, "sleep", lambda _seconds: None)  # 别真等 23s 退避

    def down(**kwargs):
        raise ConnectionError("Remote end closed connection without response")

    _fake_akshare(monkeypatch, stock_zh_a_daily=down, stock_zh_a_hist=down)

    with pytest.raises(RuntimeError, match="双源均失败"):
        AkshareTechnicalProvider().get_technical("300750")

    llm = _FakeLLM()
    app = build_graph(technical_llm=llm, technical_provider=AkshareTechnicalProvider())
    final = app.invoke({"query": "宁德时代 300750 的投资价值"})

    payload = final["analyses"]["technical_analyst"]
    assert payload["status"] == "unavailable"
    assert payload["error_kind"] == "data_source"
    assert "双源均失败" in final["analyst_errors"]["technical_analyst"]
    assert llm.calls == 0


# --------------------------------------------------------------------------- 行业


HIERARCHY = [
    {"BOARD_CODE": "1200", "BOARD_NAME": "电力设备", "BOARD_RANK": "1", "BOARD_LEVEL": "1"},
    {"BOARD_CODE": "1303", "BOARD_NAME": "锂电池", "BOARD_RANK": "3", "BOARD_LEVEL": "3"},
    {"BOARD_CODE": "1033", "BOARD_NAME": "电池", "BOARD_RANK": "2", "BOARD_LEVEL": "2"},
]
MEMBERS = [
    {"SECURITY_CODE": "300750", "SECURITY_NAME_ABBR": "宁德时代"},
    {"SECURITY_CODE": "002466", "SECURITY_NAME_ABBR": "天齐锂业"},
    {"SECURITY_CODE": "300390", "SECURITY_NAME_ABBR": "天华新能"},
    {"SECURITY_CODE": "300409", "SECURITY_NAME_ABBR": "道氏技术"},
    {"SECURITY_CODE": "301238", "SECURITY_NAME_ABBR": "瑞泰新材"},
]


def test_secucode_suffix():
    assert _secucode("300750").endswith(".SZ")
    assert _secucode("600519").endswith(".SH")


def test_pick_industry_prefers_rank2():
    picked = _pick_industry(HIERARCHY)

    assert (picked["name"], picked["code"], picked["rank"]) == ("电池", "1033", 2)


def test_pick_industry_falls_back_to_deepest():
    picked = _pick_industry([HIERARCHY[0], HIERARCHY[1]])

    assert (picked["name"], picked["rank"]) == ("锂电池", 3)


def test_industry_full_path_with_stubbed_datacenter(monkeypatch):
    """整条路径：datacenter 层级/成分股打桩 + 同花顺板块行情 + stock_value_em 估值。"""

    def fake_rows(filter_expr, page_size=100):
        if "BOARD_CODE" in filter_expr:
            return MEMBERS
        return HIERARCHY

    monkeypatch.setattr(industry_data, "_coretheme_rows", fake_rows)
    summary = pd.DataFrame(
        [
            {
                "序号": 1,
                "板块": "电池",
                "涨跌幅": -2.87,
                "总成交量": 1.0,
                "总成交额": 100.5,
                "净流入": -8.1,
                "上涨家数": 20,
                "下跌家数": 80,
                "均价": 1.0,
                "领涨股": "宁德时代",
                "领涨股-最新价": 1.0,
                "领涨股-涨跌幅": 1.0,
            }
        ]
    )
    valuation = pd.DataFrame(
        [
            {
                "数据日期": "2026-10-13",
                "当日收盘价": 280.0,
                "当日涨跌幅": -1.0,
                "总市值": 1.2e12,
                "PE(TTM)": 15.0,
                "市净率": 3.4,
            },
            {
                "数据日期": "2026-10-14",
                "当日收盘价": 280.12,
                "当日涨跌幅": -0.9,
                "总市值": 1.3e12,
                "PE(TTM)": 15.249211,
                "市净率": 3.416825,
            },
        ]
    )
    _fake_akshare(
        monkeypatch,
        stock_board_industry_summary_ths=lambda: summary,
        stock_value_em=lambda symbol: valuation,
    )

    data = AkshareIndustryProvider().get_industry("300750")

    assert data["industry"] == "电池"
    assert data["industry_code"] == "1033"
    assert data["board"]["source"] == "akshare:stock_board_industry_summary_ths"
    assert data["board"]["change_pct"] == pytest.approx(-2.87)
    assert data["board"]["leader"] == "宁德时代"
    assert len(data["peers"]) == 3
    assert "300750" not in [peer["code"] for peer in data["peers"]]
    assert data["peers"][0]["code"] == "002466"
    assert data["peers"][0]["pe_ttm"] == pytest.approx(15.249211)
    assert data["peers"][0]["as_of"] == "2026-10-14"
    assert json.dumps(data, ensure_ascii=False)


# --------------------------------------------------------------------------- 超时兜底


HANG = threading.Event()


def _hang_news(symbol):
    """模拟 akshare 调用卡死：用 Event 而不是 time.sleep，避免被重试间隔的补丁一起清掉。"""
    HANG.wait(1.0)
    return _news_frame([])


class _FakeLLM:
    """假 LLM：本组测试只关心它在数据源超时的路径上有没有被调用。"""

    def __init__(self):
        self.calls = 0

    def chat(self, system, user):
        self.calls += 1
        return (
            '{"agent": "sentiment_analyst", "conclusion": "中性", "confidence": "低",'
            ' "key_points": [], "concerns": [], "assumptions": []}'
        )


def test_call_with_timeout_raises():
    with pytest.raises(TimeoutError, match="未返回"):
        utils.call_with_timeout(lambda: HANG.wait(1.0), seconds=0.05, label="假接口")


def test_fetch_with_retry_raises_timeout(monkeypatch):
    monkeypatch.setattr(utils.time, "sleep", lambda _seconds: None)  # 跳过重试间隔，别真等

    with pytest.raises(TimeoutError, match="连续 2 次超时"):
        utils.fetch_with_retry(
            lambda: HANG.wait(1.0), attempts=2, delay=1.0, label="假接口", timeout=0.05
        )


def test_provider_timeout_degrades_to_unavailable(monkeypatch):
    """akshare 卡死 -> 硬超时 -> 节点降级 unavailable，且不把空数据喂给 LLM。"""
    monkeypatch.setattr(utils, "DEFAULT_TIMEOUT", 0.05)
    monkeypatch.setattr(utils.time, "sleep", lambda _seconds: None)
    _fake_akshare(monkeypatch, stock_news_em=_hang_news)

    llm = _FakeLLM()
    app = build_graph(sentiment_llm=llm, sentiment_provider=AkshareSentimentProvider())
    final = app.invoke({"query": "宁德时代 300750 的投资价值"})

    payload = final["analyses"]["sentiment_analyst"]
    assert payload["status"] == "unavailable"
    assert payload["error_kind"] == "data_source"
    assert "超时" in final["analyst_errors"]["sentiment_analyst"]
    assert llm.calls == 0


# --------------------------------------------------------------------------- 退避重试


def test_technical_backoff_sequence(monkeypatch):
    """日线接口实测会突发限流：重试间隔必须是 1.5/3/6/12 递增序列（记录 sleep，不真等）。"""
    sleeps: list[float] = []
    monkeypatch.setattr(utils.time, "sleep", sleeps.append)

    def always_disconnected(**kwargs):
        raise ConnectionError("Remote end closed connection without response")

    _fake_akshare(monkeypatch, stock_zh_a_hist=always_disconnected)

    with pytest.raises(RuntimeError, match="连续 5 次调用失败"):
        AkshareTechnicalProvider().get_technical("300750")

    assert sleeps == [1.5, 3.0, 6.0, 12.0]


# --------------------------------------------------------------------------- ASCII CA 引导


def test_ascii_ca_dirs_keeps_only_ascii(monkeypatch):
    """候选目录只保留 ASCII 路径，且环境变量覆盖项排第一。"""
    monkeypatch.setenv(utils.CA_DIR_ENV, "D:\\irt_ca_test")

    dirs = utils._ascii_ca_dirs(Path("C:\\中文目录\\cacert.pem"))

    assert dirs[0] == Path("D:\\irt_ca_test")
    assert dirs
    assert all(str(item).isascii() for item in dirs)


def test_ensure_ascii_ca_copies_when_path_not_ascii(tmp_path, monkeypatch):
    """证书在中文路径下 -> 复制到 ASCII 目录并设置三个 CA 环境变量，重复调用幂等。"""
    source = tmp_path / "中文目录" / "cacert.pem"
    source.parent.mkdir()
    source.write_text("CA-BUNDLE", encoding="utf-8")
    if str(source).isascii():  # 本机临时目录恰好是 ASCII 时该分支不成立
        pytest.skip("临时目录本身已是 ASCII 路径")

    target_dir = tmp_path / "ascii_ca"
    saved = {name: os.environ.get(name) for name in utils._CA_ENV_VARS}
    monkeypatch.delenv(utils.CA_SKIP_ENV, raising=False)
    monkeypatch.setattr(utils, "_CA_READY", False)
    monkeypatch.setattr(utils, "CA_ASCII_PATH", None)
    monkeypatch.setattr(utils, "_ascii_ca_dirs", lambda _source: [target_dir])
    monkeypatch.setattr(certifi, "where", lambda: str(source))
    try:
        first = utils.ensure_ascii_ca()
        second = utils.ensure_ascii_ca()

        assert first == str(target_dir / "cacert.pem")
        assert Path(first).read_text(encoding="utf-8") == "CA-BUNDLE"
        assert utils.CA_ASCII_PATH == first
        for name in utils._CA_ENV_VARS:
            assert os.environ[name] == first
        assert second == first
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

