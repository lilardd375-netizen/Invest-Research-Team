"""数据工具层：行业板块 Provider。

契约：provider.get_industry(ticker) 返回 **可 JSON 序列化的 dict**，含三块信息：
1. 该股所属行业（eastmoney 行业层级的一级/二级/三级，对外口径取二级行业）；
2. 板块近期表现（涨跌幅 / 成交额 / 涨跌家数 / 领涨股）；
3. 同业 2-3 家关键指标对比（PE(TTM) / 市净率 / 总市值 / 当日涨跌幅）。

数据源与实测口径（akshare 1.19.1 + pandas 3.0.6，2026-10-15 实测 300750）：
- 行业归属与成分股走 eastmoney datacenter 报表 RPT_F10_CORETHEME_BOARDTYPE
  （akshare 未封装该报表；走 push2 的 stock_board_industry_cons_em 在本机经常被对端断连）；
  实测 300750 -> 电力设备(1200, 一级) / 电池(1033, 二级) / 锂电池(1303, 三级)，电池成分股 107 只。
- 板块行情优先用同花顺口径 ak.stock_board_industry_summary_ths()（实测 0.7s，板块名与 EM 口径一致），
  失败再退回 ak.stock_board_industry_name_em()；两者都拿不到就记进 missing_fields。
- 同业指标用 ak.stock_value_em(symbol)（每只约 2s），故只取 PEER_LIMIT 只。
"""

from typing import Protocol

import requests

from src.tools.utils import clean_value, fetch_with_retry, normalize_ticker

DATACENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
CORETHEME_REPORT = "RPT_F10_CORETHEME_BOARDTYPE"
HTTP_TIMEOUT = 20.0
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
INDUSTRY_BOARD_TYPE = "行业"
PEER_LIMIT = 3
BOARD_PERFORMANCE_SOURCE = "akshare:stock_board_industry_summary_ths"
BOARD_PERFORMANCE_FALLBACK = "akshare:stock_board_industry_name_em"
PEER_METRICS_SOURCE = "akshare:stock_value_em"
HIERARCHY_SOURCE = "eastmoney:RPT_F10_CORETHEME_BOARDTYPE"


class IndustryProvider(Protocol):
    """行业数据源协议。实现方需保证 get_industry 返回可 JSON 序列化的 dict。"""

    def get_industry(self, ticker: str) -> dict:  # pragma: no cover - 协议声明
        ...


class FakeIndustryProvider:
    """测试 / 离线用：构造时注入 ticker -> data 映射。"""

    def __init__(self, mapping: dict = None):
        self.mapping = mapping or {}
        self.calls: list[str] = []

    def get_industry(self, ticker: str) -> dict:
        self.calls.append(ticker)
        if ticker not in self.mapping:
            raise KeyError(f"未配置 ticker 的假数据: {ticker}")
        return self.mapping[ticker]


def _secucode(code: str) -> str:
    """6 位代码 -> eastmoney 的 SECUCODE（6/9 开头为沪市，其余深市）。"""
    return f"{code}.SH" if code.startswith(("6", "9")) else f"{code}.SZ"


def _coretheme_rows(filter_expr: str, page_size: int = 100) -> list[dict]:
    """查 eastmoney 行业/板块报表，返回 result.data 列表（失败抛 RuntimeError）。"""

    def _call():
        response = requests.get(
            DATACENTER_URL,
            params={
                "reportName": CORETHEME_REPORT,
                "columns": "ALL",
                "pageSize": str(page_size),
                "source": "HSF10",
                "client": "PC",
                "filter": filter_expr,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    payload = fetch_with_retry(_call, label=f"eastmoney 行业报表 {filter_expr}")
    result = payload.get("result") or {}
    return result.get("data") or []


def _rank_of(row: dict) -> int:
    try:
        return int(str(row.get("BOARD_RANK")).strip())
    except (TypeError, ValueError):
        return 0


def _pick_industry(hierarchy: list[dict]) -> dict:
    """从行业层级里选对外口径：优先二级行业，其次最深一级。"""
    for row in hierarchy:
        if _rank_of(row) == 2:
            return {"name": str(row.get("BOARD_NAME")).strip(), "code": str(row.get("BOARD_CODE")).strip(), "rank": 2}
    row = max(hierarchy, key=_rank_of)
    return {"name": str(row.get("BOARD_NAME")).strip(), "code": str(row.get("BOARD_CODE")).strip(), "rank": _rank_of(row)}


def _empty_board(industry: str) -> dict:
    return {
        "name": industry,
        "source": None,
        "change_pct": None,
        "amount": None,
        "net_inflow": None,
        "advancers": None,
        "decliners": None,
        "leader": None,
    }


def _board_performance(industry: str, missing: list[str]) -> dict:
    """板块近期表现：同花顺口径优先，退回 EM 板块行情，都失败则全空 + missing。"""
    board = _empty_board(industry)

    try:
        import akshare as ak  # 延迟导入：测试环境不装 akshare 也能 import 本模块

        summary = fetch_with_retry(
            lambda: ak.stock_board_industry_summary_ths(), label="stock_board_industry_summary_ths"
        )
        if summary is not None and not summary.empty and "板块" in summary.columns:
            hit = summary[summary["板块"].astype(str).str.strip() == industry]
            if not hit.empty:
                row = hit.iloc[0]
                board.update(
                    {
                        "source": BOARD_PERFORMANCE_SOURCE,
                        "change_pct": clean_value(row.get("涨跌幅")),
                        "amount": clean_value(row.get("总成交额")),
                        "net_inflow": clean_value(row.get("净流入")),
                        "advancers": clean_value(row.get("上涨家数")),
                        "decliners": clean_value(row.get("下跌家数")),
                        "leader": str(row.get("领涨股")).strip(),
                    }
                )
                return board
    except Exception as exc:  # noqa: BLE001
        missing.append(f"board(ths): {type(exc).__name__}")

    try:
        import akshare as ak

        listing = fetch_with_retry(
            lambda: ak.stock_board_industry_name_em(), label="stock_board_industry_name_em"
        )
        if listing is not None and not listing.empty and "板块名称" in listing.columns:
            hit = listing[listing["板块名称"].astype(str).str.strip() == industry]
            if not hit.empty:
                row = hit.iloc[0]
                board.update(
                    {
                        "source": BOARD_PERFORMANCE_FALLBACK,
                        "change_pct": clean_value(row.get("涨跌幅")),
                        "amount": clean_value(row.get("总成交额")),
                        "advancers": clean_value(row.get("上涨家数")),
                        "decliners": clean_value(row.get("下跌家数")),
                        "leader": str(row.get("领涨股票")).strip(),
                    }
                )
                return board
    except Exception as exc:  # noqa: BLE001
        missing.append(f"board(em): {type(exc).__name__}")

    for field in ("change_pct", "amount", "net_inflow", "advancers", "decliners", "leader"):
        missing.append(f"board.{field}")
    return board


def _peer_metrics(peer: dict) -> dict:
    """取单只同业的最新估值/市值指标（stock_value_em 的日期最新一行）。"""
    import akshare as ak

    frame = fetch_with_retry(
        lambda: ak.stock_value_em(symbol=peer["code"]), label=f"stock_value_em({peer['code']})"
    )
    if frame is None or frame.empty or "数据日期" not in frame.columns:
        raise RuntimeError("stock_value_em 未返回数据")
    row = frame.assign(_date=frame["数据日期"].astype(str)).sort_values("_date").iloc[-1]

    metrics = {}
    for column, field in (
        ("当日收盘价", "close"),
        ("当日涨跌幅", "change_pct"),
        ("PE(TTM)", "pe_ttm"),
        ("市净率", "pb"),
        ("总市值", "total_market_cap"),
    ):
        metrics[field] = clean_value(row.get(column))
    metrics["as_of"] = str(clean_value(row.get("数据日期")) or "").strip()[:10]
    return metrics


def _peers(board_code: str, self_code: str, missing: list[str]) -> tuple[list[dict], str]:
    """同业对比：板块成分股（剔除自己）里取前 PEER_LIMIT 只，逐只补指标。"""
    rows = _coretheme_rows(f'(BOARD_CODE="{board_code}")', page_size=500)
    candidates = [row for row in rows if str(row.get("SECURITY_CODE")).strip() != self_code]
    peers: list[dict] = []
    for row in candidates[:PEER_LIMIT]:
        peer = {
            "code": str(row.get("SECURITY_CODE")).strip(),
            "name": str(row.get("SECURITY_NAME_ABBR")).strip(),
            "as_of": None,
            "close": None,
            "change_pct": None,
            "pe_ttm": None,
            "pb": None,
            "total_market_cap": None,
        }
        try:
            peer.update(_peer_metrics(peer))
        except Exception as exc:  # noqa: BLE001
            missing.append(f"peers.{peer['code']}: {type(exc).__name__}")
        peers.append(peer)
    note = (
        f"成分股共 {len(rows)} 只，按报表返回顺序取前 {PEER_LIMIT} 只"
        "（该报表不含市值排序字段，故不是按市值挑选）"
    )
    return peers, note


class AkshareIndustryProvider:
    """真实数据源：eastmoney 行业报表 + 同花顺板块行情 + stock_value_em 估值。"""

    SOURCE = f"{HIERARCHY_SOURCE}+{BOARD_PERFORMANCE_SOURCE}+{PEER_METRICS_SOURCE}"

    def get_industry(self, ticker: str) -> dict:
        code = normalize_ticker(ticker)
        missing: list[str] = []

        hierarchy = _coretheme_rows(f'(SECUCODE="{_secucode(code)}")(BOARD_TYPE="{INDUSTRY_BOARD_TYPE}")')
        if not hierarchy:
            raise RuntimeError(f"eastmoney 行业报表未返回 {code} 的行业归属（可能不是 A 股标的）")
        levels = [
            {
                "name": str(row.get("BOARD_NAME")).strip(),
                "code": str(row.get("BOARD_CODE")).strip(),
                "rank": _rank_of(row),
            }
            for row in hierarchy
        ]
        target = _pick_industry(hierarchy)
        board = _board_performance(target["name"], missing)
        peers, peers_note = _peers(target["code"], code, missing)

        return {
            "ticker": code,
            "source": self.SOURCE,
            "industry": target["name"],
            "industry_code": target["code"],
            "industry_rank": target["rank"],
            "industry_levels": levels,
            "board": board,
            "peers": peers,
            "peers_note": peers_note,
            "missing_fields": sorted(set(missing)),
        }
