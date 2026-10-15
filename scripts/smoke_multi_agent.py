"""第七步真实冒烟（两阶段，日志文件友好）：四数据源 + 四分析师并行 + 质询回传。

推荐跑法（分钟级真实冒烟一律写日志文件，避免 tool call 中断丢掉全过程）：

    .\.venv\Scripts\python.exe -u -X utf8 scripts\smoke_multi_agent.py --stage 1 > _smoke7_stage1.log 2>&1
    .\.venv\Scripts\python.exe -u -X utf8 scripts\smoke_multi_agent.py --stage 2 > _smoke7_stage2.log 2>&1

- 每行输出都带 `[HH:MM:SS +Ns]` 时间戳（进程启动起算），便于事后从日志恢复各节点进出时间；
- `--stage 1` 只做纯数据取数（不调 LLM），并把结果快照写到 --snapshot（默认 _smoke7_snapshot.json）；
- `--stage 2` 读该快照跑全图（不碰 akshare），真实 key 只从环境变量 / .env 读取；
- 结束标记：`SMOKE STAGE1 DONE` / `SMOKE STAGE2 DONE`（看到它才说明整段跑完）；
- 输出出现 key 明文即 exit 2。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 允许直接 python scripts/xxx.py

from src.graph.builder import build_graph
from src.llm.client import DeepSeekClient
from src.tools.financial_data import AkshareProvider
from src.tools.industry_data import AkshareIndustryProvider
from src.tools.sentiment_data import AkshareSentimentProvider
from src.tools.technical_data import AkshareTechnicalProvider

TICKER = "300750"
QUERY = "宁德时代 300750 的投资价值"
AGENTS = ("financial_analyst", "industry_analyst", "sentiment_analyst", "technical_analyst")
DEFAULT_SNAPSHOT = "_smoke7_snapshot.json"
START = time.time()
OUTPUT: list[str] = []


def emit(text: str = "") -> None:
    """打印 + 留档。每行都带墙钟时间与进程内耗时，便于从日志文件恢复全过程。"""
    line = f"[{datetime.now():%H:%M:%S} +{time.time() - START:6.1f}s] {text}"
    print(line, flush=True)
    OUTPUT.append(line)


class CountingLLM:
    """透明代理：统计某个节点真实调用 LLM 的次数。"""

    def __init__(self, inner, label: str):
        self.inner = inner
        self.label = label
        self.calls = 0

    def chat(self, system: str, user: str) -> str:
        self.calls += 1
        return self.inner.chat(system, user)

    def plan_outline(self, query: str) -> str:
        self.calls += 1
        return self.inner.plan_outline(query)


class SnapshotProvider:
    """阶段二用：把阶段一取到的数据喂给图，阶段二完全不碰 akshare。"""

    def __init__(self, data: dict | None, label: str):
        self.data = data
        self.label = label
        self.calls = 0

    def _fetch(self, ticker: str) -> dict:
        self.calls += 1
        if self.data is None:
            raise RuntimeError(f"阶段一未取得 {self.label} 数据（快照缺失）")
        return self.data

    def get_financials(self, ticker: str) -> dict:
        return self._fetch(ticker)

    def get_industry(self, ticker: str) -> dict:
        return self._fetch(ticker)

    def get_sentiment(self, ticker: str) -> dict:
        return self._fetch(ticker)

    def get_technical(self, ticker: str) -> dict:
        return self._fetch(ticker)


def _metric(data: dict, name: str):
    value = data.get(name)
    return value.get("value") if isinstance(value, dict) else value


def _summarize_financial(data: dict) -> list[str]:
    return [
        f"报告期 {data.get('report_period')} / as_of {data.get('as_of')} / source {data.get('source')}",
        "roe={roe} net_margin={net_margin} debt_ratio={debt_ratio} revenue_growth={revenue_growth}".format(
            roe=_metric(data, "roe"),
            net_margin=_metric(data, "net_margin"),
            debt_ratio=_metric(data, "debt_ratio"),
            revenue_growth=_metric(data, "revenue_growth"),
        ),
        f"missing_metrics={data.get('missing_metrics')}",
    ]


def _summarize_industry(data: dict) -> list[str]:
    board = data.get("board") or {}
    peers = data.get("peers") or []
    return [
        f"行业 {data.get('industry')}({data.get('industry_code')}) rank={data.get('industry_rank')} "
        f"层级 {[item.get('name') for item in (data.get('industry_levels') or [])]}",
        f"板块: 涨跌幅={board.get('change_pct')} 成交额={board.get('amount')} 领涨={board.get('leader')} "
        f"source={board.get('source')}",
        "同业: " + "；".join(
            f"{peer.get('code')}/{peer.get('name')} PE(TTM)={peer.get('pe_ttm')} PB={peer.get('pb')}" for peer in peers
        ),
        f"missing_fields={data.get('missing_fields')}",
    ]


def _summarize_sentiment(data: dict) -> list[str]:
    news = data.get("news") or []
    return [
        f"新闻 {data.get('count')} 条 / 最新 {data.get('latest_published_at')} / source {data.get('source')}",
        f"首条: {news[0].get('title') if news else '(无)'}",
        f"首条摘要: {(news[0].get('summary') if news else '')[:60]}…",
        f"missing_fields={data.get('missing_fields')}",
    ]


def _summarize_technical(data: dict) -> list[str]:
    window = data.get("window") or {}
    return [
        f"截至 {data.get('as_of')} / {window.get('rows')} 个交易日（{window.get('start')} ~ {window.get('end')}）"
        f" / adjust={data.get('adjust')}",
        f"MA5={data.get('ma5')} MA20={data.get('ma20')} MA60={data.get('ma60')} 收盘={data.get('close')}",
        f"近20日涨跌={data.get('change_20d_pct')}% 区间 {data.get('low_20d')}~{data.get('high_20d')} "
        f"量比(5/60)={data.get('volume_ratio_5_60')}",
        f"missing_fields={data.get('missing_fields')}",
    ]


def read_snapshot(path: Path) -> dict:
    if not path.is_file():
        emit(f"[!] 快照文件不存在：{path}（阶段二四个数据源都会降级 unavailable）")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        emit(f"[!] 快照文件读取失败：{type(exc).__name__}: {exc}")
        return {}
    emit(f"快照载入 {path}：{sorted(data)}（抓取时间 {data.get('_fetched_at', '未知')}）")
    return data


def write_snapshot(path: Path, payload: dict) -> None:
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        emit(f"快照已写入 {path}（{len(json.dumps(payload, ensure_ascii=False))} 字符）")
    except OSError as exc:
        emit(f"[!] 快照写入失败：{type(exc).__name__}: {exc}")


def stage1(snapshot_path: Path) -> dict:
    emit("=" * 78)
    emit("[阶段一] 四个数据源真实取数（不调 LLM）")
    sources = [
        ("financial", AkshareProvider(), "get_financials", _summarize_financial),
        ("industry", AkshareIndustryProvider(), "get_industry", _summarize_industry),
        ("sentiment", AkshareSentimentProvider(), "get_sentiment", _summarize_sentiment),
        ("technical", AkshareTechnicalProvider(), "get_technical", _summarize_technical),
    ]
    results: dict[str, dict] = {}
    for label, provider, method, summarize in sources:
        emit(f"进入 {label} 取数")
        started = time.time()
        try:
            data = getattr(provider, method)(TICKER)
        except Exception as exc:  # noqa: BLE001 - 单源失败不阻塞其余
            emit(f"[FAIL] {label} 退出：失败（{time.time() - started:.1f}s）: {type(exc).__name__}: {exc}")
            continue
        results[label] = data
        size = len(json.dumps(data, ensure_ascii=False, default=str))
        emit(f"{label} 退出：[OK] 耗时 {time.time() - started:.1f}s / JSON {size} 字符")
        for line in summarize(data):
            emit(f"     {line}")
    emit(f"阶段一小结：成功 {len(results)}/4 -> {sorted(results)}")
    write_snapshot(snapshot_path, {**results, "_fetched_at": datetime.now().isoformat(timespec="seconds")})
    emit(f"SMOKE STAGE1 DONE 成功 {len(results)}/4")
    return results


def _progress_tracker():
    """把状态快照变成"节点进入/退出 + severity"进度行（按变化打印，日志可还原全过程）。"""
    seen = {"outline": False, "analysts": set(), "all_analysts": False, "review": None, "round": 0, "done": False}

    def track(snapshot: dict) -> None:
        analyses = snapshot.get("analyses") or {}
        if not seen["outline"] and snapshot.get("outline"):
            seen["outline"] = True
            sections = (snapshot.get("outline") or {}).get("sections") or []
            emit(f"planner 退出：{len(sections)} 个板块")
            emit("进入四分析师 fan-out（financial / industry / sentiment / technical）")
        for agent in AGENTS:
            if agent in analyses and agent not in seen["analysts"]:
                seen["analysts"].add(agent)
                payload = analyses[agent]
                state = payload.get("conclusion") or payload.get("status")
                emit(f"{agent} 进入→退出：{state}")
        if not seen["all_analysts"] and len(seen["analysts"]) == len(AGENTS):
            seen["all_analysts"] = True
            emit("四分析师 退出：全部完成（fan-in 到 risk_review）")
        review = snapshot.get("risk_review")
        if review and review != seen["review"]:
            seen["review"] = review
            challenges = review.get("challenges") or []
            targets: dict[str, int] = {}
            for challenge in challenges:
                targets[challenge.get("target_agent")] = targets.get(challenge.get("target_agent"), 0) + 1
            emit(
                f"risk_review 退出：severity={review.get('severity')} 质询 {len(challenges)} 条 "
                f"分布={json.dumps(targets, ensure_ascii=False)}"
            )
        round_now = snapshot.get("revision_round", 0)
        if round_now > seen["round"]:
            seen["round"] = round_now
            emit(f"revise 退出：已分发质询，进入第 {round_now} 轮返工")
            emit(f"进入四分析师 fan-out（第 {round_now} 轮）")
        if snapshot.get("final_report") and not seen["done"]:
            seen["done"] = True
            emit("finalize 退出")

    return track


def stage2(snapshot: dict) -> int:
    emit("\n" + "=" * 78)
    emit("[阶段二] 全图真实链路（真实 key + 阶段一数据快照，阶段二不碰 akshare）")
    emit(f"密钥: DEEPSEEK_API_KEY(已配置) len={len(os.environ.get('DEEPSEEK_API_KEY', ''))}")

    real = DeepSeekClient()
    planner = CountingLLM(real, "planner")
    llms = {agent: CountingLLM(real, agent) for agent in AGENTS}
    risk_llm = CountingLLM(real, "risk_review")
    providers = {
        "financial_provider": SnapshotProvider(snapshot.get("financial"), "financial"),
        "industry_provider": SnapshotProvider(snapshot.get("industry"), "industry"),
        "sentiment_provider": SnapshotProvider(snapshot.get("sentiment"), "sentiment"),
        "technical_provider": SnapshotProvider(snapshot.get("technical"), "technical"),
    }

    app = build_graph(
        planner_llm=planner,
        financial_llm=llms["financial_analyst"],
        industry_llm=llms["industry_analyst"],
        sentiment_llm=llms["sentiment_analyst"],
        technical_llm=llms["technical_analyst"],
        risk_llm=risk_llm,
        **providers,
    )

    track = _progress_tracker()
    emit("进入 planner")
    snapshots = []
    for state in app.stream({"query": QUERY}, stream_mode="values"):
        snapshots.append(state)
        track(state)
    final = snapshots[-1]

    outline = final.get("outline") or {}
    emit(f"[1] 大纲：status={outline.get('status')} sections={len(outline.get('sections') or [])}")

    seen_review = None
    round_index = 0
    for snap in snapshots:
        review = snap.get("risk_review")
        if not review or review == seen_review:
            continue
        seen_review = review
        round_index += 1
        challenges = review.get("challenges") or []
        distribution: dict[str, int] = {}
        for challenge in challenges:
            target = challenge.get("target_agent")
            distribution[target] = distribution.get(target, 0) + 1
        emit(
            f"[2] 第 {round_index} 次复核：severity={review.get('severity')} "
            f"质询 {len(challenges)} 条 分布={json.dumps(distribution, ensure_ascii=False)} "
            f"revision_round={snap.get('revision_round', 0)}"
        )
        emit(f"    summary: {review.get('summary')}")
        for challenge in challenges:
            emit(f"    - [{challenge.get('target_agent')}] {challenge.get('issue')}")
        for agent in AGENTS:
            payload = (snap.get("analyses") or {}).get(agent)
            if payload:
                emit(f"    {_agent_line(agent, payload)}")

    analyses = final.get("analyses") or {}
    emit("[3] 最终四分析师结果：")
    for agent in AGENTS:
        payload = analyses.get(agent)
        emit(f"    {_agent_line(agent, payload) if payload else f'{agent}: 无结果'}")

    emit("[4] 循环与最终路径：")
    emit(f"    返工轮次 revision_round={final.get('revision_round', 0)}")
    emit(f"    round_at_last_review={final.get('round_at_last_review')}")
    emit(f"    final_report={final.get('final_report')!r}")
    emit(f"    planner_error={final.get('planner_error')!r}")
    emit(f"    risk_error={final.get('risk_error')!r}")
    emit(f"    analyst_errors={final.get('analyst_errors')!r}")
    emit(f"    analyst_error={final.get('analyst_error')!r}")

    emit("[5] 调用计数（真实网络）：")
    emit(
        f"    LLM: planner={planner.calls} "
        + " ".join(f"{agent}={llms[agent].calls}" for agent in AGENTS)
        + f" risk_review={risk_llm.calls}"
    )
    emit("    数据快照取数: " + " ".join(f"{key}={item.calls}" for key, item in providers.items()))
    emit(f"SMOKE STAGE2 DONE elapsed={time.time() - START:.1f}s")
    return 0


def _agent_line(name: str, payload: dict) -> str:
    if payload.get("status") == "unavailable":
        return f"{name}: 降级 unavailable（error_kind={payload.get('error_kind')}）"
    return (
        f"{name}: conclusion={payload.get('conclusion')} confidence={payload.get('confidence')} "
        f"key_points={len(payload.get('key_points') or [])} revised_in_round={payload.get('revised_in_round')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="第七步真实冒烟（两阶段，日志文件友好）")
    parser.add_argument(
        "--stage",
        choices=("1", "2", "all"),
        default="all",
        help="1=只取数（不调 LLM）；2=只跑全图（读快照，不碰 akshare）；all=先 1 后 2",
    )
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT, help=f"快照文件路径（默认 {DEFAULT_SNAPSHOT}）")
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    emit("== 第七步真实冒烟：多分析师并行 + 质询回传 ==")
    emit(f"SMOKE START stage={args.stage} 标的={TICKER} 查询={QUERY} 快照={snapshot_path}")

    snapshot: dict = {}
    if args.stage in ("1", "all"):
        snapshot = stage1(snapshot_path)
    if args.stage in ("2", "all"):
        if not snapshot:
            snapshot = read_snapshot(snapshot_path)
        code = stage2(snapshot)
        if code:
            return code

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    leaked = bool(key) and key in "\n".join(OUTPUT)
    emit(f"[检查] key 明文出现: {leaked}")
    if leaked:
        return 2
    if args.stage == "all":
        emit("SMOKE ALL DONE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAIL {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1)
