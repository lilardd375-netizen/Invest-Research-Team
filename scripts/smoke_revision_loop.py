"""真实链路冒烟（第六步）：DeepSeek + akshare，验证「质询回传 -> 修订 -> 复审」。

用法：
    .venv\\Scripts\\python.exe scripts\\smoke_revision_loop.py

要求：
- 需要 DEEPSEEK_API_KEY（.env 或环境变量），**不打印任何 key 明文**；
- 会消耗真实 token，并每轮访问 akshare 取真实财务数据（不做数据缓存）；
- 打印每轮风控 severity / 每轮质询数 / 分析师修订后的关键变化 / 走了几轮 / final 路径。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.builder import MAX_REVISION_ROUNDS, build_graph  # noqa: E402
from src.llm.client import DeepSeekClient  # noqa: E402
from src.tools.financial_data import AkshareProvider  # noqa: E402

QUERY = "宁德时代 300750 的投资价值"
TICKER = "300750"

WATCHED = (
    "outline",
    "analyses",
    "risk_review",
    "pending_challenges",
    "revision_round",
    "final_report",
)

LOG: list[str] = []


def emit(line: str = "") -> None:
    LOG.append(line)
    print(line, flush=True)


class CountingLLM:
    """透明代理：统计真实 LLM 调用次数。"""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def chat(self, system: str, user: str) -> str:
        self.calls += 1
        return self.inner.chat(system, user)

    def plan_outline(self, query: str) -> str:
        self.calls += 1
        return self.inner.plan_outline(query)


class CountingProvider:
    """透明代理：统计真实取数次数与期次（验证每轮重新取数、不做缓存）。"""

    def __init__(self, inner):
        self.inner = inner
        self.rounds: list[str] = []
        self.last = None

    def get_financials(self, ticker):
        data = self.inner.get_financials(ticker)
        self.last = data
        self.rounds.append(str(data.get("report_period")))
        return data


def main() -> int:
    llm = CountingLLM(DeepSeekClient())
    provider = CountingProvider(AkshareProvider())

    app = build_graph(
        planner_llm=llm,
        financial_llm=llm,
        risk_llm=llm,
        data_provider=provider,
    )

    snapshots = list(
        app.stream(
            {"query": QUERY, "revision_round": 0, "round_at_last_review": -1},
            config={"recursion_limit": 40},
            stream_mode="values",
        )
    )
    result = snapshots[-1]

    # ---- 从状态快照还原执行顺序 ----
    risk_events, analyst_events, revise_events, finalize_events = [], [], [], []
    prev = snapshots[0]
    for snap in snapshots[1:]:
        changed = [k for k in WATCHED if snap.get(k) != prev.get(k)]
        if "risk_review" in changed:
            review = snap.get("risk_review") or {}
            risk_events.append(
                {
                    "round": snap.get("revision_round", 0),
                    "severity": review.get("severity"),
                    "challenges": review.get("challenges") or [],
                    "summary": review.get("summary"),
                }
            )
        elif "analyses" in changed:
            analysis = (snap.get("analyses") or {}).get("financial_analyst") or {}
            analyst_events.append(
                {
                    "round": snap.get("revision_round", 0),
                    "revised_in_round": analysis.get("revised_in_round"),
                    "conclusion": analysis.get("conclusion"),
                    "key_points": len(analysis.get("key_points") or []),
                    "concerns": len(analysis.get("concerns") or []),
                    "status": analysis.get("status", "ok"),
                }
            )
        elif "pending_challenges" in changed or "revision_round" in changed:
            pending = snap.get("pending_challenges") or {}
            revise_events.append(
                {
                    "round": snap.get("revision_round", 0),
                    "distribution": {k: len(v) for k, v in pending.items()},
                }
            )
        elif "final_report" in changed:
            finalize_events.append(snap.get("final_report"))
        prev = snap

    data = provider.last or {}

    emit("=== 数据源（真实 akshare）===")
    emit(f"  报告期 = {data.get('report_period')}   as_of = {data.get('as_of')}")
    emit(f"  缺失指标 = {data.get('missing_metrics')}")
    emit(f"  图内取数次数 = {len(provider.rounds)}，期次 = {provider.rounds}")

    emit("=== 每轮分析师产出 ===")
    for ev in analyst_events:
        emit(
            f"  revision_round={ev['round']}  revised_in_round={ev['revised_in_round']}"
            f"  conclusion={ev['conclusion']}  要点 {ev['key_points']} 条"
            f"  concerns {ev['concerns']} 条  status={ev['status']}"
        )

    emit("=== 每轮风控 ===")
    for ev in risk_events:
        emit(
            f"  第 {ev['round']} 轮：severity={ev['severity']}，质询 {len(ev['challenges'])} 条"
            f"  summary={ev['summary']}"
        )

    emit("=== 质询分发（revise）===")
    for ev in revise_events:
        emit(f"  -> revision_round={ev['round']}  pending={ev['distribution']}")

    emit("=== 最终路径 ===")
    rounds = result.get("revision_round", 0)
    emit(f"  返工轮次 = {rounds} / 上限 {MAX_REVISION_ROUNDS}")
    emit(f"  final_report = {result.get('final_report')!r}")
    emit(f"  LLM 总调用次数 = {llm.calls}")
    errors = {k: v for k, v in result.items() if k.endswith("_error")}
    emit(f"  降级信息 = {errors or '无'}")
    emit(f"  收尾事件 = {finalize_events}")

    # ---- 验收重点：最后一轮风控是否因质询被回应而降温 ----
    last = risk_events[-1] if risk_events else {}
    first = risk_events[0] if risk_events else {}
    emit("=== 验收判定 ===")
    emit(f"  首轮 severity = {first.get('severity')}（质询 {len(first.get('challenges') or [])} 条）")
    emit(f"  末轮 severity = {last.get('severity')}（质询 {len(last.get('challenges') or [])} 条）")
    if last.get("severity") in ("medium", "low") and rounds >= 1:
        emit("  [OK] 质询回传后 severity 下降，返工真实生效")
    elif rounds >= MAX_REVISION_ROUNDS and last.get("severity") == "high":
        emit("  [!] 第二轮复审仍为 high，撞上限强制成稿。末轮质询原文：")
        for i, ch in enumerate(last.get("challenges") or [], 1):
            emit(f"   [{i}] target={ch.get('target_agent')}")
            emit(f"       issue    : {ch.get('issue')}")
            emit(f"       required : {ch.get('required_action')}")
    else:
        emit("  [i] 未触发返工（首轮即非 high），本步核心路径未被真实数据触发")

    blob = "\n".join(LOG) + json.dumps(result, ensure_ascii=False, default=str)
    key = os.environ.get("DEEPSEEK_API_KEY") or ""
    if key and key in blob:
        emit("[FAIL] 输出中出现 key 明文")
        return 1
    emit("[OK] 输出中不含 key 明文")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
