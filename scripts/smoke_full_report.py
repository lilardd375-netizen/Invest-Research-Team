"""第八步真实冒烟：全图 + 报告合成 + Markdown 落盘（两阶段，日志文件方式）。

用法（分钟级真实冒烟必须走日志文件 + DONE 标记，避免 tool call 中断丢证据）：
    .\\.venv\\Scripts\\python.exe -u -X utf8 scripts\\smoke_full_report.py --stage all > _smoke8.log 2>&1

阶段一：四个数据源真实取数（单源失败不阻塞），写快照 JSON。
阶段二：真实 key 跑全图（含 report_llm 合成 + 规则式兜底），落盘 reports/{code}_{date}.md。

兼作第七步欠下的「干净证据链」验证：
  - 逐轮 severity 是否真实下降（degraded=False 的轮次才算真实）；
  - 修订硬约束后风控是否还提「只披露未修正」。
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_SNAPSHOT = ROOT / "_smoke7_snapshot.json"
AGENTS = ("financial_analyst", "industry_analyst", "sentiment_analyst", "technical_analyst")
START = time.time()

sys.path.insert(0, str(ROOT))


def _load_smoke():
    """复用 scripts/smoke_multi_agent.py 的阶段一与工具函数（不触发它的 main）。"""
    spec = importlib.util.spec_from_file_location("smoke_multi_agent", HERE / "smoke_multi_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def emit(text: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - START:6.1f}s] {text}", flush=True)


def stage2_report(smoke, snapshot: dict) -> int:
    from src.graph.builder import build_graph
    from src.llm.client import DeepSeekClient
    from src.report.render import write_report

    emit("=" * 78)
    emit("[阶段二] 全图真实链路（真实 key + 阶段一快照） + 报告合成 + Markdown 落盘")
    emit(f"密钥: DEEPSEEK_API_KEY(已配置) len={len(os.environ.get('DEEPSEEK_API_KEY', ''))}")

    real = DeepSeekClient()
    planner = smoke.CountingLLM(real, "planner")
    llms = {agent: smoke.CountingLLM(real, agent) for agent in AGENTS}
    risk_llm = smoke.CountingLLM(real, "risk_review")
    report_llm = smoke.CountingLLM(real, "finalize_report")
    providers = {
        "financial_provider": smoke.SnapshotProvider(snapshot.get("financial"), "financial"),
        "industry_provider": smoke.SnapshotProvider(snapshot.get("industry"), "industry"),
        "sentiment_provider": smoke.SnapshotProvider(snapshot.get("sentiment"), "sentiment"),
        "technical_provider": smoke.SnapshotProvider(snapshot.get("technical"), "technical"),
    }

    app = build_graph(
        planner_llm=planner,
        financial_llm=llms["financial_analyst"],
        industry_llm=llms["industry_analyst"],
        sentiment_llm=llms["sentiment_analyst"],
        technical_llm=llms["technical_analyst"],
        risk_llm=risk_llm,
        report_llm=report_llm,
        **providers,
    )

    track = smoke._progress_tracker()
    emit("进入 planner")
    snapshots = []
    for state in app.stream({"query": smoke.QUERY}, stream_mode="values"):
        snapshots.append(state)
        track(state)
    final = snapshots[-1]

    outline = final.get("outline") or {}
    emit(f"[1] 大纲：status={outline.get('status')} sections={len(outline.get('sections') or [])}")

    emit("[2] 每轮风控复核（severity / 是否降级 / 质询分布）：")
    severities: list[str] = []
    seen_review = None
    round_index = 0
    unresolved_note_rounds: list[int] = []
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
            issue = str(challenge.get("issue") or "")
            if "只披露" in issue or "未修正" in issue:
                unresolved_note_rounds.append(round_index)
        severities.append(str(review.get("severity")))
        emit(
            f"    第 {round_index} 次复核：severity={review.get('severity')} "
            f"degraded={bool(review.get('degraded'))} 质询 {len(challenges)} 条 "
            f"分布={json.dumps(distribution, ensure_ascii=False)} revision_round={snap.get('revision_round', 0)}"
        )
        emit(f"      summary: {review.get('summary')}")
        for challenge in challenges:
            emit(f"      - [{challenge.get('target_agent')}] {challenge.get('issue')}")
        for agent in AGENTS:
            payload = (snap.get("analyses") or {}).get(agent)
            if payload:
                emit(f"      {smoke._agent_line(agent, payload)}")

    emit("[3] 最终四分析师结果：")
    for agent in AGENTS:
        payload = (final.get("analyses") or {}).get(agent)
        emit(f"    {smoke._agent_line(agent, payload) if payload else f'{agent}: 无结果'}")

    emit("[4] 循环与降级标记：")
    emit(f"    revision_round={final.get('revision_round', 0)} round_at_last_review={final.get('round_at_last_review')}")
    emit(f"    severity 序列={severities}")
    emit(f"    risk_degraded={final.get('risk_degraded')} risk_degraded_round={final.get('risk_degraded_round')}")
    emit(f"    review_history 轮数={len(final.get('review_history') or [])}")
    emit(f"    report_llm_degraded={final.get('report_llm_degraded')} report_error={final.get('report_error')!r}")
    emit(f"    planner_error={final.get('planner_error')!r} risk_error={final.get('risk_error')!r}")
    emit(f"    analyst_errors={final.get('analyst_errors')!r}")

    report = final.get("report") or {}
    emit("[5] 合成报告（ReportModel）：")
    emit(f"    title={report.get('title')!r} stock_code={report.get('stock_code')} date={report.get('report_date')}")
    emit(f"    overall_conclusion={report.get('overall_conclusion')} confidence={report.get('overall_confidence')}")
    emit(f"    rule_based={report.get('rule_based')}")
    for item in report.get("per_agent_summaries") or []:
        emit(f"    - {item.get('agent')}: {item.get('conclusion')}/{item.get('confidence')} status={item.get('status')}｜{item.get('summary')}")
    for label, key in (("key_thesis", "key_thesis"), ("key_risks", "key_risks")):
        for item in report.get(key) or []:
            emit(f"    {label}: {item}")
    emit("    revision_history:")
    for entry in report.get("revision_history") or []:
        emit(
            f"      - 第 {entry.get('round')} 轮 severity={entry.get('severity')} degraded={entry.get('degraded')} "
            f"目标={json.dumps(entry.get('challenge_targets') or {}, ensure_ascii=False)} "
            f"修订者={entry.get('revised_agents')}"
        )
    emit("    disclosures:")
    for item in report.get("disclosures") or []:
        emit(f"      - {item}")

    markdown = final.get("final_report") or ""
    path = write_report(final)
    emit(f"[6] Markdown 落盘：{path}")
    emit(f"    字符数={len(markdown)} 行数={len(markdown.splitlines())}")
    emit("    ---- 报告全文 ----")
    for line in markdown.splitlines():
        emit(f"    {line}")
    emit("    ---- 全文结束 ----")

    emit("[7] 调用计数（真实网络）：")
    emit(
        f"    LLM: planner={planner.calls} "
        + " ".join(f"{agent}={llms[agent].calls}" for agent in AGENTS)
        + f" risk_review={risk_llm.calls} finalize_report={report_llm.calls}"
    )
    emit("    快照取数: " + " ".join(f"{key}={item.calls}" for key, item in providers.items()))

    emit("[8] 证据链观察（第七步欠账）：")
    real_drop = len(severities) >= 2 and "high" in severities and severities[-1] in ("medium", "low")
    degraded_last = bool((report.get("revision_history") or [{}])[-1].get("degraded"))
    emit(
        f"    severity 真实下降（非降级产物）={(real_drop and not degraded_last)}"
        f"；最后一轮 degraded={degraded_last}；序列={severities}"
    )
    emit(
        f"    风控仍提「只披露未修正」的轮次={unresolved_note_rounds or '无'}"
        "（硬约束条款生效则为空或仅出现在首轮）"
    )

    payload = json.dumps({"report": report, "markdown": markdown, "severities": severities}, ensure_ascii=False)
    payload += json.dumps(final, ensure_ascii=False, default=str)
    # 只比对「运行时」从环境变量派生的凭据串，源码里不写任何真实 key 的片段
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    tokens = [key] + ([key[:6]] if len(key) > 8 else []) + ["sk-"]
    leaked = sorted({token for token in tokens if token and token in payload})
    if leaked:
        emit(f"[9] 密钥检查：失败！输出中出现 {len(leaked)} 个疑似 key 片段")
        return 2
    emit("[9] 密钥检查：通过（输出与报告中无 key 明文）")
    emit(f"SMOKE STAGE2 DONE elapsed={time.time() - START:.1f}s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["1", "2", "all"], default="all")
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    args = parser.parse_args()
    snapshot_path = Path(args.snapshot)

    smoke = _load_smoke()
    snapshot: dict = {}
    if args.stage in ("1", "all"):
        snapshot = smoke.stage1(snapshot_path)
    if args.stage in ("2", "all"):
        if not snapshot:
            snapshot = smoke.read_snapshot(snapshot_path)
        return stage2_report(smoke, snapshot)
    emit(f"SMOKE STAGE1 DONE elapsed={time.time() - START:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
