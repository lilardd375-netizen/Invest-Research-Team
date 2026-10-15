"""报告渲染与规则式组装：纯函数，state 进、Markdown 字符串出，不调用 LLM。

三条硬约束（第八步）：
1. render_report(state) 只读 state，可离线测试、可重放，不依赖网络与 LLM；
2. 「返工记录」与「披露与局限」必须独立成章——返工过程与诚实披露是这个项目的
   灵魂，不允许被合进正文冲淡；
3. 报告合成 LLM 失败时用 build_rule_based_report(state) 兜底（规则式拼接四家
   结论 + 质询列表），合成失败绝不允许丢掉全流程结果。
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

from src.tools.utils import normalize_ticker

#: 报告里固定的呈现顺序（逐家呈现，分歧必须保留）。
AGENT_ORDER = (
    "financial_analyst",
    "industry_analyst",
    "sentiment_analyst",
    "technical_analyst",
)
DEFAULT_REPORT_DIR = "reports"
MAX_THESIS = 5
MAX_RISKS = 8
MAX_DISCLOSURES = 12
MAX_ASSUMPTIONS = 4
ONE_LINE_CHARS = 160
AGENT_LABELS = {
    "financial_analyst": "财务",
    "industry_analyst": "行业",
    "sentiment_analyst": "舆情",
    "technical_analyst": "技术面",
}
_SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z._-]")


def _one_line(text, limit: int = ONE_LINE_CHARS) -> str:
    """把任意文本压成一行（换行变空格，超长截断加省略号）。"""
    flat = " ".join(str(text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _vote(values) -> str | None:
    """多数票（必须严格过半）才给结论，否则返回 None（不硬凑结论）。"""
    counts: dict[str, int] = {}
    for value in values:
        if value:
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    winners = [key for key, count in counts.items() if count == best]
    if len(winners) == 1 and best * 2 > sum(counts.values()):
        return winners[0]
    return None


def _challenge_line(challenge: dict, prefix: str = "") -> str:
    issue = _one_line(challenge.get("issue"))
    action = _one_line(challenge.get("required_action"))
    target = challenge.get("target_agent") or "unassigned"
    body = f"{issue}（要求：{action}）" if action else issue
    return f"{prefix}[{target}] {body}"


def build_rule_based_report(state: dict) -> dict:
    """规则式组装报告（无 LLM）：直接拼接四家结论 + 质询列表。

    用在两条降级路径：finalize 未注入 report_llm，或合成 LLM 调用/校验失败。
    这是「宁可朴素也不能丢结果」的兜底——报告可以不加修饰，但不能因为合成
    Agent 挂了就丢掉四个分析师和两轮返工的全部产出。
    """
    analyses = _as_dict(state.get("analyses"))
    history = [entry for entry in (state.get("review_history") or []) if isinstance(entry, dict)]
    query = str(state.get("query", "") or "")
    stock_code = normalize_ticker(query) or "unknown"
    report_date = date.today().isoformat()

    summaries: list[dict] = []
    conclusions: list[str] = []
    confidences: list[str] = []
    degradations: list[str] = []
    assumptions: list[str] = []
    thesis: list[str] = []
    risks: list[str] = []

    for agent in AGENT_ORDER:
        if agent not in analyses:
            continue
        value = _as_dict(analyses.get(agent))
        status = str(value.get("status") or "ok")
        conclusion = str(value.get("conclusion") or "")
        confidence = str(value.get("confidence") or "")
        points = value.get("key_points") or []
        if status == "unavailable":
            error = _one_line(value.get("error"))
            summary = f"该维度数据/LLM 不可用，未产出结论：{error}"
            degradations.append(f"{agent} 不可用：{error}")
        else:
            conclusions.append(conclusion)
            confidences.append(confidence)
            first = _as_dict(points[0]) if points else {}
            summary = _one_line(first.get("finding")) or "（未给出要点）"
        missing = value.get("missing_fields") or value.get("missing_metrics") or []
        if missing:
            degradations.append(f"{agent} 缺失字段：{'、'.join(str(item) for item in missing)}")
        for point in points[:2]:
            point = _as_dict(point)
            finding = _one_line(point.get("finding"))
            if finding and len(thesis) < MAX_THESIS:
                source = _one_line(point.get("data_source"), 60)
                thesis.append(f"{finding}（来源：{source}）" if source else finding)
        for concern in (value.get("concerns") or [])[:2]:
            if len(risks) < MAX_RISKS:
                risks.append(f"{agent} 自述关注：{_one_line(concern)}")
        for item in (value.get("assumptions") or [])[:2]:
            if len(assumptions) < MAX_ASSUMPTIONS:
                assumptions.append(f"{agent} 的假设：{_one_line(item)}")
        summaries.append(
            {
                "agent": agent,
                "status": status,
                "conclusion": conclusion,
                "confidence": confidence,
                "summary": summary,
            }
        )

    revision_history: list[dict] = []
    high_challenges: list[str] = []
    medium_challenges: list[str] = []
    for entry in history:
        severity = str(entry.get("severity") or "")
        degraded = bool(entry.get("degraded"))
        note = _one_line(entry.get("summary"))
        if degraded:
            note = (
                f"该轮风控复核降级，severity 是兜底值而非真实评审结论。{note}"
            ).strip()
        for challenge in entry.get("challenges") or []:
            challenge = _as_dict(challenge)
            if not challenge:
                continue
            line = _challenge_line(challenge)
            if severity == "high":
                if len(high_challenges) < MAX_RISKS:
                    high_challenges.append(line)
            elif severity == "medium":
                if len(medium_challenges) < MAX_DISCLOSURES:
                    medium_challenges.append(line)
        revision_history.append(
            {
                "round": int(entry.get("round", 0) or 0),
                "severity": severity,
                "degraded": degraded,
                "challenge_targets": dict(entry.get("challenge_targets") or {}),
                "revised_agents": list(entry.get("revised_agents") or []),
                "note": note,
            }
        )

    risks.extend(high_challenges)

    disclosures: list[str] = []
    disclosures.extend(medium_challenges)
    for entry in revision_history:
        if entry["degraded"]:
            disclosures.append(
                f"第 {entry['round']} 轮风控复核因故障降级：该轮 severity={entry['severity']}"
                "是兜底值，不代表真实评审结论。"
            )
    if revision_history and revision_history[-1]["severity"] == "high":
        disclosures.append(
            "返工轮次上限用尽时最后一轮风控仍判 high：上列 high 质询未全部解决，"
            "本报告按流程成稿，结论仅供参考。"
        )
    disclosures.extend(degradations)
    disclosures.append(
        "覆盖边界：本项目分析师只覆盖财务 / 行业 / 舆情 / 技术面四个维度，"
        "估值、宏观、政策维度暂无数据源（已知任务边界）。"
    )
    disclosures.extend(assumptions)
    vote = _vote(conclusions)
    if vote is None and len(set(conclusions)) > 1:
        counts = "、".join(
            f"{value} {conclusions.count(value)} 家" for value in dict.fromkeys(conclusions) if value
        )
        disclosures.append(
            f"四家结论未形成严格多数（{counts}），综合结论按保守口径取「中性」，"
            "真实分歧见「各分析师结论」章节。"
        )
    if state.get("report_llm_degraded"):
        disclosures.append("本报告由规则式组装生成（report_llm_degraded=True），未经合成 LLM 交叉核对。")
    dropped = max(0, len(disclosures) - MAX_DISCLOSURES)
    if dropped:
        disclosures = disclosures[:MAX_DISCLOSURES] + [f"…另有 {dropped} 条披露未列出"]

    degraded_now = state.get("report_llm_degraded") or state.get("risk_degraded")
    if any(entry["degraded"] for entry in revision_history):
        degraded_now = True
    if any("不可用" in item for item in degradations):
        degraded_now = True

    return {
        "title": f"{stock_code} 投资研究报告",
        "stock_code": stock_code,
        "report_date": report_date,
        "overall_conclusion": vote or "中性",
        "overall_confidence": "低" if degraded_now else (_vote(confidences) or "中"),
        "per_agent_summaries": summaries,
        "key_thesis": thesis,
        "key_risks": risks,
        "revision_history": revision_history,
        "disclosures": disclosures,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rule_based": True,
    }


def render_report(state: dict) -> str:
    """把 state 渲染成 Markdown 字符串（纯函数，无副作用）。

    state["report"] 缺失时用规则式组装兜底，因此离线测试可以直接给一个
    只含 analyses / review_history 的 state。
    """
    report = _as_dict(state.get("report")) or build_rule_based_report(state)
    stock_code = str(report.get("stock_code") or normalize_ticker(state.get("query", "")) or "unknown")
    report_date = str(report.get("report_date") or date.today().isoformat())
    title = str(report.get("title") or f"{stock_code} 投资研究报告")

    lines: list[str] = [f"# {title}", ""]
    lines.append(
        f"> 标的：{stock_code} ｜ 报告日期：{report_date} ｜ "
        f"生成时间：{report.get('generated_at') or '—'}"
    )
    if state.get("report_llm_degraded"):
        error = _one_line(state.get("report_error"))
        lines.append(">")
        lines.append(
            "> 注意：本报告为**降级产物**（报告合成未成功，由规则式组装生成）"
            + (f"：{error}" if error else "。")
        )
    lines.append("")

    lines.append("## 综合结论")
    lines.append("")
    lines.append(
        f"**{report.get('overall_conclusion') or '中性'}**"
        f"（信心：{report.get('overall_confidence') or '中'}）"
    )
    lines.append("")
    lines.append("以下逐家列出各分析师的原始结论，分歧一律保留。")
    lines.append("")

    lines.append("## 各分析师结论")
    lines.append("")
    summaries = [item for item in (report.get("per_agent_summaries") or []) if isinstance(item, dict)]
    if summaries:
        for item in summaries:
            agent = str(item.get("agent") or "unknown")
            label = AGENT_LABELS.get(agent, agent)
            lines.append(f"### {label}（{agent}）")
            lines.append("")
            status = str(item.get("status") or "ok")
            conclusion = item.get("conclusion") or "未给出"
            confidence = item.get("confidence") or "—"
            if status != "ok":
                conclusion = f"{conclusion}（status={status}）"
            lines.append(f"- 结论：{conclusion}")
            lines.append(f"- 信心：{confidence}")
            lines.append(f"- 要点：{_one_line(item.get('summary')) or '（无）'}")
            lines.append("")
    else:
        lines.append("（本轮没有任何分析师产出结果）")
        lines.append("")

    lines.append("## 核心逻辑")
    lines.append("")
    thesis = report.get("key_thesis") or []
    lines.extend(f"- {_one_line(item)}" for item in thesis if _one_line(item))
    if not thesis:
        lines.append("- （无）")
    lines.append("")

    lines.append("## 主要风险")
    lines.append("")
    risks = report.get("key_risks") or []
    lines.extend(f"- {_one_line(item)}" for item in risks if _one_line(item))
    if not risks:
        lines.append("- （无）")
    lines.append("")

    lines.append("## 返工记录")
    lines.append("")
    lines.append("> 每轮风控复核的 severity、质询分布与谁真的改了，逐轮留档。")
    lines.append("")
    history = [item for item in (report.get("revision_history") or []) if isinstance(item, dict)]
    if history:
        for entry in history:
            targets = entry.get("challenge_targets") or {}
            target_text = "、".join(f"{key} {value} 条" for key, value in targets.items()) or "无"
            revised = "、".join(entry.get("revised_agents") or []) or "无"
            flag = "（降级产物，非真实评审）" if entry.get("degraded") else ""
            lines.append(
                f"- 第 {entry.get('round', 0)} 轮：severity={entry.get('severity') or '—'}{flag}"
                f" ｜ 质询：{target_text} ｜ 修订者：{revised}"
            )
            note = _one_line(entry.get("note"))
            if note:
                lines.append(f"  - 风控总评：{note}")
    else:
        lines.append("- （没有风控复核记录）")
    lines.append("")

    lines.append("## 披露与局限")
    lines.append("")
    disclosures = report.get("disclosures") or []
    lines.extend(f"- {_one_line(item)}" for item in disclosures if _one_line(item))
    if not disclosures:
        lines.append("- （无）")
    lines.append("")

    return "\n".join(lines)


def _safe_name(value: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", str(value or "").strip())
    return cleaned.strip("._") or "unknown"


def report_path(state: dict, output_dir: str | Path = DEFAULT_REPORT_DIR) -> Path:
    """算出报告落盘路径：{output_dir}/{stock}_{date}.md。"""
    report = _as_dict(state.get("report")) or build_rule_based_report(state)
    stock = _safe_name(report.get("stock_code") or "unknown")
    day = _safe_name(report.get("report_date") or date.today().isoformat())
    return Path(output_dir) / f"{stock}_{day}.md"


def write_report(state: dict, output_dir: str | Path = DEFAULT_REPORT_DIR) -> Path:
    """渲染并落盘，返回写入路径（目录不存在会自动创建）。"""
    path = report_path(state, output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(state), encoding="utf-8")
    return path
