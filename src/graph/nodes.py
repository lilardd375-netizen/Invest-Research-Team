"""占位节点 + 各 Agent 节点工厂。"""

import json
import re
from datetime import date

from pydantic import ValidationError

from src.llm.errors import LLMFatalError, LLMTransientError
from src.llm.schemas import (
    FinancialAnalysisModel,
    OutlineModel,
    RiskReviewModel,
)

# 主编 Agent 的 system prompt（模块级常量，中文）。
PLANNER_SYSTEM_PROMPT = """你是一名资深投资研究主编，负责为个股研究制定大纲。

要求：
1. 围绕用户给出的标的，拆解出 4-6 个研究章节。
2. 每个章节给出 2-4 个具体、可查证的研究问题。
3. 研究问题要落到可验证的事实上（营收结构、产能、客户集中度、
   竞争格局、政策风险、估值区间等），不要空泛。

只输出 JSON，不要其他文字。JSON 结构必须是：
{
  "sections": [
    {"title": "章节标题", "research_questions": ["问题1", "问题2"]}
  ]
}"""

# 财务分析 Agent 的 system prompt（模块级常量，中文）。
FINANCIAL_ANALYST_SYSTEM_PROMPT = """你是一名严谨的财务分析 Agent，服务于投资研究团队。

你会收到某标的的结构化财务指标（含报告期 as_of）。要求：

1. 只依据给定指标下结论，禁止编造缺失的数据；
   数据缺失就写进 assumptions 或 concerns，不要猜数字。
2. key_points 每条必须标注 data_source，写清引用的是哪个指标
   （例如 "roe (2025-03)"），且 finding 里带上具体数值。
3. conclusion 只能是 增持 / 中性 / 减持 三者之一。
4. confidence 只能是 高 / 中 / 低 三者之一；数据越不完整就越低。
5. concerns 写风险点，assumptions 写结论所依赖的假设。

只输出 JSON，不要其他文字。JSON 结构必须是：
{
  "agent": "financial_analyst",
  "conclusion": "增持|中性|减持",
  "confidence": "高|中|低",
  "key_points": [{"finding": "...", "data_source": "..."}],
  "concerns": ["..."],
  "assumptions": ["..."]
}"""

# 重试次数（1 次重试 = 最多 2 次调用）。
MAX_ATTEMPTS = 2
PLANNER_MAX_ATTEMPTS = MAX_ATTEMPTS

# 风控 Agent 的 system prompt（模块级常量，中文）。
RISK_REVIEW_SYSTEM_PROMPT = """你是投资研究团队里的反方辩手（风控 Agent），天性多疑。

你的目标不是否定一切，而是让这份报告经得起考验——能站住的地方要认，
站不住的地方必须指出来。审查清单：

1. 数据质疑：结论引用的数据来源是否可靠、是否过期、是否被选择性使用？
2. 逻辑漏洞：从数据到结论的推理是否成立？有没有跳跃、循环论证或过度外推？
3. 反例寻找：有没有与结论相反的证据、被忽略的同类公司或历史反例？
4. 假设压力测试：结论依赖的假设若不成立，后果是什么？假设是否偏激进？
5. 确认偏误：是否只找支持性的证据，而忽略了不利信息？

打分标准（决定是否返工）：
- high   = 存在数据错误或口径错误，或已覆盖维度里的重大逻辑漏洞，结论不可采信，应返工；
- medium = 假设偏激进但可接受，报告中披露即可；
- low    = 常规提示，并入风险章节即可。

任务边界（已知，不是缺陷）：当前已实现的分析师只有 financial_analyst。
大纲中其他板块暂无数据源是已知的任务边界：可以在 summary 或 challenges 里
提示覆盖缺口，但不应仅因覆盖不足就把 severity 判为 high。
severity=high 只用于：存在数据错误、口径错误、或已覆盖维度的逻辑漏洞。

不确定时标注需要补充数据（写进 challenges 的 required_action），
而不是武断下结论。特别地：如果某个分析 Agent 的数据缺失、返回
unavailable 或报错，必须把这件事本身列为 challenge，或在 summary 里明确指出——
"这份分析没有数据支撑"本身就是最重要的风险。

只输出 JSON，不要其他文字。JSON 结构必须是：
{
  "severity": "high|medium|low",
  "challenges": [
    {"target_agent": "financial_analyst", "issue": "...", "required_action": "..."}
  ],
  "summary": "一句话总评"
}"""

# 送进风控 prompt 的章节标题上限（防止大纲过长把 prompt 撑爆）。
MAX_OUTLINE_SECTIONS = 12

# 送进复审 prompt 的上一轮质询条数上限。
MAX_CHALLENGES_IN_DIGEST = 8

# 已实现的分析师（revise 只把质询分发给它们，其余归 unassigned）。
# 第七步实现行业/舆情/技术面 Agent 后在这里扩充。
KNOWN_AGENTS = ("financial_analyst",)

_TICKER_RE = re.compile(r"\d{6}")


def extract_ticker(text: str) -> str:
    """从 query 里取 6 位数字作为 ticker，找不到就用整个 query。"""
    match = _TICKER_RE.search(text or "")
    return match.group(0) if match else (text or "")


def _prepare_payload(raw: str, model_cls):
    """字符串 -> JSON 解析 -> Pydantic 校验，失败一律归为可重试错误。"""
    try:
        data = json.loads(raw)
    except Exception as exc:
        raise LLMTransientError(f"JSON 解析失败: {exc}") from exc
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise LLMTransientError(f"结构校验失败: {exc}") from exc


def _retry_or_fail(fetch, model_cls, degrade, max_attempts: int = MAX_ATTEMPTS):
    """调用 fetch -> 解析/校验，按异常类别决定重试或直接降级。

    degrade(error_kind, summary) -> 该 Agent 自己的降级 dict。
    返回 (payload, None) 或 (None, 降级 dict)。
    可重试错误用满 max_attempts 次；LLMFatalError 立刻降级不重试，
    降级信息里标明错误类别，避免 401 这类硬故障被当成「模型输出不稳」。
    """
    last_error = None
    attempts = 0
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        try:
            raw = fetch()
            return _prepare_payload(raw, model_cls), None
        except LLMTransientError as exc:
            last_error = exc
            continue  # 唯一会重试的类别
        except LLMFatalError as exc:
            last_error = exc
            break

    is_fatal = isinstance(last_error, LLMFatalError)
    summary = str(last_error)
    if not is_fatal:
        summary = f"重试 {attempts} 次后仍失败: {summary}"
    return None, degrade("fatal" if is_fatal else "transient", summary)


def make_planner_node(llm):
    """planner 节点工厂：注入 llm，返回 node_fn(state) -> dict。

    llm 需提供 plan_outline(query) -> str（也可以是 chat(system, user)）。
    """

    def planner_node(state: dict) -> dict:
        payload, failure = _retry_or_fail(
            lambda: llm.plan_outline(state.get("query", "")),
            OutlineModel,
            # planner 的降级契约是 status="stub"（与未注入 LLM 时一致）
            lambda kind, summary: {
                "outline": {"status": "stub", "error": summary, "error_kind": kind}
            },
        )
        if failure is not None:
            failure["planner_error"] = failure["outline"]["error"]
            return failure
        return {"outline": payload.model_dump()}

    return planner_node


def make_financial_analyst_node(llm, data_provider):
    """财务分析节点工厂：注入 llm + data_provider。

    流程：提取 ticker -> 取财务指标 -> 指标序列化进 user prompt ->
    llm.chat -> Pydantic 校验 -> {"analyses": {"financial_analyst": ...}}。
    Provider 抛异常时不调用 LLM，直接降级，不编造数据。

    第六步新增：若 state["pending_challenges"] 里有发给本 Agent 的质询
    （revise 分发而来），把质询追加进 user prompt 要求逐条正面回应，并在
    产出里标记 revised_in_round；成功后把本 Agent 那一格的待办清空
    （其他 Agent 的待办保留，避免被覆盖丢失）。
    """

    def financial_analyst_node(state: dict) -> dict:
        query = state.get("query", "")
        ticker = extract_ticker(query)
        pending = dict(state.get("pending_challenges") or {})
        challenges = pending.get("financial_analyst") or []
        try:
            financials = data_provider.get_financials(ticker)
        except Exception as exc:
            summary = f"数据源不可用: {type(exc).__name__}: {exc}"
            return {
                "analyses": {
                    "financial_analyst": {
                        "status": "unavailable",
                        "error": summary,
                        "error_kind": "data_source",
                    }
                },
                "analyst_error": summary,
            }

        user_prompt = (
            f"研究标的：{query}\n"
            f"提取到的股票代码：{ticker}\n"
            f"财务指标（JSON）："
            f"{json.dumps(financials, ensure_ascii=False, default=str)}\n\n"
            "请基于以上指标输出分析 JSON。"
        )
        if challenges:
            user_prompt = f"{user_prompt}\n\n{_challenge_block(challenges)}"
        payload, failure = _retry_or_fail(
            lambda: llm.chat(FINANCIAL_ANALYST_SYSTEM_PROMPT, user_prompt),
            FinancialAnalysisModel,
            # 财务 Agent 的降级契约是 status="unavailable"（与数据源不可用一致）
            lambda kind, summary: {
                "analyses": {
                    "financial_analyst": {
                        "status": "unavailable",
                        "error": summary,
                        "error_kind": kind,
                    }
                }
            },
        )
        if failure is not None:
            # 降级时 failure 已是 {"analyses": {"financial_analyst": {...}}}
            # 质询没被正面回应，就不清空待办（不当成已消费）。
            failure["analyst_error"] = failure["analyses"]["financial_analyst"]["error"]
            return failure

        analysis = payload.model_dump()
        analysis["revised_in_round"] = state.get("revision_round", 0)
        result = {"analyses": {"financial_analyst": analysis}}
        if pending:
            result["pending_challenges"] = _consume_pending(pending, "financial_analyst")
        return result

    return financial_analyst_node


def _challenge_block(challenges) -> str:
    """把待回应的风控质询渲染成 prompt 区块。"""
    lines = []
    for idx, ch in enumerate(challenges, start=1):
        if isinstance(ch, dict):
            issue = (ch.get("issue") or "").strip()
            action = (ch.get("required_action") or "").strip()
        else:
            issue, action = str(ch).strip(), ""
        lines.append(f"{idx}. 问题：{issue}\n   要求：{action}")
    return (
        "== 上一轮风控质询（必须逐条正面回应）==\n"
        + "\n".join(lines)
        + "\n本轮分析必须正面回应上面每一条质询：修正表述、重算口径，"
        "或明确说明为何不采纳，禁止无视。"
    )


def _consume_pending(pending: dict, agent: str) -> dict:
    """消费掉某个 Agent 的待办质询，同时保留其他 Agent 的待办。"""
    updated = dict(pending)
    updated[agent] = []
    return updated


def _missing_llm_stub_node(state: dict) -> dict:
    """未注入 LLM 时的 planner 占位（保持向后兼容）。"""
    return {"outline": {"status": "stub"}}


def _missing_analyst_stub_node(state: dict) -> dict:
    """未注入 LLM 时的财务分析占位：返回空 analyses。"""
    return {"analyses": {}}


# 模块级默认节点：build_graph 未注入依赖时使用
planner_node = _missing_llm_stub_node


def risk_review_node(state: dict) -> dict:
    """风控复核节点（占位）。

    节点本身不做判断：把入参里的 severity 原样回显，缺省为 "low"
    （真实系统里 severity 由风控 Agent 重新评估产生）。
    同时计算保险丝：本轮 revision_round 与上次复核时相同 -> 说明
    上一轮返工没有推进轮次，判定为 stuck。
    """
    severity = (state.get("risk_review") or {}).get("severity", "low")
    return _with_fuse(state, {"severity": severity})


def _outline_digest(outline) -> str:
    """把 outline 压成概要：章节标题 + 每章问题数，避免整份大纲塞满 prompt。"""
    if not isinstance(outline, dict):
        return str(outline)
    if outline.get("status") == "stub" or not outline.get("sections"):
        return f"（大纲不可用/占位：{outline.get('status', 'unknown')}）"

    lines = []
    for section in outline.get("sections", [])[:MAX_OUTLINE_SECTIONS]:
        title = section.get("title", "（无标题）")
        count = len(section.get("research_questions") or [])
        lines.append(f"- {title}（{count} 个研究问题）")
    total = len(outline.get("sections", []))
    if total > MAX_OUTLINE_SECTIONS:
        lines.append(f"- …另有 {total - MAX_OUTLINE_SECTIONS} 个章节未列出")
    return "\n".join(lines)


def _previous_challenge_digest(review) -> str:
    """把上一轮风控质询压成摘要，让复审能判断「质询是否被回应」。"""
    challenges = (review or {}).get("challenges") or []
    lines = []
    for ch in challenges[:MAX_CHALLENGES_IN_DIGEST]:
        if not isinstance(ch, dict):
            lines.append(f"- [unassigned] {ch}")
            continue
        target = ch.get("target_agent") or "unassigned"
        issue = (ch.get("issue") or "").strip()
        action = (ch.get("required_action") or "").strip()
        lines.append(f"- [{target}] {issue}（要求：{action}）")
    if not lines:
        return "（上一轮没有质询）"
    if len(challenges) > MAX_CHALLENGES_IN_DIGEST:
        lines.append(f"- …另有 {len(challenges) - MAX_CHALLENGES_IN_DIGEST} 条未列出")
    return "\n".join(lines)


def _revised_marks(analyses) -> str:
    """列出各分析师本轮的 revised_in_round 标记（判断质询是否被回应）。"""
    marks = []
    for name, value in (analyses or {}).items():
        if isinstance(value, dict) and "revised_in_round" in value:
            marks.append(f"- {name}: revised_in_round={value['revised_in_round']}")
    return "\n".join(marks) if marks else "（本轮没有分析师标记返工轮次）"


def _build_risk_user_prompt(state: dict) -> str:
    """组装风控 user prompt：时间锚点 / query / analyses / outline / 复审上下文。"""
    previous_review = state.get("risk_review") or {}
    previous = previous_review.get("severity")
    revision_round = state.get("revision_round", 0)
    analyses = state.get("analyses") or {}

    if previous:
        prev_count = len(previous_review.get("challenges") or [])
        round_hint = (
            f"这是第 {revision_round} 轮返工后的复审"
            f"（上一轮风控结论：{previous}，上一轮质询 {prev_count} 条）。"
            "请重点确认上一轮的问题是否已解决，以及返工是否引入了新的问题。"
        )
    else:
        round_hint = f"这是首轮评审（revision_round={revision_round}）。"

    return (
        f"当前系统日期：{date.today().isoformat()}。"
        "数据来自 akshare 实时接口，报告期晚于你的知识截止是正常情况，"
        "属于真实数据，不要因数据较新而质疑其真实性；"
        "如有疑问请质疑数据口径，而非数据存在性。\n\n"
        f"研究标的：{state.get('query', '')}\n\n"
        "== 各分析 Agent 的输出（JSON，含 unavailable/error 信息）==\n"
        f"{json.dumps(analyses, ensure_ascii=False, default=str)}\n\n"
        "== 研究大纲概要 ==\n"
        f"{_outline_digest(state.get('outline'))}\n\n"
        "== 评审上下文 ==\n"
        f"{round_hint}\n"
        f"上一轮质询摘要：\n{_previous_challenge_digest(previous_review)}\n"
        f"本轮分析师返工标记：\n{_revised_marks(analyses)}\n\n"
        "请按审查清单质询以上分析，输出评审 JSON。"
    )


def make_risk_review_node(llm):
    """风控节点工厂：注入 llm，返回 node_fn(state) -> dict。

    职责：
    1. 让 LLM 以「反方辩手」身份审查 analyses/outline，产出 severity +
       challenges + summary；
    2. 保险丝：LLM 挂掉时降级为 severity="low" 并写 risk_error，
       **保证流程照常走到 finalize**（风控挂了不该卡死报告）；
    3. 保留占位节点的既有职责——计算 stuck、回写 round_at_last_review。
    """

    def risk_review_node_with_llm(state: dict) -> dict:
        prompt = _build_risk_user_prompt(state)
        payload, failure = _retry_or_fail(
            lambda: llm.chat(RISK_REVIEW_SYSTEM_PROMPT, prompt),
            RiskReviewModel,
            # 风控降级契约：severity 必须落在合法枚举里（"low"），
            # 否则路由拿到非法值会让行为变得不可预测。
            lambda kind, summary: {
                "risk_review": {
                    "severity": "low",
                    "challenges": [],
                    "summary": "风控 Agent 不可用，本轮未产生质询（降级为 low）。",
                    "error": summary,
                    "error_kind": kind,
                }
            },
        )
        review = payload.model_dump() if failure is None else failure["risk_review"]
        result = _with_fuse(state, review)
        if failure is not None:
            # risk_review 里已带 error_kind/error，顶层再给一份摘要供下游消费
            result["risk_error"] = (
                f"{review['error_kind']}: {review['error']}"
            )
        return result

    return risk_review_node_with_llm


def _with_fuse(state: dict, review: dict) -> dict:
    """占位节点的原有职责（保险丝），LLM 路径与占位路径共用。

    本轮 revision_round 与上次复核时相同 -> 上一轮返工没推进轮次，判定 stuck。
    """
    revision_round = state.get("revision_round", 0)
    stuck = revision_round == state.get("round_at_last_review", -1)
    merged = dict(review)
    merged["stuck"] = stuck
    return {"risk_review": merged, "round_at_last_review": revision_round}


def revise_node(state: dict) -> dict:
    """返工节点：把风控质询按 target_agent 分发到各分析师的待办箱。

    - 只分发给已实现的分析师（KNOWN_AGENTS），其余 target 一律归入
      "unassigned" 而不是丢弃——未知目标本身就是需要人看的覆盖问题；
    - 同时推进 revision_round（stuck 保险丝依赖这个，不能丢）；
    - 不再返回空的 final_report 或其他副作用。
    """
    challenges = (state.get("risk_review") or {}).get("challenges") or []
    pending: dict[str, list[dict]] = {}
    for ch in challenges:
        if not isinstance(ch, dict):
            ch = {"target_agent": "unassigned", "issue": str(ch), "required_action": ""}
        target = (ch.get("target_agent") or "").strip()
        bucket = target if target in KNOWN_AGENTS else "unassigned"
        pending.setdefault(bucket, []).append(ch)
    return {
        "revision_round": state.get("revision_round", 0) + 1,
        "pending_challenges": pending,
    }


def finalize_node(state: dict) -> dict:
    """成稿节点（占位）。"""
    return {"final_report": "stub report"}
