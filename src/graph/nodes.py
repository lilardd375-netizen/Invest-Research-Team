"""占位节点 + 各 Agent 节点工厂。"""

import json
import re

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
- high   = 存在数据错误或重大逻辑漏洞，结论不可采信，应返工；
- medium = 假设偏激进但可接受，报告中披露即可；
- low    = 常规提示，并入风险章节即可。

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
    """

    def financial_analyst_node(state: dict) -> dict:
        query = state.get("query", "")
        ticker = extract_ticker(query)
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
            failure["analyst_error"] = failure["analyses"]["financial_analyst"]["error"]
            return failure
        return {"analyses": {"financial_analyst": payload.model_dump()}}

    return financial_analyst_node


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


def _build_risk_user_prompt(state: dict) -> str:
    """组装风控的 user prompt：query / analyses / outline 概要 / 复审上下文。"""
    previous = (state.get("risk_review") or {}).get("severity")
    revision_round = state.get("revision_round", 0)
    if previous:
        round_hint = (
            f"这是第 {revision_round} 轮返工后的复审"
            f"（上一轮风控结论：{previous}）。请重点确认上一轮的问题是否已解决，"
            "以及返工是否引入了新的问题。"
        )
    else:
        round_hint = f"这是首轮评审（revision_round={revision_round}）。"

    return (
        f"研究标的：{state.get('query', '')}\n\n"
        "== 各分析 Agent 的输出（JSON，含 unavailable/error 信息）==\n"
        f"{json.dumps(state.get('analyses') or {}, ensure_ascii=False, default=str)}\n\n"
        "== 研究大纲概要 ==\n"
        f"{_outline_digest(state.get('outline'))}\n\n"
        "== 评审上下文 ==\n"
        f"{round_hint}\n\n"
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
    """返工节点（占位）。

    本步只推进轮次，不把 challenges 发回分析师重新分析——那是第六步。
    """
    return {"revision_round": state["revision_round"] + 1}


def finalize_node(state: dict) -> dict:
    """成稿节点（占位）。"""
    return {"final_report": "stub report"}
