"""占位节点 + 各 Agent 节点工厂。"""

import json
import re
from datetime import date, datetime

from pydantic import ValidationError

from src.llm.errors import LLMFatalError, LLMTransientError
from src.llm.schemas import (
    FinancialAnalysisModel,
    IndustryAnalysisModel,
    OutlineModel,
    ReportModel,
    RiskReviewModel,
    SentimentAnalysisModel,
    TechnicalAnalysisModel,
)
from src.report.render import build_rule_based_report, render_report
from src.tools.utils import normalize_ticker

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

# 行业分析 Agent 的 system prompt（模块级常量，中文）。
INDUSTRY_ANALYST_SYSTEM_PROMPT = """你是一名行业分析 Agent，服务于投资研究团队。

你会收到某标的的行业数据（所属板块、板块近期表现、同业关键指标对比）。要求：

1. 只依据给定数据下结论，禁止编造缺失的数据；数据缺失就写进 assumptions 或 concerns。
2. key_points 每条必须标注 data_source，写清引用的是哪个字段
   （例如 "板块涨跌幅"），且 finding 里带上具体数值。
3. 同业对比必须点名具体公司与指标，不要只给"行业地位领先"这类空结论。
4. conclusion 只能是 增持 / 中性 / 减持 三者之一（判断该标的在行业中的相对位置）。
5. confidence 只能是 高 / 中 / 低 三者之一；同业样本少、板块数据缺失就越低。
6. concerns 写行业风险（产能过剩、价格战、政策变化等），assumptions 写结论依赖的假设。

只输出 JSON，不要其他文字。JSON 结构必须是：
{
  "agent": "industry_analyst",
  "conclusion": "增持|中性|减持",
  "confidence": "高|中|低",
  "key_points": [{"finding": "...", "data_source": "..."}],
  "concerns": ["..."],
  "assumptions": ["..."]
}"""

# 舆情分析 Agent 的 system prompt（模块级常量，中文）。
SENTIMENT_ANALYST_SYSTEM_PROMPT = """你是一名舆情分析 Agent，服务于投资研究团队。

你会收到某标的最近若干条新闻（标题 + 摘要 + 发布时间）。要求：

1. 只依据给定新闻下结论，禁止编造新闻或数据；样本少就降低 confidence。
2. key_points 每条必须标注 data_source，写清引用的是哪条新闻
   （例如 "新闻：<标题>（2026-06-30）"），finding 里说明该信息的性质。
3. 区分事实与情绪：媒体的用词倾向、标题情绪与实质经营变化要分开评价。
4. 单条新闻不能代表趋势；只有多条同向新闻才可支撑较强结论，
   并在 assumptions 里写明样本范围（条数、时间跨度）。
5. conclusion 只能是 增持 / 中性 / 减持 三者之一（站在市场情绪角度）。
6. confidence 只能是 高 / 中 / 低 三者之一；concerns 写负面舆情与信息可信度风险。

只输出 JSON，不要其他文字。JSON 结构必须是：
{
  "agent": "sentiment_analyst",
  "conclusion": "增持|中性|减持",
  "confidence": "高|中|低",
  "key_points": [{"finding": "...", "data_source": "..."}],
  "concerns": ["..."],
  "assumptions": ["..."]
}"""

# 技术面分析 Agent 的 system prompt（模块级常量，中文）。
TECHNICAL_ANALYST_SYSTEM_PROMPT = """你是一名技术面分析 Agent，服务于投资研究团队。

你会收到某标的的日线统计指标（均线、区间涨跌幅与高低点、量比等，均由历史价格算出）。要求：

1. 只依据给定指标下结论，禁止编造价格或指标数值；缺失就写进 assumptions 或 concerns。
2. key_points 每条必须标注 data_source，写清引用的是哪个指标
   （例如 "MA20 (2026-06-30)"），finding 里带上具体数值。
3. 明确技术面只反映历史价格行为，不能替代基本面判断；
   不要给出精确目标价，也不要承诺涨跌。
4. 均线多空排列、价格相对均线的位置、区间位置、量能变化要分别说明。
5. conclusion 只能是 增持 / 中性 / 减持 三者之一（站在价格趋势角度）。
6. confidence 只能是 高 / 中 / 低 三者之一；区间震荡或量能异常时降低它。

只输出 JSON，不要其他文字。JSON 结构必须是：
{
  "agent": "technical_analyst",
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

任务边界（已知，不是缺陷）：当前已实现的分析师为 financial / industry / sentiment / technical 四个。
估值、宏观、政策等维度暂无数据源是已知的任务边界：可以在 challenges 或 summary 里
提示覆盖缺口，但不应仅因覆盖不足就把 severity 判为 high。
severity=high 只用于：存在数据错误、口径错误、或已覆盖维度的逻辑漏洞。

验算义务（重要）：质疑指标一致性（"口径矛盾""无法自洽"）之前，必须先做恒等式验算：
ROE(摊薄) ≈ 总资产利润率 × 权益乘数；总资产净利润率 ≈ 销售净利率 × 总资产周转率。
验算通过（偏差在 5% 以内）就不得再以口径矛盾为由质询。若要质疑口径，必须写明：
① 具体是哪一项的哪种口径错（利润总额 vs 净利润、摊薄 vs 加权、归母 vs 全部权益、
单期 vs 年化），② 你的验算算式与数值。另外比率多为季度 / 半年度累计值而非年化值，
不要用年化阈值（例如"总资产周转率应高于 1"）直接判断高低。

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

# 送进风控 prompt 的 analyses JSON 体量上限：超过就瘦身（每 Agent 只保留
# conclusion / confidence / key_points 前 N 条 / status 等关键字段），
# 避免四个 Agent 全量输出把 prompt 撑爆。
MAX_ANALYSES_JSON_CHARS = 6000
MAX_KEY_POINTS_IN_DIGEST = 4

# 报告合成 prompt 的质询历史上限：全部轮次合计最多 12 条（超了从旧轮开始丢，
# 最新一轮全量保留），每条正文截断到 200 字符。按最坏情况设计，不依赖单次冒烟实测。
MAX_CHALLENGES_TOTAL = 12
CHALLENGE_TEXT_CHARS = 200
# 合成 LLM 没有逐家呈现时，兜底补齐用的字段。
REPORT_SUMMARY_FALLBACK = "（合成 LLM 未逐家呈现，此处按原始分析结果补齐）"

# 报告合成 Agent 的 system prompt（模块级常量，中文）。
REPORT_SYSTEM_PROMPT = """你是投资研究团队的报告主编，负责把四个分析师的结论、风控的多轮质询与返工记录，合成一份诚实的机构研究报告 JSON。

硬性要求：
1. 逐家呈现：per_agent_summaries 必须覆盖输入里出现的每一个分析师（财务/行业/舆情/技术面），
   哪怕结论互相矛盾也要保留分歧，绝对不许只给一句「综合结论」把分歧吞掉。
   status 为 unavailable 的分析师也要列出，conclusion 留空、summary 说明其不可用原因。
2. 综合结论只能来自四家已有结论（多数且过半才给方向，票数分散就给「中性」），
   不许引入任何外部评级数据。
3. key_thesis 给 3-5 条支撑逻辑，key_risks 给主要风险（含风控质询里未解决的问题）。
4. revision_history 必须照抄输入的每一轮：round / severity / degraded / challenge_targets /
   revised_agents，degraded=true 的轮次要保留该标记，不许抹平成正常评审。
5. disclosures 必须收编：全部 medium 质询、数据源或 LLM 降级情况、风控降级轮次、
   假设与局限（覆盖边界：只有财务/行业/舆情/技术面四个维度，估值/宏观/政策无数据源）。
6. 只输出 JSON，不要任何其他文字。

输出 JSON 结构：
{
  "title": "报告标题",
  "stock_code": "6 位代码",
  "report_date": "YYYY-MM-DD",
  "overall_conclusion": "增持/中性/减持",
  "overall_confidence": "高/中/低",
  "per_agent_summaries": [
    {"agent": "financial_analyst", "status": "ok", "conclusion": "增持", "confidence": "中", "summary": "一句话要点"}
  ],
  "key_thesis": ["支撑逻辑"],
  "key_risks": ["风险"],
  "revision_history": [
    {"round": 0, "severity": "high", "degraded": false, "challenge_targets": {"financial_analyst": 4},
     "revised_agents": [], "note": "风控总评"}
  ],
  "disclosures": ["必须披露的事项"],
  "generated_at": "ISO 时间"
}"""

# 已实现的分析师（revise 只把质询分发给它们，其余归 unassigned）。
KNOWN_AGENTS = (
    "financial_analyst",
    "industry_analyst",
    "sentiment_analyst",
    "technical_analyst",
)

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
        if _should_skip(state, "financial_analyst"):
            return {}

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
        "或明确说明为何不采纳，禁止无视。\n"
        "硬约束：对每条质询，必须要么给出重算/修正后的数值与依据，"
        "要么明确说明无法修正及原因；禁止仅表态承认而不修正。"
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


def _missing_domain_analyst_stub_node(state: dict) -> dict:
    """未注入 LLM 的新分析 Agent：整节点跳过（不写 analyses，旧图行为不变）。"""
    return {}


def _should_skip(state: dict, agent: str) -> bool:
    """幂等跳过：本 Agent 已有结果、且没有发给自己的新质询 -> 不调 LLM。

    revise 之后质询会 fan-out 给全部分析师，无事者靠这个守卫空转跳过；
    否则每轮返工都会把所有 Agent 重跑一遍（浪费调用，还会改坏没被质询的结论）。
    """
    if (state.get("pending_challenges") or {}).get(agent):
        return False
    return agent in (state.get("analyses") or {})


def _make_domain_analyst_node(
    llm, data_provider, *, agent, model_cls, system_prompt, data_label, fetch
):
    """行业 / 舆情 / 技术面分析节点的公共实现（财务 Agent 保持独立实现，但收尾时也补了同样的幂等跳过）。

    fetch(provider, ticker) -> dict：取该维度数据，数据不可用时由 fetch 抛异常。
    契约与财务 Agent 一致：数据源故障不调 LLM、降级 status="unavailable"；
    LLM 错误按可重试/不可重试分类；消费自己的待办质询并标记 revised_in_round。
    差异：降级写 analyst_errors[agent]（or_ reducer 通道），不动财务 Agent 的单数 analyst_error。
    """

    def domain_analyst_node(state: dict) -> dict:
        if _should_skip(state, agent):
            return {}

        query = state.get("query", "")
        ticker = extract_ticker(query)
        pending = dict(state.get("pending_challenges") or {})
        challenges = pending.get(agent) or []

        try:
            dataset = fetch(data_provider, ticker)
        except Exception as exc:
            summary = f"数据源不可用: {type(exc).__name__}: {exc}"
            return {
                "analyses": {
                    agent: {
                        "status": "unavailable",
                        "error": summary,
                        "error_kind": "data_source",
                    }
                },
                "analyst_errors": {agent: summary},
            }

        user_prompt = (
            f"研究标的：{query}\n"
            f"提取到的股票代码：{ticker}\n"
            f"{data_label}（JSON）："
            f"{json.dumps(dataset, ensure_ascii=False, default=str)}\n\n"
            "请基于以上数据输出分析 JSON。"
        )
        if challenges:
            user_prompt = f"{user_prompt}\n\n{_challenge_block(challenges)}"

        payload, failure = _retry_or_fail(
            lambda: llm.chat(system_prompt, user_prompt),
            model_cls,
            lambda kind, summary: {
                "analyses": {
                    agent: {
                        "status": "unavailable",
                        "error": summary,
                        "error_kind": kind,
                    }
                }
            },
        )
        if failure is not None:
            # 质询没被正面回应，不清空待办；错误另行汇总到 analyst_errors
            failure["analyst_errors"] = {agent: failure["analyses"][agent]["error"]}
            return failure

        analysis = payload.model_dump()
        analysis["revised_in_round"] = state.get("revision_round", 0)
        result = {"analyses": {agent: analysis}}
        if challenges:
            # 只写自己那一格：or_ 合并下等价于清空自己的待办，且不会覆盖别人的
            result["pending_challenges"] = {agent: []}
        return result

    return domain_analyst_node


def make_industry_analyst_node(llm, data_provider):
    """行业分析节点工厂：llm.chat + IndustryProvider。"""
    return _make_domain_analyst_node(
        llm,
        data_provider,
        agent="industry_analyst",
        model_cls=IndustryAnalysisModel,
        system_prompt=INDUSTRY_ANALYST_SYSTEM_PROMPT,
        data_label="行业数据（所属板块 / 板块近期表现 / 同业关键指标）",
        fetch=lambda provider, ticker: provider.get_industry(ticker),
    )


def make_sentiment_analyst_node(llm, data_provider):
    """舆情分析节点工厂：llm.chat + SentimentProvider。"""
    return _make_domain_analyst_node(
        llm,
        data_provider,
        agent="sentiment_analyst",
        model_cls=SentimentAnalysisModel,
        system_prompt=SENTIMENT_ANALYST_SYSTEM_PROMPT,
        data_label="近期新闻（标题 / 摘要截断 / 发布时间）",
        fetch=lambda provider, ticker: provider.get_sentiment(ticker),
    )


def make_technical_analyst_node(llm, data_provider):
    """技术面分析节点工厂：llm.chat + TechnicalProvider。"""
    return _make_domain_analyst_node(
        llm,
        data_provider,
        agent="technical_analyst",
        model_cls=TechnicalAnalysisModel,
        system_prompt=TECHNICAL_ANALYST_SYSTEM_PROMPT,
        data_label="日线技术指标（均线 / 区间涨跌与高低点 / 量比）",
        fetch=lambda provider, ticker: provider.get_technical(ticker),
    )


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


def _analyses_digest(analyses) -> str:
    """序列化各 Agent 输出；超过体量上限就瘦身，只留关键字段。"""
    blob = json.dumps(analyses, ensure_ascii=False, default=str)
    if len(blob) <= MAX_ANALYSES_JSON_CHARS:
        return blob

    keep_fields = ("conclusion", "confidence", "status", "error", "error_kind", "revised_in_round")
    slim = {}
    for name, value in (analyses or {}).items():
        if not isinstance(value, dict):
            slim[name] = value
            continue
        slim[name] = {field: value[field] for field in keep_fields if field in value}
        points = value.get("key_points")
        if isinstance(points, list):
            slim[name]["key_points"] = points[:MAX_KEY_POINTS_IN_DIGEST]
    return (
        json.dumps(slim, ensure_ascii=False, default=str)
        + f"\n（注：原始 analyses 共 {len(blob)} 字符，超上限已截断，"
        f"每 Agent 只保留关键字段与 key_points 前 {MAX_KEY_POINTS_IN_DIGEST} 条）"
    )


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
        f"{_analyses_digest(analyses)}\n\n"
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
        result = _with_fuse(state, review, degraded=failure is not None)
        if failure is not None:
            # risk_review 里已带 error_kind/error，顶层再给一份摘要供下游消费
            result["risk_error"] = (
                f"{review['error_kind']}: {review['error']}"
            )
        return result

    return risk_review_node_with_llm


def _challenge_target_counts(challenges) -> dict:
    """统计质询按 target_agent 的分布（报告「返工记录」章节用）。"""
    counts: dict[str, int] = {}
    for item in challenges or []:
        if isinstance(item, dict):
            target = (item.get("target_agent") or "").strip() or "unassigned"
        else:
            target = "unassigned"
        counts[target] = counts.get(target, 0) + 1
    return counts


def _revised_this_round(state: dict, revision_round: int) -> list:
    """本轮真正被返工改过的分析师（首轮不是返工，返回空列表）。"""
    if revision_round <= 0:
        return []
    marks = []
    for name, value in (state.get("analyses") or {}).items():
        if isinstance(value, dict) and value.get("revised_in_round") == revision_round:
            marks.append(name)
    return sorted(marks)


def _with_fuse(state: dict, review: dict, *, degraded: bool = False) -> dict:
    """占位节点的原有职责（保险丝），LLM 路径与占位路径共用。

    本轮 revision_round 与上次复核时相同 -> 上一轮返工没推进轮次，判定 stuck。

    第八步新增两件事：
    1. 把本轮评审追加进 review_history（risk_review 通道是 LastValue，最后一轮会
       覆盖前几轮，报告需要逐轮留档才能写出「返工记录」章节）；
    2. 显式标记 degraded —— 降级时写出的 severity 是兜底值，"没评成" 和
       "评出来是 low" 必须在 state 里可区分，否则报告会把降级当成结论。
    """
    revision_round = state.get("revision_round", 0)
    stuck = revision_round == state.get("round_at_last_review", -1)
    merged = dict(review)
    merged["stuck"] = stuck
    merged["degraded"] = degraded
    challenges = [
        dict(item)
        if isinstance(item, dict)
        else {"target_agent": "unassigned", "issue": str(item), "required_action": ""}
        for item in (merged.get("challenges") or [])
    ]
    entry = {
        "round": revision_round,
        "severity": merged.get("severity", ""),
        "degraded": degraded,
        "summary": merged.get("summary", ""),
        "challenges": challenges,
        "challenge_targets": _challenge_target_counts(challenges),
        "revised_agents": _revised_this_round(state, revision_round),
    }
    return {
        "risk_review": merged,
        "round_at_last_review": revision_round,
        "risk_degraded": degraded,
        "risk_degraded_round": revision_round if degraded else -1,
        "review_history": [entry],
    }


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


def _truncate_text(text, limit: int = CHALLENGE_TEXT_CHARS) -> str:
    """把一条质询正文压成单行并截断到上限。"""
    flat = " ".join(str(text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def _history_entries(state: dict) -> list:
    """取逐轮风控留档；没有时退化成当前 risk_review（兼容单节点调用与旧状态）。"""
    history = [entry for entry in (state.get("review_history") or []) if isinstance(entry, dict)]
    if history:
        return history
    review = state.get("risk_review") or {}
    if not review:
        return []
    challenges = [item for item in (review.get("challenges") or []) if isinstance(item, dict)]
    return [
        {
            "round": state.get("revision_round", 0),
            "severity": review.get("severity", ""),
            "degraded": bool(review.get("degraded")),
            "summary": review.get("summary", ""),
            "challenges": challenges,
            "challenge_targets": _challenge_target_counts(challenges),
            "revised_agents": [],
        }
    ]


def _challenge_history_digest(state: dict) -> str:
    """把全部轮次的风控质询压成 prompt 区块。

    上限按最坏情况设计：全部轮次合计 ≤ MAX_CHALLENGES_TOTAL 条，
    旧轮每个 target 只留 1 条代表，超限时从最早的开始丢（最新一轮全量保留），
    每条正文截断到 CHALLENGE_TEXT_CHARS 字符。
    """
    history = _history_entries(state)
    selected: list[tuple[int, dict]] = []
    for index, entry in enumerate(history):
        challenges = [item for item in (entry.get("challenges") or []) if isinstance(item, dict)]
        round_no = int(entry.get("round", index) or 0)
        if index == len(history) - 1:
            selected.extend((round_no, item) for item in challenges)
            continue
        seen: set[str] = set()
        for item in challenges:
            target = (item.get("target_agent") or "").strip() or "unassigned"
            if target in seen:
                continue
            seen.add(target)
            selected.append((round_no, item))
    dropped = max(0, len(selected) - MAX_CHALLENGES_TOTAL)
    if dropped:
        selected = selected[dropped:]
    lines = []
    for round_no, item in selected:
        target = (item.get("target_agent") or "").strip() or "unassigned"
        issue = _truncate_text(item.get("issue"))
        action = _truncate_text(item.get("required_action"))
        lines.append(f"- [第 {round_no} 轮][{target}] {issue}（要求：{action}）")
    if not lines:
        return "（没有任何轮次产生质询）"
    if dropped:
        lines.append(f"- …另有 {dropped} 条较早的质询按上限省略")
    return "\n".join(lines)


def _review_history_digest(state: dict) -> str:
    """逐轮列出 severity / 质询分布 / 谁修订过 / 是否降级（合成报告要照抄）。"""
    history = _history_entries(state)
    if not history:
        return "（还没有风控评审记录）"
    lines = []
    for entry in history:
        targets = entry.get("challenge_targets") or _challenge_target_counts(entry.get("challenges"))
        target_text = "、".join(f"{key} {value} 条" for key, value in targets.items()) or "无"
        revised = "、".join(entry.get("revised_agents") or []) or "无"
        flag = "（降级产物，非真实评审结论）" if entry.get("degraded") else ""
        lines.append(
            f"- 第 {entry.get('round', 0)} 轮：severity={entry.get('severity') or '—'}{flag}"
            f"；质询分布：{target_text}；本轮修订者：{revised}"
        )
        summary = _truncate_text(entry.get("summary"))
        if summary:
            lines.append(f"  - 风控总评：{summary}")
    return "\n".join(lines)


def _build_report_user_prompt(state: dict) -> str:
    """组装报告合成的 user prompt：四家输出 + 全部质询 + 逐轮 severity/返工记录。"""
    ticker = extract_ticker(state.get("query", ""))
    return (
        f"当前系统日期：{date.today().isoformat()}。\n\n"
        f"研究标的：{state.get('query', '')}（股票代码：{ticker}）\n\n"
        "== 四个分析师的输出（JSON，含 unavailable/error）==\n"
        f"{_analyses_digest(state.get('analyses'))}\n\n"
        "== 研究大纲概要 ==\n"
        f"{_outline_digest(state.get('outline'))}\n\n"
        "== 全部轮次的风控质询（已按上限截断）==\n"
        f"{_challenge_history_digest(state)}\n\n"
        "== 每轮 severity 与返工记录（必须照抄进 revision_history）==\n"
        f"{_review_history_digest(state)}\n\n"
        f"风控降级标记：risk_degraded={bool(state.get('risk_degraded'))}，"
        f"risk_degraded_round={state.get('risk_degraded_round', -1)}"
        "（为 True 时该轮 severity 是兜底值，必须在 disclosures 里披露）。\n\n"
        "请按 system 要求合成报告 JSON。"
    )


def _complete_report(report: dict, state: dict) -> dict:
    """兜底补齐合成 LLM 可能漏掉的东西：逐家结论与返工记录都不许丢。"""
    analyses = state.get("analyses") or {}
    report = dict(report)
    # 元信息以代码为准：合成 LLM 会自己编 generated_at（实测编出 00:00:00Z），不可信。
    report["stock_code"] = str(report.get("stock_code") or "").strip() or (
        normalize_ticker(state.get("query", "")) or "unknown"
    )
    report["report_date"] = str(report.get("report_date") or "").strip() or date.today().isoformat()
    report["title"] = str(report.get("title") or "").strip() or f"{report['stock_code']} 投资研究报告"
    report["generated_at"] = datetime.now().isoformat(timespec="seconds")
    summaries = [item for item in (report.get("per_agent_summaries") or []) if isinstance(item, dict)]
    present = {str(item.get("agent") or "") for item in summaries}
    missing = [name for name in KNOWN_AGENTS if name in analyses and name not in present]
    if missing:
        for name in missing:
            value = analyses.get(name) or {}
            summaries.append(
                {
                    "agent": name,
                    "status": str(value.get("status") or "ok"),
                    "conclusion": str(value.get("conclusion") or ""),
                    "confidence": str(value.get("confidence") or ""),
                    "summary": REPORT_SUMMARY_FALLBACK,
                }
            )
        report["per_agent_summaries"] = summaries
        report["disclosures"] = list(report.get("disclosures") or []) + [
            "报告合成 LLM 未逐家呈现下列分析师，已按原始结果补齐：" + "、".join(missing)
        ]
    if not (report.get("revision_history") or []) and _history_entries(state):
        report["revision_history"] = [
            {
                "round": int(entry.get("round", 0) or 0),
                "severity": str(entry.get("severity") or ""),
                "degraded": bool(entry.get("degraded")),
                "challenge_targets": dict(entry.get("challenge_targets") or {}),
                "revised_agents": list(entry.get("revised_agents") or []),
                "note": _truncate_text(entry.get("summary")),
            }
            for entry in _history_entries(state)
        ]
        report["disclosures"] = list(report.get("disclosures") or []) + [
            "返工记录由 state 的 review_history 补齐（合成 LLM 未给出）。"
        ]
    return report


def make_finalize_node(llm):
    """成稿节点工厂：注入 report_llm，合成 ReportModel 并渲染 Markdown。

    失败兜底：LLM 不可用 / 校验失败 -> build_rule_based_report 规则式组装，
    写 report_llm_degraded=True —— 合成失败绝不允许丢掉四个分析师和两轮返工的结果。
    """

    def finalize_node_with_llm(state: dict) -> dict:
        prompt = _build_report_user_prompt(state)
        payload, failure = _retry_or_fail(
            lambda: llm.chat(REPORT_SYSTEM_PROMPT, prompt),
            ReportModel,
            lambda kind, summary: {
                "report_llm_degraded": True,
                "report_error": f"{kind}: {summary}",
            },
        )
        if failure is not None:
            # 降级路径：把降级标记先塞进 state 再组装/渲染，否则报告里
            # 「本报告为降级产物」的披露与横幅不会出现（合成失败却不披露＝骗读者）。
            degraded_state = {
                **state,
                "report_llm_degraded": True,
                "report_error": failure.get("report_error", ""),
            }
            rule_based = build_rule_based_report(degraded_state)
            failure["report"] = rule_based
            failure["final_report"] = render_report({**degraded_state, "report": rule_based})
            return failure
        report = _complete_report(payload.model_dump(), state)
        return {
            "report": report,
            "final_report": render_report({**state, "report": report}),
            "report_llm_degraded": False,
            "report_error": "",
        }

    return finalize_node_with_llm


def finalize_node(state: dict) -> dict:
    """成稿节点（未注入 report_llm）：规则式组装报告并渲染 Markdown。

    第八步起 finalize 不再是 stub：没有合成 LLM 也要产出完整报告，并显式标注
    report_llm_degraded=True（规则式组装本身就是合成降级路径）。
    """
    note = "未启用报告合成 LLM：走规则式组装"
    degraded_state = {**state, "report_llm_degraded": True, "report_error": note}
    report = build_rule_based_report(degraded_state)
    return {
        "report": report,
        "final_report": render_report({**degraded_state, "report": report}),
        "report_llm_degraded": True,
        "report_error": note,
    }
