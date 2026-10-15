"""第六步：返工回环测试（全部 FakeLLM/FakeProvider，不联网、不 import akshare）。

覆盖：质询分发 -> 分析师消费 -> 复审降级为 low -> 成稿；未知 target 不丢失；
返工上限未变；stuck 保险丝未被改造破坏；风控 prompt 的时间锚点与边界感知条款。
"""

import json
from datetime import date

from src.graph.builder import build_graph, route_after_review
from src.graph.nodes import (
    FINANCIAL_ANALYST_SYSTEM_PROMPT,
    RISK_REVIEW_SYSTEM_PROMPT,
    make_financial_analyst_node,
    make_risk_review_node,
)
from src.tools.financial_data import FakeProvider

FAKE_FINANCIALS = {
    "ticker": "300750",
    "source": "fake",
    "report_period": "2026-06-30",
    "as_of": "2026-06",
    "roe": {"value": 11.41, "as_of": "2026-06"},
    "net_margin": {"value": 16.98, "as_of": "2026-06"},
    "debt_ratio": {"value": 63.65, "as_of": "2026-06"},
    "missing_metrics": ["gross_margin"],
}


def analysis_json(conclusion: str, finding: str) -> str:
    """构造一份合法的财务分析输出。"""
    return json.dumps(
        {
            "agent": "financial_analyst",
            "conclusion": conclusion,
            "confidence": "中",
            "key_points": [{"finding": finding, "data_source": "roe (2026-06)"}],
            "concerns": ["毛利率数据缺失"],
            "assumptions": ["假设报告期后无重大资产重组"],
        },
        ensure_ascii=False,
    )


ANALYSIS_V1 = analysis_json("增持", "ROE 11.41%，盈利质量较好")
ANALYSIS_V2 = analysis_json("中性", "已按质询重算口径，下调为中性")
ANALYSIS_V3 = analysis_json("中性", "第三轮修订")

CHALLENGE_ISSUE = "净利增速低于营收增速，却写成增长质量较好，属逻辑跳跃"
CHALLENGE_ACTION = "重算净利率口径，或删除该表述"


def risk_json(severity: str, challenges, summary: str = "风控总评") -> str:
    return json.dumps(
        {"severity": severity, "challenges": challenges, "summary": summary},
        ensure_ascii=False,
    )


FIN_CHALLENGE = {
    "target_agent": "financial_analyst",
    "issue": CHALLENGE_ISSUE,
    "required_action": CHALLENGE_ACTION,
}
RISK_HIGH = risk_json("high", [FIN_CHALLENGE])
RISK_LOW = risk_json("low", [], "质询已回应，风险可控")
COVERAGE_ONLY_HIGH = risk_json(
    "high",
    [
        {
            "target_agent": "industry_analyst",
            "issue": "大纲中的行业板块没有数据源，覆盖不足",
            "required_action": "补充行业分析",
        }
    ],
    "覆盖不足，无法采信",
)


class RoutingLLM:
    """按 system prompt 分发返回序列的假 LLM：分析师队列与风控队列分开。

    - messages 记录 (system, user)；analyst_prompts / risk_prompts 便于断言；
    - 队列最后一个元素重复使用（模拟稳定输出）。
    """

    def __init__(self, analyst_responses, risk_responses):
        self.queues = {
            FINANCIAL_ANALYST_SYSTEM_PROMPT: list(analyst_responses),
            RISK_REVIEW_SYSTEM_PROMPT: list(risk_responses),
        }
        self.calls = 0
        self.messages = []
        self.analyst_prompts = []
        self.risk_prompts = []

    def chat(self, system: str, user: str) -> str:
        assert system in self.queues, f"未知 system prompt: {system[:40]}"
        self.calls += 1
        self.messages.append((system, user))
        if system == FINANCIAL_ANALYST_SYSTEM_PROMPT:
            self.analyst_prompts.append(user)
        else:
            self.risk_prompts.append(user)

        queue = self.queues[system]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        return item


def build(llm, provider=None):
    provider = provider or FakeProvider({"300750": FAKE_FINANCIALS})
    return build_graph(financial_llm=llm, risk_llm=llm, data_provider=provider), provider


def test_revise_dispatches():
    """风控 high + 1 条质询 -> 第二次分析师 prompt 含质询原文，待办被清空。"""
    llm = RoutingLLM([ANALYSIS_V1, ANALYSIS_V2], [RISK_HIGH, RISK_LOW])
    app, provider = build(llm)
    result = app.invoke({"query": "宁德时代 300750", "revision_round": 0})

    assert len(llm.analyst_prompts) == 2, "返工后分析师必须重跑"
    assert CHALLENGE_ISSUE in llm.analyst_prompts[1]
    assert CHALLENGE_ACTION in llm.analyst_prompts[1]
    assert "上一轮风控质询" in llm.analyst_prompts[1]
    assert "上一轮风控质询" not in llm.analyst_prompts[0], "首轮不应带质询"
    assert provider.calls == ["300750", "300750"], "本步不做数据缓存，provider 每次调用"

    assert result["pending_challenges"]["financial_analyst"] == [], "待办必须被消费"
    assert result["revision_round"] == 1


def test_severity_drops_after_revision():
    """核心：high+质询 -> 修订 -> 复审降为 low -> 正常成稿。"""
    llm = RoutingLLM([ANALYSIS_V1, ANALYSIS_V2], [RISK_HIGH, RISK_LOW])
    app, _ = build(llm)
    result = app.invoke({"query": "宁德时代 300750", "revision_round": 0})

    assert result["risk_review"]["severity"] == "low"
    assert result["risk_review"]["challenges"] == []
    assert result["revision_round"] == 1
    assert result["analyses"]["financial_analyst"]["revised_in_round"] == 1
    assert result["analyses"]["financial_analyst"]["conclusion"] == "中性"
    assert result["final_report"] == "stub report"
    assert "risk_error" not in result
    assert len(llm.risk_prompts) == 2
    # 复审 prompt 必须带上「上一轮质询 + 本轮返工标记」，否则风控无法判断是否被回应
    assert CHALLENGE_ISSUE in llm.risk_prompts[1]
    assert "revised_in_round=1" in llm.risk_prompts[1]


def test_unknown_target_preserved():
    """未知 target 归入 unassigned 且不丢失，图不崩。"""
    llm = RoutingLLM([ANALYSIS_V1, ANALYSIS_V2], [COVERAGE_ONLY_HIGH, RISK_LOW])
    app, _ = build(llm)
    result = app.invoke({"query": "宁德时代 300750", "revision_round": 0})

    pending = result["pending_challenges"]
    assert "unassigned" in pending, "未知 target 不能丢"
    assert pending["unassigned"][0]["target_agent"] == "industry_analyst"
    assert pending["unassigned"][0]["issue"]  # 内容完整保留
    assert len(llm.analyst_prompts) == 2
    # 别人的质询不倒给财务分析师
    assert "行业板块没有数据源" not in llm.analyst_prompts[1]
    assert result["revision_round"] == 1
    assert result["final_report"] == "stub report"


def test_cap_still_enforced():
    """风控一直 high：仍在 2 轮封顶后成稿（上限语义不变）。"""
    llm = RoutingLLM(
        [ANALYSIS_V1, ANALYSIS_V2, ANALYSIS_V3], [RISK_HIGH, RISK_HIGH, RISK_HIGH]
    )
    app, provider = build(llm)
    result = app.invoke({"query": "宁德时代 300750", "revision_round": 0})

    assert result["revision_round"] == 2
    assert len(llm.analyst_prompts) == 3, "首轮 + 2 次返工"
    assert len(llm.risk_prompts) == 3
    assert provider.calls == ["300750"] * 3
    assert result["risk_review"]["severity"] == "high"
    assert result["final_report"] == "stub report"


def test_stuck_guard_intact():
    """保险丝回归：轮次没推进就不再返工（纯函数 + 节点两层）。"""
    assert (
        route_after_review(
            {"risk_review": {"severity": "high", "stuck": True}, "revision_round": 1}
        )
        == "finalize"
    )
    assert (
        route_after_review(
            {"risk_review": {"severity": "high", "stuck": False}, "revision_round": 1}
        )
        == "revise"
    )

    node = make_risk_review_node(RoutingLLM([], [RISK_LOW]))
    first = node({"query": "宁德时代 300750", "revision_round": 0})
    assert first["risk_review"]["stuck"] is False
    assert first["round_at_last_review"] == 0

    # 同一个 revision_round 又复核一次 -> 说明上一轮返工没推进轮次 -> stuck
    second = node(
        {
            "query": "宁德时代 300750",
            "revision_round": 0,
            "round_at_last_review": first["round_at_last_review"],
            "risk_review": first["risk_review"],
        }
    )
    assert second["risk_review"]["stuck"] is True


def test_time_and_coverage_in_prompt():
    """任务1：时间锚点 + 边界感知必须进 prompt（模型行为本身不做强断言）。"""
    llm = RoutingLLM([ANALYSIS_V1], [COVERAGE_ONLY_HIGH])
    node = make_risk_review_node(llm)
    out = node(
        {
            "query": "宁德时代 300750",
            "revision_round": 0,
            "analyses": {"financial_analyst": json.loads(ANALYSIS_V1)},
            "outline": {"sections": [{"title": "行业格局", "research_questions": ["?"]}]},
        }
    )

    prompt = llm.risk_prompts[0]
    assert str(date.today()) in prompt, "必须注入今天日期作为时间锚点"
    assert "当前系统日期" in prompt
    assert "akshare" in prompt and "知识截止" in prompt
    assert "数据口径" in prompt

    # 边界感知条款写在 system prompt 里
    assert "已知的任务边界" in RISK_REVIEW_SYSTEM_PROMPT
    assert "不应仅因覆盖不足就把 severity 判为 high" in RISK_REVIEW_SYSTEM_PROMPT
    assert "severity=high 只用于" in RISK_REVIEW_SYSTEM_PROMPT

    # 代码不得改写模型给出的 severity（prompt 到位即可，这里只确认没有越权改写）
    assert out["risk_review"]["severity"] == "high"


def test_analyst_consumes_challenges_directly():
    """节点级：直接把带待办的 state 交给分析师，确认回应要求进了 prompt。"""
    llm = RoutingLLM([ANALYSIS_V2], [])
    node = make_financial_analyst_node(llm, FakeProvider({"300750": FAKE_FINANCIALS}))
    out = node(
        {
            "query": "宁德时代 300750",
            "revision_round": 1,
            "pending_challenges": {"financial_analyst": [FIN_CHALLENGE]},
        }
    )

    user_prompt = llm.messages[0][1]
    assert CHALLENGE_ISSUE in user_prompt
    assert "必须逐条正面回应" in user_prompt
    assert out["analyses"]["financial_analyst"]["revised_in_round"] == 1
    assert out["pending_challenges"]["financial_analyst"] == []
