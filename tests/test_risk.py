"""风控 Agent 测试：FakeLLM，不联网。"""

import json

from src.graph.builder import build_graph
from src.graph.nodes import make_risk_review_node, risk_review_node
from src.llm.errors import LLMFatalError, LLMTransientError

VALID_HIGH = json.dumps(
    {
        "severity": "high",
        "challenges": [
            {
                "target_agent": "financial_analyst",
                "issue": "营收增速 54.8% 被用来支撑长期增长，但未扣除并表因素",
                "required_action": "拆分内生增长与并表贡献",
            }
        ],
        "summary": "增长质量的论证存在重大逻辑漏洞，应返工。",
    },
    ensure_ascii=False,
)

VALID_LOW = json.dumps(
    {
        "severity": "low",
        "challenges": [
            {
                "target_agent": "financial_analyst",
                "issue": "仅单期数据，缺少历史趋势",
                "required_action": "报告中注明数据期间",
            }
        ],
        "summary": "结论方向可用，属常规提示。",
    },
    ensure_ascii=False,
)

UNAVAILABLE_ANALYSES = {
    "financial_analyst": {
        "status": "unavailable",
        "error": "数据源不可用: KeyError: 未配置 ticker 的假数据: 300750",
        "error_kind": "data_source",
    }
}


class FakeLLM:
    """可预设返回序列的假 LLM，记录收到的 (system, user)，元素可为异常实例。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.messages = []

    def chat(self, system: str, user: str) -> str:
        self.calls += 1
        self.messages.append((system, user))
        idx = min(self.calls - 1, len(self.responses) - 1)
        item = self.responses[idx]
        if isinstance(item, BaseException):
            raise item
        return item

    def plan_outline(self, query: str) -> str:
        return self.chat("", query)


def test_risk_valid_low():
    """severity=low：结构正确、无 risk_error、直接 finalize。"""
    llm = FakeLLM([VALID_LOW])
    node = make_risk_review_node(llm)
    out = node({"query": "宁德时代 300750", "analyses": {}, "revision_round": 0})

    review = out["risk_review"]
    assert review["severity"] == "low"
    assert len(review["challenges"]) == 1
    assert review["challenges"][0]["target_agent"] == "financial_analyst"
    assert review["summary"]
    assert review["stuck"] is False  # 保险丝字段仍在
    assert out["round_at_last_review"] == 0  # 占位逻辑已迁移
    assert "risk_error" not in out
    assert llm.calls == 1


def test_risk_low_finalize_path_in_full_graph():
    """整图验证 low 走 finalize：不返工、无 risk_error。"""
    llm = FakeLLM([VALID_LOW])
    app = build_graph(risk_llm=llm)
    result = app.invoke({"query": "宁德时代", "revision_round": 0})

    assert result["risk_review"]["severity"] == "low"
    assert result["revision_round"] == 0
    # 第八步：finalize 不再是 stub，改为规则式组装并渲染 Markdown 报告
    assert result["final_report"].startswith("# ")
    assert "risk_error" not in result


def test_risk_high_triggers_revise():
    """severity=high 真实驱动回环：第 1 次质疑 -> revise，第 2 次放行 -> finalize。"""
    llm = FakeLLM([VALID_HIGH, VALID_LOW])
    app = build_graph(risk_llm=llm)
    result = app.invoke({"query": "宁德时代", "revision_round": 0})

    assert llm.calls == 2  # 被复审了第二次，说明确实走了 revise 分支
    assert result["revision_round"] == 1  # revise 推进了轮次
    assert result["risk_review"]["severity"] == "low"  # 第二轮放行
    assert result["risk_review"]["stuck"] is False
    assert result["round_at_last_review"] == 1  # 已回写为最新轮次
    # 第八步：finalize 不再是 stub，改为规则式组装并渲染 Markdown 报告
    assert result["final_report"].startswith("# ")


def test_risk_high_until_cap():
    """一直 high：返工到轮次上限就停，不许无限循环。"""
    llm = FakeLLM([VALID_HIGH])  # 无限重复 high
    app = build_graph(risk_llm=llm)
    result = app.invoke({"query": "宁德时代", "revision_round": 0})

    assert result["revision_round"] == 2  # MAX_REVISION_ROUNDS
    assert result["risk_review"]["severity"] == "high"
    assert result["round_at_last_review"] == 2
    # 第八步：finalize 不再是 stub，改为规则式组装并渲染 Markdown 报告
    assert result["final_report"].startswith("# ")
    assert llm.calls == 3  # 首轮 + 两轮返工后各复核一次


def test_risk_transient_retry():
    """可重试错误重试一次后成功（与 planner 同模式）。"""
    llm = FakeLLM([LLMTransientError("JSON 解析失败: 坏输出"), VALID_LOW])
    node = make_risk_review_node(llm)
    out = node({"query": "宁德时代", "analyses": {}, "revision_round": 0})

    assert llm.calls == 2
    assert out["risk_review"]["severity"] == "low"
    assert "risk_error" not in out


def test_risk_fatal_no_retry():
    """不可重试错误（401）不浪费重试：只调 1 次。"""
    llm = FakeLLM([LLMFatalError("鉴权失败（不可重试）: AuthenticationError")])
    node = make_risk_review_node(llm)
    out = node({"query": "宁德时代", "analyses": {}, "revision_round": 0})

    assert llm.calls == 1
    review = out["risk_review"]
    assert review["severity"] == "low"  # 降级为合法枚举值
    assert review["challenges"] == []
    assert review["error_kind"] == "fatal"
    assert "鉴权失败" in review["error"]


def test_risk_error_degrades_to_low():
    """Fatal 降级后：severity=low、risk_error 非空、整图正常成稿不卡死。"""
    llm = FakeLLM([LLMFatalError("鉴权失败（不可重试）: AuthenticationError")])
    app = build_graph(risk_llm=llm)
    result = app.invoke({"query": "宁德时代", "revision_round": 0})

    assert result["risk_review"]["severity"] == "low"
    assert result["risk_error"]
    assert "fatal" in result["risk_error"]
    # 第八步：finalize 不再是 stub，改为规则式组装并渲染 Markdown 报告
    assert result["final_report"].startswith("# ")  # 风控挂了不卡死报告
    assert result["revision_round"] == 0


def test_risk_sees_unavailable_analyst():
    """分析师挂掉的信息必须出现在风控 prompt 里。"""
    llm = FakeLLM([VALID_HIGH])
    node = make_risk_review_node(llm)
    node(
        {
            "query": "宁德时代 300750",
            "analyses": UNAVAILABLE_ANALYSES,
            "outline": {"sections": [{"title": "公司概况", "research_questions": ["a"]}]},
            "revision_round": 0,
        }
    )

    system, user = llm.messages[0]
    assert "unavailable" in user  # 看到了 unavailable 标记
    assert "数据源不可用" in user  # 也看到了具体错误信息
    assert "data_source" in user  # 以及 error_kind
    assert "反方辩手" in system  # 角色定位
    assert "只输出 JSON" in system
    assert "宁德时代 300750" in user  # 标的
    assert "公司概况" in user  # 大纲概要
    assert "首轮评审" in user  # 首轮上下文


def test_risk_prompt_carries_review_context():
    """复审时 prompt 要带上上一轮结论，让风控知道自己在复审。"""
    llm = FakeLLM([VALID_LOW])
    node = make_risk_review_node(llm)
    node(
        {
            "query": "宁德时代",
            "analyses": {},
            "risk_review": {"severity": "high", "stuck": False},
            "revision_round": 1,
        }
    )

    _, user = llm.messages[0]
    assert "复审" in user
    assert "high" in user
    assert "1" in user


def test_risk_stub_compat():
    """不传 risk_llm：旧占位行为不变（原样回显 severity）。"""
    # 第八步：risk_review 里多了 degraded 标记（占位路径恒为 False），
    # 用于把「降级写死的 low」与「真实评审的 low」区分开。
    assert risk_review_node({"revision_round": 0})["risk_review"] == {
        "severity": "low",
        "stuck": False,
        "degraded": False,
    }

    out = risk_review_node(
        {"risk_review": {"severity": "high"}, "revision_round": 3, "round_at_last_review": 2}
    )
    assert out["risk_review"]["severity"] == "high"
    assert out["risk_review"]["stuck"] is False
    assert out["round_at_last_review"] == 3

    # 未推进轮次 -> stuck 保险丝仍然生效
    stuck = risk_review_node(
        {"risk_review": {"severity": "high"}, "revision_round": 2, "round_at_last_review": 2}
    )
    assert stuck["risk_review"]["stuck"] is True
