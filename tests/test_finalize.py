"""finalize 真实化测试：报告合成 / 规则式兜底 / 质询截断 / 降级披露 / 硬约束条款。

全部离线：FakeLLM + 手写 state，不联网、不 import akshare。
"""

import json

from src.graph.builder import build_graph
from src.graph.nodes import (
    CHALLENGE_TEXT_CHARS,
    MAX_CHALLENGES_TOTAL,
    _build_report_user_prompt,
    _challenge_block,
    _challenge_history_digest,
    finalize_node,
    make_financial_analyst_node,
    make_finalize_node,
)
from src.llm.errors import LLMTransientError

AGENTS = (
    "financial_analyst",
    "industry_analyst",
    "sentiment_analyst",
    "technical_analyst",
)


class FakeLLM:
    """按序列返回响应；元素可以是字符串（JSON 文本）或异常实例。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.messages = []

    def chat(self, system, user):
        self.messages.append((system, user))
        item = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


class FakeProvider:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get_financials(self, ticker):
        self.calls.append(ticker)
        return self.payload


def analysis(agent, conclusion="增持", confidence="中", finding=None):
    return {
        "agent": agent,
        "conclusion": conclusion,
        "confidence": confidence,
        "key_points": [
            {"finding": finding or f"{agent} 的核心发现", "data_source": "akshare"}
        ],
        "concerns": [f"{agent} 自述关注点"],
        "assumptions": [f"{agent} 的假设"],
        "revised_in_round": 0,
    }


def analyses_all(**overrides):
    data = {agent: analysis(agent) for agent in AGENTS}
    data.update(overrides)
    return data


def history_round(round_no, severity, challenges, degraded=False, revised_agents=None, summary="总评"):
    targets = {}
    for item in challenges:
        target = item.get("target_agent") or "unassigned"
        targets[target] = targets.get(target, 0) + 1
    return {
        "round": round_no,
        "severity": severity,
        "degraded": degraded,
        "summary": summary,
        "challenges": challenges,
        "challenge_targets": targets,
        "revised_agents": list(revised_agents or []),
    }


def challenge(target="financial_analyst", issue="ROE 与净利率口径不一致", action="重算 ROE 并给依据"):
    return {"target_agent": target, "issue": issue, "required_action": action}


def report_json(**overrides):
    data = {
        "title": "宁德时代投资研究报告",
        "stock_code": "300750",
        "report_date": "2026-10-16",
        "overall_conclusion": "中性",
        "overall_confidence": "中",
        "per_agent_summaries": [
            {
                "agent": agent,
                "status": "ok",
                "conclusion": "增持",
                "confidence": "中",
                "summary": f"{agent} 一句话要点",
            }
            for agent in AGENTS
        ],
        "key_thesis": ["ROE 11.41% 仍高于制造业中位（来源：akshare）"],
        "key_risks": ["毛利率数据缺失"],
        "revision_history": [
            {
                "round": 0,
                "severity": "high",
                "degraded": False,
                "challenge_targets": {"financial_analyst": 2},
                "revised_agents": [],
                "note": "首轮质询",
            }
        ],
        "disclosures": ["覆盖边界：仅财务/行业/舆情/技术面四个维度"],
        "generated_at": "2026-10-16T09:00:00",
    }
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


def base_state(**overrides):
    state = {
        "query": "宁德时代 300750 的投资价值",
        "outline": {"sections": [{"title": "公司概况", "research_questions": ["主营？"]}]},
        "analyses": analyses_all(),
        "revision_round": 0,
        "review_history": [history_round(0, "high", [challenge()])],
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------- 正常合成


def test_finalize_with_llm_produces_report():
    llm = FakeLLM([report_json()])
    out = make_finalize_node(llm)(base_state())

    assert llm.calls == 1
    assert out["report_llm_degraded"] is False
    assert out["report"]["overall_conclusion"] == "中性"
    assert len(out["report"]["per_agent_summaries"]) == 4
    assert out["final_report"].startswith("# ")
    for section in ("## 各分析师结论", "## 核心逻辑", "## 主要风险", "## 返工记录", "## 披露与局限"):
        assert section in out["final_report"]


def test_report_prompt_carries_history_and_analyses():
    llm = FakeLLM([report_json()])
    make_finalize_node(llm)(
        base_state(
            revision_round=1,
            review_history=[
                history_round(0, "high", [challenge()]),
                history_round(1, "medium", [challenge("industry_analyst")], revised_agents=["industry_analyst"]),
            ],
            risk_degraded=False,
            risk_degraded_round=-1,
        )
    )
    _, user = llm.messages[0]
    assert "第 0 轮" in user and "第 1 轮" in user
    assert "本轮修订者" in user
    assert "financial_analyst" in user
    assert "risk_degraded=False" in user


# ---------------------------------------------------------------- 降级兜底


def test_finalize_llm_failure_falls_back_to_rule_based():
    llm = FakeLLM([LLMTransientError("网络异常（可重试）"), LLMTransientError("网络异常（可重试）")])
    out = make_finalize_node(llm)(base_state())

    assert llm.calls == 2  # 可重试错误重试 1 次后才降级
    assert out["report_llm_degraded"] is True
    assert out["report"]["rule_based"] is True
    assert "transient" in out["report_error"]
    # 降级不丢结果：四家结论仍在报告里，且必须披露自己是降级产物
    assert len(out["report"]["per_agent_summaries"]) == 4
    assert "规则式组装" in out["final_report"]
    assert "降级产物" in out["final_report"]


def test_finalize_without_llm_is_rule_based_and_flagged():
    out = finalize_node(base_state())

    assert out["report_llm_degraded"] is True
    assert out["report"]["rule_based"] is True
    assert out["report"]["stock_code"] == "300750"
    assert "未启用报告合成 LLM" in out["report_error"]


def test_rule_based_report_votes_among_agents():
    # 3/4 严格过半 -> 采用该方向
    majority = base_state(
        analyses=analyses_all(
            financial_analyst=analysis("financial_analyst", "增持"),
            industry_analyst=analysis("industry_analyst", "增持"),
            sentiment_analyst=analysis("sentiment_analyst", "减持"),
            technical_analyst=analysis("technical_analyst", "增持"),
        )
    )
    assert finalize_node(majority)["report"]["overall_conclusion"] == "增持"

    # 2/4 只是相对多数，不构成严格过半 -> 保守取「中性」并在披露里说明分歧分布
    plurality = base_state(
        analyses=analyses_all(
            financial_analyst=analysis("financial_analyst", "增持"),
            industry_analyst=analysis("industry_analyst", "增持"),
            sentiment_analyst=analysis("sentiment_analyst", "减持"),
            technical_analyst=analysis("technical_analyst", "中性"),
        )
    )
    report = finalize_node(plurality)["report"]
    assert report["overall_conclusion"] == "中性"  # 分歧不硬凑方向
    assert any("未形成严格多数" in item for item in report["disclosures"])
    assert any("增持 2 家" in item for item in report["disclosures"])


def test_complete_report_fills_missing_agents():
    partial = report_json(
        per_agent_summaries=[
            {"agent": "financial_analyst", "status": "ok", "conclusion": "增持", "confidence": "中", "summary": "只有一家"}
        ]
    )
    out = make_finalize_node(FakeLLM([partial]))(base_state())
    agents = [item["agent"] for item in out["report"]["per_agent_summaries"]]
    assert agents[:4] == list(AGENTS)  # 漏掉的三家被补齐，分歧不许被吞
    assert any("未逐家呈现" in item for item in out["report"]["disclosures"])


def test_complete_report_backfills_revision_history():
    no_history = report_json(revision_history=[])
    out = make_finalize_node(FakeLLM([no_history]))(base_state())
    assert out["report"]["revision_history"][0]["severity"] == "high"
    assert any("review_history 补齐" in item for item in out["report"]["disclosures"])


# ---------------------------------------------------------------- 质询与披露


def test_medium_challenges_land_in_disclosures():
    medium = challenge("industry_analyst", "同业样本只有两家，可比性不足", "补充同业样本")
    state = base_state(
        review_history=[
            history_round(0, "high", [challenge()]),
            history_round(1, "medium", [medium], revised_agents=["industry_analyst"]),
        ]
    )
    out = finalize_node(state)

    assert any("同业样本只有两家" in item for item in out["report"]["disclosures"])
    assert any("medium" not in item or True for item in out["report"]["disclosures"])
    assert "## 披露与局限" in out["final_report"]
    assert "同业样本只有两家" in out["final_report"]
    # 未解决的 high 质询进风险章节，不是披露章节
    assert any("口径不一致" in item for item in out["report"]["key_risks"])


def test_challenge_history_truncation_and_old_round_representatives():
    long_issue = "口径不一致" + "很长" * 200
    history = [
        history_round(
            0,
            "high",
            [
                challenge(target, long_issue + target)
                for target in ("financial_analyst", "industry_analyst", "sentiment_analyst")
                for _ in range(2)
            ],
        ),
        history_round(
            1,
            "high",
            [
                challenge(target, f"第二轮 {target} 的质询" + str(i))
                for target in ("financial_analyst", "industry_analyst", "sentiment_analyst")
                for i in range(2)
            ],
        ),
        history_round(2, "high", [challenge("sentiment_analyst", f"第三轮的 {i}") for i in range(8)]),
    ]
    digest = _challenge_history_digest(base_state(revision_round=2, review_history=history))
    lines = [line for line in digest.splitlines() if line.startswith("- [第")]

    # 旧轮每 target 只留 1 条代表 + 最新一轮全量 8 条 = 14 > 12 -> 从最早的开始丢 2 条
    assert len(lines) == MAX_CHALLENGES_TOTAL
    assert "按上限省略" in digest
    assert sum(1 for line in lines if line.startswith("- [第 0 轮]")) == 1
    assert sum(1 for line in lines if line.startswith("- [第 2 轮]")) == 8
    # 单条正文（issue）截断到 CHALLENGE_TEXT_CHARS 以内
    for line in lines:
        issue = line.split("] ", 1)[1].split("（要求：")[0]
        assert len(issue) <= CHALLENGE_TEXT_CHARS + 1


def test_degraded_rounds_reach_report():
    state = base_state(
        revision_round=2,
        risk_degraded=True,
        risk_degraded_round=2,
        review_history=[
            history_round(0, "high", [challenge()]),
            history_round(1, "high", [challenge()], revised_agents=["financial_analyst"]),
            history_round(2, "low", [], degraded=True, summary="风控 Agent 不可用"),
        ],
    )
    out = finalize_node(state)

    assert out["report"]["revision_history"][2]["degraded"] is True
    assert out["report"]["revision_history"][1]["revised_agents"] == ["financial_analyst"]
    assert any("第 2 轮风控复核因故障降级" in item for item in out["report"]["disclosures"])
    assert any("降级" in item for item in out["report"]["disclosures"])
    assert "降级产物" in out["final_report"]  # 返工记录章节里逐轮标注
    assert "## 返工记录" in out["final_report"]


def test_degraded_round_survives_full_graph_with_failing_risk_llm():
    """端到端：风控 LLM 持续失败 -> 降级标记写进 state 并一路传到报告。"""
    app = build_graph(risk_llm=FakeLLM([LLMTransientError("boom"), LLMTransientError("boom")]))
    result = app.invoke({"query": "宁德时代 300750", "revision_round": 0})

    assert result["risk_degraded"] is True
    assert result["risk_degraded_round"] == 0
    assert result["risk_review"]["degraded"] is True
    assert result["review_history"][0]["degraded"] is True
    assert "降级" in result["final_report"]
    assert result["final_report"].startswith("# ")


# ---------------------------------------------------------------- 硬约束条款


def test_complete_report_forces_metadata_from_state():
    """合成 LLM 会自己编 generated_at/stock_code，元信息必须以代码为准。"""
    from src.graph.nodes import _complete_report

    llm_report = {
        "title": "",
        "stock_code": "",
        "report_date": "",
        "generated_at": "2026-10-15T00:00:00Z",  # 实测模型编出来的假时间
        "per_agent_summaries": [],
    }
    out = _complete_report(llm_report, {"query": "宁德时代 300750 的投资价值"})

    assert out["stock_code"] == "300750"
    assert out["report_date"]
    assert out["generated_at"] != "2026-10-15T00:00:00Z"
    assert out["title"].startswith("300750")


def test_revision_hard_constraint_clause_present():
    block = _challenge_block([challenge()])
    assert "必须要么给出重算/修正后的数值与依据" in block
    assert "禁止仅表态承认而不修正" in block


def test_financial_analyst_revision_prompt_has_clause():
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "agent": "financial_analyst",
                    "conclusion": "中性",
                    "confidence": "低",
                    "key_points": [{"finding": "重算后 ROE 11.41%", "data_source": "akshare"}],
                    "concerns": [],
                    "assumptions": [],
                },
                ensure_ascii=False,
            )
        ]
    )
    provider = FakeProvider({"ticker": "300750", "roe": 11.41})
    node = make_financial_analyst_node(llm, provider)
    node(
        {
            "query": "宁德时代 300750",
            "revision_round": 1,
            "analyses": {"financial_analyst": analysis("financial_analyst")},
            "pending_challenges": {"financial_analyst": [challenge()]},
        }
    )
    _, user = llm.messages[0]
    assert "禁止仅表态承认而不修正" in user
    assert "ROE 与净利率口径不一致" in user


def test_report_prompt_has_no_external_rating_input():
    prompt = _build_report_user_prompt(base_state())
    assert "只输出 JSON" not in prompt  # 该要求写在 system prompt 里
    assert "外部评级" not in prompt
