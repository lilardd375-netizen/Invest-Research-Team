"""冒烟测试：图能跑通 + 状态 schema 字段完整。"""

from src.graph.builder import build_graph
from src.graph.state import ResearchState

EXPECTED_FIELDS = [
    "query",
    "outline",
    "analyses",
    "risk_review",
    "revision_round",
    "final_report",
]


def test_graph_runs():
    app = build_graph()
    result = app.invoke({"query": "宁德时代"})
    assert result["final_report"] == "stub report"
    assert result["query"] == "宁德时代"
    # 新图里 planner 之后接 risk_review：outline 仍是 planner 写的 "stub"
    assert result["outline"]["status"] == "stub"
    # 未预设 severity 时默认 low，直接成稿，不返工
    assert result["risk_review"]["severity"] == "low"
    assert result["risk_review"]["stuck"] is False
    assert result["round_at_last_review"] == 0
    # 未注入 financial_llm/data_provider 时财务分析走占位节点
    assert result["analyses"] == {}


def test_state_schema():
    annotations = ResearchState.__annotations__
    assert annotations
    for field in EXPECTED_FIELDS:
        assert field in annotations, f"缺少字段: {field}"
