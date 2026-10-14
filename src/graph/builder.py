"""按 LangGraph API 组装图：
planner -> financial_analyst -> risk_review -> 条件路由 -> revise/finalize。
"""

from langgraph.graph import END, StateGraph

from src.graph.nodes import (
    _missing_analyst_stub_node,
    finalize_node,
    make_financial_analyst_node,
    make_planner_node,
    make_risk_review_node,
    planner_node,
    revise_node,
    risk_review_node,
)
from src.graph.state import ResearchState

MAX_REVISION_ROUNDS = 2


def route_after_review(state: dict) -> str:
    """条件路由：只返回分支名字符串。

    severity 为 high、返工次数未达上限、且上一轮确实推进了轮次
    （not stuck）-> 走返工；否则直接成稿。
    """
    review = state.get("risk_review") or {}
    severity = review.get("severity", "low")
    stuck = review.get("stuck", False)
    revision_round = state.get("revision_round", 0)
    if severity == "high" and revision_round < MAX_REVISION_ROUNDS and not stuck:
        return "revise"
    return "finalize"


def build_graph(
    planner_llm=None, financial_llm=None, risk_llm=None, data_provider=None
):
    """组装并 compile 返回。

    - planner_llm 为 None 时 planner 用占位节点（向后兼容）。
    - financial_llm 为 None 时财务分析用占位节点，返回空 analyses。
    - data_provider 为 None 时财务分析同样走占位节点。
    - risk_llm 为 None 时风控用占位节点（原样回显 severity，向后兼容）；
      传入时风控由真实 LLM 评审，其 severity 真实驱动 revise/finalize 路由。
    """
    planner_fn = planner_node if planner_llm is None else make_planner_node(planner_llm)
    if financial_llm is None or data_provider is None:
        financial_fn = _missing_analyst_stub_node
    else:
        financial_fn = make_financial_analyst_node(financial_llm, data_provider)
    risk_fn = risk_review_node if risk_llm is None else make_risk_review_node(risk_llm)

    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_fn)
    graph.add_node("financial_analyst", financial_fn)
    graph.add_node("risk_review", risk_fn)
    graph.add_node("revise", revise_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "financial_analyst")
    graph.add_edge("financial_analyst", "risk_review")
    graph.add_conditional_edges(
        "risk_review",  # 从哪个节点出发
        route_after_review,  # 路由函数：fn(state) -> 返回分支名
        {"revise": "revise", "finalize": "finalize"},  # 分支名->目标节点
    )
    graph.add_edge("revise", "risk_review")  # 回环
    graph.add_edge("finalize", END)
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    print(app.invoke({"query": "宁德时代", "revision_round": 0}))
