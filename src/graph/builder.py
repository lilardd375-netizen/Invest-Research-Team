"""按 LangGraph API 组装图：
planner -> 四个分析师并行（fan-out）-> risk_review -> 条件路由 -> revise/finalize，
其中 revise 不回到风控，而是回到四个分析师（带质询重跑），形成
risk_review --revise--> revise -> 四分析师 -> risk_review 的返工环。
"""

from langgraph.graph import END, StateGraph

from src.graph.nodes import (
    _missing_analyst_stub_node,
    _missing_domain_analyst_stub_node,
    finalize_node,
    make_financial_analyst_node,
    make_industry_analyst_node,
    make_planner_node,
    make_risk_review_node,
    make_sentiment_analyst_node,
    make_technical_analyst_node,
    planner_node,
    revise_node,
    risk_review_node,
)
from src.graph.state import ResearchState

MAX_REVISION_ROUNDS = 2

# 并行分析师节点：planner 的 fan-out、risk_review 的 fan-in、revise 的回环都按这个集合连边。
ANALYST_NODES = (
    "financial_analyst",
    "industry_analyst",
    "sentiment_analyst",
    "technical_analyst",
)


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
    planner_llm=None,
    financial_llm=None,
    industry_llm=None,
    sentiment_llm=None,
    technical_llm=None,
    risk_llm=None,
    financial_provider=None,
    industry_provider=None,
    sentiment_provider=None,
    technical_provider=None,
    data_provider=None,
):
    """组装并 compile 返回。全部参数 None 默认，不传 = 全占位图（向后兼容）。

    - planner_llm 为 None 时 planner 用占位节点。
    - financial_llm 或财务数据源为 None 时财务分析用占位节点，返回空 analyses（旧行为不变）。
    - industry_llm / sentiment_llm / technical_llm 为 None（或对应数据源为 None）时，
      该节点整体 skip：不写 analyses、不调 LLM。
    - risk_llm 为 None 时风控用占位节点（原样回显 severity）；传入时风控由真实 LLM
      评审，其 severity 真实驱动 revise/finalize 路由。
    - data_provider 是 financial_provider 的旧参数名，仍然可用。
    """
    financial_provider = (
        financial_provider if financial_provider is not None else data_provider
    )

    planner_fn = planner_node if planner_llm is None else make_planner_node(planner_llm)
    if financial_llm is None or financial_provider is None:
        financial_fn = _missing_analyst_stub_node
    else:
        financial_fn = make_financial_analyst_node(financial_llm, financial_provider)

    def _domain_fn(llm, provider, factory):
        """三个新 Agent 的通用装配：缺 LLM 或缺数据源 -> 整节点 skip。"""
        if llm is None or provider is None:
            return _missing_domain_analyst_stub_node
        return factory(llm, provider)

    risk_fn = risk_review_node if risk_llm is None else make_risk_review_node(risk_llm)

    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_fn)
    graph.add_node("financial_analyst", financial_fn)
    graph.add_node(
        "industry_analyst", _domain_fn(industry_llm, industry_provider, make_industry_analyst_node)
    )
    graph.add_node(
        "sentiment_analyst",
        _domain_fn(sentiment_llm, sentiment_provider, make_sentiment_analyst_node),
    )
    graph.add_node(
        "technical_analyst",
        _domain_fn(technical_llm, technical_provider, make_technical_analyst_node),
    )
    graph.add_node("risk_review", risk_fn)
    graph.add_node("revise", revise_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("planner")
    for name in ANALYST_NODES:
        graph.add_edge("planner", name)  # fan-out：四个分析师并行
        graph.add_edge(name, "risk_review")  # fan-in：等四个都完成再评审
    graph.add_conditional_edges(
        "risk_review",  # 从哪个节点出发
        route_after_review,  # 路由函数：fn(state) -> 返回分支名
        {"revise": "revise", "finalize": "finalize"},  # 分支名->目标节点
    )
    for name in ANALYST_NODES:
        graph.add_edge("revise", name)  # 返工回环：带质询 fan-out 给四个分析师
    graph.add_edge("finalize", END)
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    print(app.invoke({"query": "宁德时代", "revision_round": 0}))
