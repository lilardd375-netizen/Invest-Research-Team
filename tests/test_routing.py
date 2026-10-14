"""条件路由 + 返工回环测试。"""

from src.graph.builder import build_graph, route_after_review


def test_high_routes_to_revise():
    app = build_graph()
    result = app.invoke(
        {
            "query": "宁德时代",
            "risk_review": {"severity": "high"},
            "revision_round": 0,
            "round_at_last_review": -1,
        }
    )
    # 走了 revise 分支：revision_round 从 0 被推到上限 2
    # （severity 全程为 high，路由规则决定了会连续返工两轮才停）
    assert result["revision_round"] == 2
    assert result["final_report"] == "stub report"


def test_revision_cap():
    app = build_graph()
    result = app.invoke(
        {
            "query": "宁德时代",
            "risk_review": {"severity": "high"},
            "revision_round": 2,
            "round_at_last_review": -1,
        }
    )
    assert result["revision_round"] == 2  # 未再返工
    assert result["final_report"] == "stub report"


def test_low_routes_directly():
    app = build_graph()
    result = app.invoke(
        {
            "query": "宁德时代",
            "risk_review": {"severity": "low"},
            "revision_round": 0,
            "round_at_last_review": -1,
        }
    )
    assert result["final_report"] == "stub report"
    assert result["revision_round"] == 0


def test_stuck_guard():
    """保险丝纯函数单测：轮次没推进（stuck）时即使 high 也直接成稿。"""
    state = {
        "risk_review": {"severity": "high", "stuck": True},
        "revision_round": 1,
        "round_at_last_review": 1,
    }
    assert route_after_review(state) == "finalize"


def test_not_stuck_still_revises():
    """对照：同样 high + round 1，但轮次有推进 -> 仍应返工。"""
    state = {
        "risk_review": {"severity": "high", "stuck": False},
        "revision_round": 1,
        "round_at_last_review": 0,
    }
    assert route_after_review(state) == "revise"
