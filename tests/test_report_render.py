"""Markdown 导出的离线测试：章节完整性 / 降级标注 / 空质询仍披露 / 落盘。

render_report 是纯函数（state 进、字符串出），不依赖 LLM 与网络。
"""

from src.report.render import report_path, render_report, write_report

AGENT_LABELS = {
    "financial_analyst": "财务",
    "industry_analyst": "行业",
    "sentiment_analyst": "舆情",
    "technical_analyst": "技术面",
}


def analysis(agent, conclusion="增持", confidence="中", assumptions=None):
    return {
        "agent": agent,
        "conclusion": conclusion,
        "confidence": confidence,
        "key_points": [{"finding": f"{agent} 的发现", "data_source": "akshare"}],
        "concerns": [f"{agent} 自述关注点"],
        "assumptions": assumptions if assumptions is not None else [f"{agent} 的假设"],
    }


def analyses_all():
    return {
        "financial_analyst": analysis("financial_analyst", "增持"),
        "industry_analyst": analysis("industry_analyst", "中性"),
        "sentiment_analyst": analysis("sentiment_analyst", "减持"),
        "technical_analyst": analysis("technical_analyst", "增持", "低", assumptions=[]),
    }


def state_without_report(**overrides):
    state = {
        "query": "宁德时代 300750 的投资价值",
        "analyses": analyses_all(),
        "revision_round": 0,
        "review_history": [
            {
                "round": 0,
                "severity": "medium",
                "degraded": False,
                "summary": "数据基本自洽，覆盖不足属已知边界",
                "challenges": [
                    {
                        "target_agent": "industry_analyst",
                        "issue": "同业样本只有两家，可比性不足",
                        "required_action": "补充同业样本",
                    }
                ],
                "challenge_targets": {"industry_analyst": 1},
                "revised_agents": [],
            }
        ],
    }
    state.update(overrides)
    return state


def test_render_contains_all_sections():
    text = render_report(state_without_report())

    for section in ("## 综合结论", "## 各分析师结论", "## 核心逻辑", "## 主要风险", "## 返工记录", "## 披露与局限"):
        assert section in text
    assert text.startswith("# ") and "宁德时代 300750" in text.splitlines()[0] or "300750" in text
    for agent, label in AGENT_LABELS.items():
        assert f"### {label}（{agent}）" in text


def test_render_keeps_agent_divergence():
    text = render_report(state_without_report())

    # 四家结论分歧必须逐家呈现，不许只留综合结论
    assert "增持" in text and "中性" in text and "减持" in text
    assert text.count("### ") == 4


def test_render_empty_challenges_still_discloses_sources_and_assumptions():
    text = render_report(state_without_report(review_history=[]))

    assert "## 披露与局限" in text
    assert "覆盖边界" in text  # 数据源/维度边界
    assert "financial_analyst 的假设" in text  # 各 agent 的假设
    assert "（没有风控复核记录）" in text or "没有风控复核记录" in text
    assert "## 返工记录" in text


def test_render_degrades_banner_and_round_flag():
    text = render_report(
        state_without_report(
            report_llm_degraded=True,
            report_error="transient: 重试 2 次后仍失败",
            review_history=[
                {
                    "round": 0,
                    "severity": "high",
                    "degraded": False,
                    "summary": "首轮",
                    "challenges": [
                        {"target_agent": "financial_analyst", "issue": "ROE 口径不一致", "required_action": "重算"}
                    ],
                    "challenge_targets": {"financial_analyst": 1},
                    "revised_agents": ["financial_analyst"],
                },
                {
                    "round": 1,
                    "severity": "low",
                    "degraded": True,
                    "summary": "风控 Agent 不可用",
                    "challenges": [],
                    "challenge_targets": {},
                    "revised_agents": [],
                },
            ],
        )
    )

    assert "降级产物" in text
    assert "transient: 重试 2 次后仍失败" in text
    assert "（降级产物，非真实评审）" in text
    assert "规则式组装" in text
    assert "第 0 轮" in text and "第 1 轮" in text
    assert "修订者：financial_analyst" in text


def test_render_rule_based_report_from_bare_state():
    text = render_report(state_without_report())

    assert "同业样本只有两家" in text
    # 2 增持 / 1 中性 / 1 减持：无严格多数 -> 综合结论保守取中性并披露分歧
    assert "**中性**" in text
    assert "未形成严格多数" in text
    assert "覆盖边界" in text


def test_report_path_uses_stock_code_and_date():
    path = report_path(state_without_report())
    assert path.name.startswith("300750_")
    assert path.name.endswith(".md")
    assert path.parent.name == "reports"


def test_write_report_creates_file(tmp_path):
    state = state_without_report()
    path = write_report(state, tmp_path)

    assert path.exists()
    assert path.read_text(encoding="utf-8") == render_report(state)
    assert path.name.endswith(".md")


def test_write_report_unknown_ticker_falls_back():
    path = report_path({"query": "没有代码的查询"})
    assert path.name.startswith("unknown_")
