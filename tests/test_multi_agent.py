"""第七步：行业 / 舆情 / 技术面三个分析 Agent 的测试（全 Fake，不联网、不 import akshare）；
第十步起扩充为四个域分析师（新增估值 Agent）。

覆盖三层：
1. 每个域 Agent 各 5 条节点级契约（valid / transient_retry / fatal_no_retry /
   provider_failure / consumes_challenges）；
2. 幂等跳过守卫（已有结果且无待办 -> 不调 LLM）；
3. 整图：五分析师并行 fan-out、定向返工只重跑被质询方、
   「high -> 修订 -> 复审降级」的构造式闭环（复刻第六步冒烟口径）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from src.graph.builder import build_graph
from src.graph.nodes import (
    make_industry_analyst_node,
    make_sentiment_analyst_node,
    make_technical_analyst_node,
    make_valuation_analyst_node,
)
from src.llm.errors import LLMFatalError, LLMTransientError
from src.tools.financial_data import FakeProvider as FakeFinancialProvider
from src.tools.industry_data import FakeIndustryProvider
from src.tools.sentiment_data import FakeSentimentProvider
from src.tools.technical_data import FakeTechnicalProvider
from src.tools.valuation_data import FakeValuationProvider

TICKER = "300750"
QUERY = "宁德时代 300750 的投资价值"


class FakeLLM:
    """按顺序返回预设响应；元素可以是字符串，也可以是异常实例。

    只有一个元素时该响应会被重复使用（方便构造「一直失败」的场景）。
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.messages: list[tuple[str, str]] = []

    def chat(self, system: str, user: str) -> str:
        self.calls += 1
        self.messages.append((system, user))
        if not self.responses:
            raise AssertionError("FakeLLM 的预设响应已用尽")
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, BaseException):
            raise item
        return item


class SeededFirstResponseLLM:
    """首次 chat 返回预置的「有缺陷」分析，之后委托内层 FakeLLM 给出修订版。

    复刻第六步构造式冒烟：不依赖真实模型的判断力，用种子数据确保首轮必然有缺陷。
    """

    def __init__(self, seeded: str, inner: FakeLLM):
        self.seeded = seeded
        self.inner = inner
        self.calls = 0
        self.messages: list[tuple[str, str]] = []

    def chat(self, system: str, user: str) -> str:
        self.calls += 1
        self.messages.append((system, user))
        if self.calls == 1:
            return self.seeded
        return self.inner.chat(system, user)


def analysis_json(agent: str, *, conclusion="增持", confidence="高", finding="板块景气温和回升", source="fake-data") -> str:
    return json.dumps(
        {
            "agent": agent,
            "conclusion": conclusion,
            "confidence": confidence,
            "key_points": [{"finding": finding, "data_source": source}],
            "concerns": ["缺少中报细项"],
            "assumptions": ["假设行业景气延续"],
        },
        ensure_ascii=False,
    )


FAKE_INDUSTRY = {
    "ticker": TICKER,
    "source": "fake:industry",
    "as_of": "2026-10-14",
    "industry": "电池",
    "board": {"name": "电池", "change_pct": 1.23, "turnover_rate": 2.1},
    "peers": [
        {"code": "002594", "name": "比亚迪", "change_pct": -0.5, "pe_ratio": 22.1},
        {"code": "002074", "name": "国轩高科", "change_pct": 0.8, "pe_ratio": 35.6},
    ],
    "missing_fields": [],
}

FAKE_SENTIMENT = {
    "ticker": TICKER,
    "source": "fake:sentiment",
    "as_of": "2026-10-14",
    "count": 2,
    "news": [
        {"title": "宁德时代发布新一代神行电池", "summary": "公司称能量密度提升", "published_at": "2026-10-13 09:30:00"},
        {"title": "机构上调电池板块评级", "summary": "多家券商发布研报", "published_at": "2026-10-12 15:10:00"},
    ],
    "missing_fields": [],
}

FAKE_TECHNICAL = {
    "ticker": TICKER,
    "source": "fake:technical",
    "as_of": "2026-10-14",
    "close": 210.5,
    "ma5": 205.1,
    "ma20": 198.3,
    "ma60": 190.7,
    "change_20d_pct": 6.4,
    "high_20d": 215.0,
    "low_20d": 188.2,
    "volume_ratio_5_60": 1.32,
    "missing_fields": [],
}

FAKE_FINANCIALS = {
    "ticker": TICKER,
    "source": "fake:financial",
    "report_period": "2026-06-30",
    "as_of": "2026-06",
    "roe": 11.41,
    "net_margin": 16.98,
    "debt_ratio": 63.65,
    "revenue_growth": 54.8,
    "missing_metrics": ["gross_margin"],
}

FAKE_VALUATION = {
    "ticker": TICKER,
    "source": "fake:valuation",
    "as_of": "2026-10-15",
    "window": {
        "label": "近三年",
        "start": "2023-10-15",
        "end": "2026-10-15",
        "trading_days": 1097,
    },
    "metrics": {
        "pe_ttm": {"current": 15.16, "min": 13.98, "max": 32.04, "median": 21.81, "percentile": 2.19, "samples": 1097},
        "pb": {"current": 3.4, "min": 3.4, "max": 6.71, "median": 4.64, "percentile": 0.09, "samples": 1097},
    },
    "industry": {"name": "电气机械和器材制造业", "code": "C38", "pe_static_weighted": 20.39},
    "missing_fields": [],
}


@dataclass(frozen=True)
class AgentCase:
    name: str
    make_node: object
    provider_cls: object
    fetch_method: str
    data: dict
    marker: str  # 必须出现在 user prompt 里的数据标记


AGENT_CASES = [
    AgentCase("industry_analyst", make_industry_analyst_node, FakeIndustryProvider, "get_industry", FAKE_INDUSTRY, "电池"),
    AgentCase(
        "sentiment_analyst",
        make_sentiment_analyst_node,
        FakeSentimentProvider,
        "get_sentiment",
        FAKE_SENTIMENT,
        "宁德时代发布新一代神行电池",
    ),
    AgentCase(
        "technical_analyst",
        make_technical_analyst_node,
        FakeTechnicalProvider,
        "get_technical",
        FAKE_TECHNICAL,
        "volume_ratio_5_60",
    ),
    AgentCase(
        "valuation_analyst",
        make_valuation_analyst_node,
        FakeValuationProvider,
        "get_valuation",
        FAKE_VALUATION,
        "pe_static_weighted",
    ),
]

CASE_IDS = [case.name for case in AGENT_CASES]


def base_state(**overrides) -> dict:
    state = {"query": QUERY, "revision_round": 0}
    state.update(overrides)
    return state


def make_broken_provider(method: str):
    """构造一个在指定 fetch 方法上必然抛错的数据源。"""

    class BrokenProvider:
        def __init__(self):
            self.calls = 0

    def _boom(self, ticker):
        self.calls += 1
        raise RuntimeError("akshare 接口连接失败")

    setattr(BrokenProvider, method, _boom)
    return BrokenProvider()


# ---------------------------------------------------------------- 节点级契约


@pytest.mark.parametrize("case", AGENT_CASES, ids=CASE_IDS)
def test_agent_valid(case):
    llm = FakeLLM([analysis_json(case.name)])
    provider = case.provider_cls({TICKER: case.data})

    out = case.make_node(llm, provider)(base_state())

    analysis = out["analyses"][case.name]
    assert analysis["agent"] == case.name
    assert analysis["conclusion"] == "增持"
    assert analysis["confidence"] == "高"
    assert analysis["revised_in_round"] == 0
    assert llm.calls == 1
    assert provider.calls == [TICKER]
    assert case.marker in llm.messages[0][1]
    assert "analyst_errors" not in out
    assert "pending_challenges" not in out  # 没有待办就不写这一格


@pytest.mark.parametrize("case", AGENT_CASES, ids=CASE_IDS)
def test_agent_transient_retry(case):
    llm = FakeLLM([LLMTransientError("模型返回非法 JSON（可重试）"), analysis_json(case.name)])
    provider = case.provider_cls({TICKER: case.data})

    out = case.make_node(llm, provider)(base_state())

    assert llm.calls == 2
    assert out["analyses"][case.name]["conclusion"] == "增持"
    assert "analyst_errors" not in out


@pytest.mark.parametrize("case", AGENT_CASES, ids=CASE_IDS)
def test_agent_fatal_no_retry(case):
    llm = FakeLLM([LLMFatalError("鉴权失败：API key 无效（不可重试）")])
    provider = case.provider_cls({TICKER: case.data})

    out = case.make_node(llm, provider)(base_state())

    assert llm.calls == 1, "不可重试错误不应重试"
    analysis = out["analyses"][case.name]
    assert analysis["status"] == "unavailable"
    assert analysis["error_kind"] == "fatal"
    assert "鉴权失败" in analysis["error"]
    assert "鉴权失败" in out["analyst_errors"][case.name]


@pytest.mark.parametrize("case", AGENT_CASES, ids=CASE_IDS)
def test_agent_provider_failure(case):
    llm = FakeLLM([analysis_json(case.name)])
    provider = make_broken_provider(case.fetch_method)

    out = case.make_node(llm, provider)(base_state())

    assert llm.calls == 0, "数据源挂了就不许调 LLM 编造"
    assert provider.calls == 1
    analysis = out["analyses"][case.name]
    assert analysis["status"] == "unavailable"
    assert analysis["error_kind"] == "data_source"
    assert "akshare 接口连接失败" in analysis["error"]
    assert set(analysis) == {"status", "error", "error_kind"}


@pytest.mark.parametrize("case", AGENT_CASES, ids=CASE_IDS)
def test_agent_consumes_challenges(case):
    llm = FakeLLM([analysis_json(case.name)])
    provider = case.provider_cls({TICKER: case.data})
    challenge = {
        "target_agent": case.name,
        "issue": f"{case.name} 的结论缺证据",
        "required_action": "补齐关键数据来源",
    }
    other = {"target_agent": "financial_analyst", "issue": "别人的待办", "required_action": "别动我"}
    state = base_state(
        revision_round=1,
        pending_challenges={case.name: [challenge], "financial_analyst": [other]},
    )

    out = case.make_node(llm, provider)(state)

    prompt = llm.messages[0][1]
    assert "必须逐条正面回应" in prompt
    assert f"{case.name} 的结论缺证据" in prompt
    assert "补齐关键数据来源" in prompt
    # 修订硬约束条款（第十步任务 0 核查补断言）：域分析师路径此前只有 financial 有断言
    assert "禁止仅表态承认而不修正" in prompt
    assert out["analyses"][case.name]["revised_in_round"] == 1
    # 只清自己那一格，别人的待办原样保留
    assert out["pending_challenges"] == {case.name: []}


@pytest.mark.parametrize("case", AGENT_CASES, ids=CASE_IDS)
def test_skip_when_no_pending(case):
    llm = FakeLLM([analysis_json(case.name)])
    provider = case.provider_cls({TICKER: case.data})
    state = base_state(revision_round=1, analyses={case.name: {"conclusion": "增持"}})

    out = case.make_node(llm, provider)(state)

    assert out == {}, "已有结果且无待办时应整节点跳过"
    assert llm.calls == 0
    assert provider.calls == []


# ---------------------------------------------------------------- 整图行为


def make_providers():
    return {
        "financial_analyst": FakeFinancialProvider({TICKER: FAKE_FINANCIALS}),
        "industry_analyst": FakeIndustryProvider({TICKER: FAKE_INDUSTRY}),
        "sentiment_analyst": FakeSentimentProvider({TICKER: FAKE_SENTIMENT}),
        "technical_analyst": FakeTechnicalProvider({TICKER: FAKE_TECHNICAL}),
        "valuation_analyst": FakeValuationProvider({TICKER: FAKE_VALUATION}),
    }


def make_llms(responses=None):
    responses = responses or {}
    return {name: FakeLLM(responses.get(name, [analysis_json(name)])) for name in make_providers()}


RISK_LOW = json.dumps({"severity": "low", "challenges": [], "summary": "常规提示，无重大问题"}, ensure_ascii=False)

RISK_HIGH_INDUSTRY = json.dumps(
    {
        "severity": "high",
        "challenges": [
            {
                "target_agent": "industry_analyst",
                "issue": "同业对比只给了两家且口径不明，不足以支撑板块判断",
                "required_action": "补齐同业指标口径并说明数据来源",
            }
        ],
        "summary": "行业结论证据不足，需返工",
    },
    ensure_ascii=False,
)

RISK_MEDIUM = json.dumps(
    {
        "severity": "medium",
        "challenges": [
            {
                "target_agent": "industry_analyst",
                "issue": "同业数据仍偏旧，但不影响方向性结论",
                "required_action": "在报告中披露数据时效",
            }
        ],
        "summary": "上一轮质询已正面回应，剩余为可接受假设",
    },
    ensure_ascii=False,
)


def build_multi_graph(llms, providers, risk_llm):
    return build_graph(
        financial_llm=llms["financial_analyst"],
        industry_llm=llms["industry_analyst"],
        sentiment_llm=llms["sentiment_analyst"],
        technical_llm=llms["technical_analyst"],
        valuation_llm=llms["valuation_analyst"],
        risk_llm=risk_llm,
        financial_provider=providers["financial_analyst"],
        industry_provider=providers["industry_analyst"],
        sentiment_provider=providers["sentiment_analyst"],
        technical_provider=providers["technical_analyst"],
        valuation_provider=providers["valuation_analyst"],
    )


def test_parallel_all_run():
    llms = make_llms()
    providers = make_providers()
    risk_llm = FakeLLM([RISK_LOW])
    app = build_multi_graph(llms, providers, risk_llm)

    result = app.invoke(base_state())

    assert set(result["analyses"]) == {
        "financial_analyst",
        "industry_analyst",
        "sentiment_analyst",
        "technical_analyst",
        "valuation_analyst",
    }
    for name, llm in llms.items():
        assert llm.calls == 1, f"{name} 应被调用 1 次"
    for name, provider in providers.items():
        assert provider.calls == [TICKER], f"{name} 数据源应被取数 1 次"
    assert risk_llm.calls == 1
    assert result["revision_round"] == 0
    assert result["risk_review"]["severity"] == "low"
    # 第八步：finalize 不再是 stub，改为规则式组装并渲染 Markdown 报告
    assert result["final_report"].startswith("# ")


def test_targeted_revision():
    llms = make_llms(
        responses={
            "industry_analyst": [
                analysis_json("industry_analyst", finding="板块景气温和回升", source="板块行情"),
                analysis_json("industry_analyst", conclusion="中性", confidence="中", finding="补齐同业口径后结论改为中性"),
            ]
        }
    )
    providers = make_providers()
    risk_llm = FakeLLM([RISK_HIGH_INDUSTRY, RISK_LOW])
    app = build_multi_graph(llms, providers, risk_llm)

    result = app.invoke(base_state())

    # 只有被点名的行业 Agent 重跑；其余四个靠幂等守卫空转跳过
    assert llms["industry_analyst"].calls == 2
    assert llms["sentiment_analyst"].calls == 1
    assert llms["technical_analyst"].calls == 1
    assert llms["valuation_analyst"].calls == 1
    # 收尾时财务 Agent 也补上了幂等跳过，所以这轮同样空转（此前预期是 2，属过时预期）
    assert llms["financial_analyst"].calls == 1

    assert risk_llm.calls == 2
    assert result["revision_round"] == 1
    assert result["analyses"]["industry_analyst"]["conclusion"] == "中性"
    assert result["analyses"]["industry_analyst"]["revised_in_round"] == 1

    # 质询进了行业 Agent 的第二轮 prompt，没有倒给别人
    assert "补齐同业指标口径并说明数据来源" in llms["industry_analyst"].messages[1][1]
    assert "同业对比只给了两家" not in llms["sentiment_analyst"].messages[0][1]
    assert "同业对比只给了两家" not in llms["technical_analyst"].messages[0][1]
    # 第二轮风控 prompt 带上了上一轮结论与本轮返工标记
    assert "上一轮风控结论：high" in risk_llm.messages[1][1]
    assert "revised_in_round=1" in risk_llm.messages[1][1]
    # 行业 Agent 的待办已消费
    assert result["pending_challenges"]["industry_analyst"] == []


FLAWED_INDUSTRY = json.dumps(
    {
        "agent": "industry_analyst",
        "conclusion": "增持",
        "confidence": "高",
        "key_points": [
            {
                "finding": "板块近 20 日涨幅 12.4%，同时板块近 20 日跌幅 5.1%，趋势明确向上",
                "data_source": "板块行情",
            },
            {"finding": "两家同业估值均低于自身，具备相对优势", "data_source": "同业指标"},
        ],
        "concerns": [],
        "assumptions": ["行业景气延续"],
    },
    ensure_ascii=False,
)

FIXED_INDUSTRY = analysis_json(
    "industry_analyst",
    conclusion="中性",
    confidence="低",
    finding="撤回自相矛盾的涨幅表述，改为按单一数据源披露板块近 20 日涨幅 7.9%",
)


def test_severity_drops_multi():
    """构造式闭环：首轮行业分析自相矛盾 -> 风控 high -> 定向返工 -> 复审降级 -> finalize。"""
    industry_llm = SeededFirstResponseLLM(FLAWED_INDUSTRY, FakeLLM([FIXED_INDUSTRY]))
    llms = make_llms()
    llms["industry_analyst"] = industry_llm
    providers = make_providers()
    risk_llm = FakeLLM([RISK_HIGH_INDUSTRY, RISK_MEDIUM])
    app = build_multi_graph(llms, providers, risk_llm)

    result = app.invoke(base_state())

    # 首轮：风控看到了自相矛盾的数据，判 high
    assert "12.4%" in risk_llm.messages[0][1]
    assert "5.1%" in risk_llm.messages[0][1]
    # 返工后：结论被改掉、轮次推进、复审降级、照常成稿
    assert industry_llm.calls == 2
    assert result["analyses"]["industry_analyst"]["conclusion"] == "中性"
    assert result["analyses"]["industry_analyst"]["confidence"] == "低"
    assert result["revision_round"] == 1
    assert result["risk_review"]["severity"] == "medium"
    # 第八步：finalize 不再是 stub，改为规则式组装并渲染 Markdown 报告
    assert result["final_report"].startswith("# ")
    assert result["pending_challenges"]["industry_analyst"] == []
    assert "revised_in_round=1" in risk_llm.messages[1][1]
