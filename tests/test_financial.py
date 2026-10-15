"""财务分析 Agent 测试：FakeLLM + FakeProvider，不联网、不 import akshare。"""

import json

from src.graph.builder import build_graph
from src.graph.nodes import extract_ticker, make_financial_analyst_node
from src.llm.errors import LLMFatalError, LLMTransientError
from src.tools.financial_data import FakeProvider

FAKE_FINANCIALS = {
    "ticker": "300750",
    "source": "fake",
    "revenue": 4.0e11,
    "net_profit": 5.0e10,
    "roe": {"value": 22.5, "as_of": "2025-03"},
    "gross_margin": {"value": 26.1, "as_of": "2025-03"},
    "net_margin": {"value": 12.5, "as_of": "2025-03"},
    "cashflow": {"value": 3.2e10, "as_of": "2025-03"},
}

VALID_ANALYSIS = json.dumps(
    {
        "agent": "financial_analyst",
        "conclusion": "增持",
        "confidence": "中",
        "key_points": [
            {"finding": "ROE 22.5%，处于行业高位", "data_source": "roe (2025-03)"},
            {"finding": "毛利率 26.1%，同比小幅回落", "data_source": "gross_margin (2025-03)"},
        ],
        "concerns": ["毛利率存在下行压力"],
        "assumptions": ["假设报告期后无重大资产重组"],
    },
    ensure_ascii=False,
)


class FakeLLM:
    """可预设返回序列的假 LLM，记录调用次数（元素可为异常实例）。"""

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


def make_node(llm, provider=None):
    provider = provider or FakeProvider({"300750": FAKE_FINANCIALS})
    return make_financial_analyst_node(llm, provider), provider


def test_analyst_valid():
    llm = FakeLLM([VALID_ANALYSIS])
    node, _ = make_node(llm)
    out = node({"query": "宁德时代 300750"})

    analysis = out["analyses"]["financial_analyst"]
    assert analysis["agent"] == "financial_analyst"
    assert analysis["conclusion"] == "增持"
    assert analysis["confidence"] == "中"
    assert len(analysis["key_points"]) == 2
    assert "analyst_error" not in out
    assert llm.calls == 1


def test_analyst_transient_retry():
    llm = FakeLLM([LLMTransientError("JSON 解析失败: 坏输出"), VALID_ANALYSIS])
    node, _ = make_node(llm)
    out = node({"query": "宁德时代 300750"})

    assert llm.calls == 2  # 重试生效
    assert out["analyses"]["financial_analyst"]["conclusion"] == "增持"
    assert "analyst_error" not in out


def test_analyst_fatal_no_retry():
    llm = FakeLLM([LLMFatalError("鉴权失败（不可重试）: AuthenticationError")])
    node, _ = make_node(llm)
    out = node({"query": "宁德时代 300750"})

    assert llm.calls == 1  # 不重试
    analysis = out["analyses"]["financial_analyst"]
    assert analysis["status"] == "unavailable"
    assert analysis["error_kind"] == "fatal"
    assert out["analyst_error"]


def test_analyst_provider_failure():
    class BoomProvider:
        def get_financials(self, ticker):
            raise RuntimeError("akshare 挂了")

    llm = FakeLLM([VALID_ANALYSIS])
    node = make_financial_analyst_node(llm, BoomProvider())
    out = node({"query": "宁德时代 300750"})

    analysis = out["analyses"]["financial_analyst"]
    assert analysis["status"] == "unavailable"
    assert analysis["error_kind"] == "data_source"
    assert "akshare 挂了" in analysis["error"]
    assert out["analyst_error"]
    assert llm.calls == 0  # 不编造数据：根本没调 LLM


def test_analyst_ticker_extraction():
    llm = FakeLLM([VALID_ANALYSIS])
    node, provider = make_node(llm)

    node({"query": "宁德时代 300750"})
    node({"query": "300750"})

    assert provider.calls == ["300750", "300750"]
    assert extract_ticker("宁德时代 300750") == "300750"
    assert extract_ticker("300750") == "300750"


def test_graph_with_analyst_runs():
    llm = FakeLLM([VALID_ANALYSIS])
    provider = FakeProvider({"300750": FAKE_FINANCIALS})
    app = build_graph(financial_llm=llm, data_provider=provider)
    result = app.invoke({"query": "宁德时代 300750", "revision_round": 0})

    assert result["analyses"]["financial_analyst"]["conclusion"] == "增持"
    assert result["risk_review"]["severity"] == "low"
    # 第八步：finalize 不再是 stub，改为规则式组装并渲染 Markdown 报告
    assert result["final_report"].startswith("# ")
    assert llm.calls == 1
    assert provider.calls == ["300750"]
