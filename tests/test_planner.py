"""planner 节点测试：全部用 FakeLLM，不联网。"""

import json

import pytest

from src.graph.builder import build_graph
from src.graph.nodes import make_planner_node
from src.llm.client import DeepSeekClient
from src.llm.errors import LLMFatalError, LLMTransientError

VALID_JSON = json.dumps(
    {
        "sections": [
            {
                "title": "公司概况与业务结构",
                "research_questions": ["主营构成是什么？", "各业务线毛利率多少？"],
            },
            {
                "title": "竞争格局与风险",
                "research_questions": ["主要竞争对手是谁？", "政策风险有哪些？"],
            },
        ]
    },
    ensure_ascii=False,
)


class FakeLLM:
    """可预设返回序列的假 LLM，记录调用次数。

    responses 元素既可以是字符串（返回值），也可以是异常实例（抛出）。
    本轮起 planner 走 llm.chat(system, user)，plan_outline 保留做兼容。
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.plan_outline_calls = 0

    def _next(self):
        self.calls += 1
        idx = min(self.calls - 1, len(self.responses) - 1)
        item = self.responses[idx]
        if isinstance(item, BaseException):
            raise item
        return item

    def chat(self, system: str, user: str) -> str:
        return self._next()

    def plan_outline(self, query: str) -> str:
        self.plan_outline_calls += 1
        return self._next()


def test_planner_valid_json():
    llm = FakeLLM([VALID_JSON])
    node = make_planner_node(llm)
    out = node({"query": "宁德时代"})

    outline = out["outline"]
    assert len(outline["sections"]) == 2
    assert outline["sections"][0]["title"] == "公司概况与业务结构"
    assert len(outline["sections"][0]["research_questions"]) == 2
    assert len(outline["sections"][1]["research_questions"]) == 2
    assert "planner_error" not in out
    assert llm.calls == 1  # 一次成功，不重试


def test_planner_invalid_then_valid():
    llm = FakeLLM(["这不是 JSON {{{", VALID_JSON])
    node = make_planner_node(llm)
    out = node({"query": "宁德时代"})

    assert llm.calls == 2  # 重试生效
    assert len(out["outline"]["sections"]) == 2
    assert "planner_error" not in out


def test_planner_fallback():
    llm = FakeLLM(["garbage one", "garbage two"])
    node = make_planner_node(llm)
    out = node({"query": "宁德时代"})

    assert llm.calls == 2  # 用满 2 次尝试
    assert out["outline"]["status"] == "stub"
    assert out["planner_error"]  # 非空错误摘要
    assert out["outline"]["error"] == out["planner_error"]


def test_planner_fatal_no_retry():
    """LLMFatalError（如 401）必须直接降级，不浪费重试。"""
    llm = FakeLLM([LLMFatalError("鉴权失败（不可重试）: AuthenticationError")])
    node = make_planner_node(llm)
    out = node({"query": "宁德时代"})

    assert llm.calls == 1  # 不重试
    assert out["outline"]["status"] == "stub"
    assert out["outline"]["error_kind"] == "fatal"
    assert "鉴权失败" in out["planner_error"]


def test_planner_transient_then_success():
    """LLMTransientError（如限流）可重试，第二次成功。"""
    llm = FakeLLM([LLMTransientError("触发限流（可重试）: RateLimitError"), VALID_JSON])
    node = make_planner_node(llm)
    out = node({"query": "宁德时代"})

    assert llm.calls == 2
    assert len(out["outline"]["sections"]) == 2
    assert "planner_error" not in out


def test_graph_with_injected_llm_runs():
    llm = FakeLLM([VALID_JSON])
    app = build_graph(planner_llm=llm)
    result = app.invoke({"query": "宁德时代", "revision_round": 0})

    assert len(result["outline"]["sections"]) == 2
    assert result["risk_review"]["severity"] == "low"  # 缺省 low，直接成稿
    # 第八步：finalize 不再是 stub，改为规则式组装并渲染 Markdown 报告
    assert result["final_report"].startswith("# ")
    assert llm.calls == 1


def test_client_requires_api_key(monkeypatch):
    """缺 DEEPSEEK_API_KEY 时必须抛 RuntimeError（不联网、不硬编码 key）。

    语义不变：conftest 已清空环境变量并把 load_dotenv 变成 no-op，
    这里再显式加固一次，防止将来加载顺序变化让真实 key 漏进测试。
    """
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("src.llm.client.load_dotenv", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        DeepSeekClient()
