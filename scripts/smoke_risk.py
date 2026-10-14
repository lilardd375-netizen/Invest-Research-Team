"""真实链路冒烟：DeepSeek + akshare，planner -> financial_analyst -> risk_review。

用法：
    .venv\\Scripts\\python.exe scripts\\smoke_risk.py

要求：
- 需要 DEEPSEEK_API_KEY（.env 或环境变量），**不打印任何 key 明文**；
- 会消耗真实 token，并访问 akshare 取真实财务数据。
- 打印风控产出的 severity / challenges / summary，以及整图最终走了几轮返工。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.builder import MAX_REVISION_ROUNDS, build_graph  # noqa: E402
from src.llm.client import DeepSeekClient  # noqa: E402
from src.tools.financial_data import AkshareProvider  # noqa: E402

QUERY = "宁德时代 300750 的投资价值"
TICKER = "300750"


class CountingLLM:
    """透明代理：包一层真实客户端，只为统计调用次数。"""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def chat(self, system: str, user: str) -> str:
        self.calls += 1
        return self.inner.chat(system, user)

    def plan_outline(self, query: str) -> str:
        self.calls += 1
        return self.inner.plan_outline(query)


class _Snapshot:
    """固定快照 provider：冒烟里只联网取一次数据，避免重复请求 akshare。"""

    def __init__(self, data):
        self.data = data

    def get_financials(self, ticker):
        return self.data


def main() -> int:
    llm = CountingLLM(DeepSeekClient())
    data = AkshareProvider().get_financials(TICKER)

    app = build_graph(
        planner_llm=llm,
        financial_llm=llm,
        risk_llm=llm,
        data_provider=_Snapshot(data),
    )

    result = app.invoke(
        {"query": QUERY, "revision_round": 0, "round_at_last_review": -1},
        {"recursion_limit": 30},
    )

    print("=== 数据源 ===")
    print(f"  report_period={data['report_period']}  as_of={data['as_of']}")
    print(f"  缺失指标={data.get('missing_metrics')}")

    print("=== 风控结论 ===")
    review = result["risk_review"]
    print(f"  severity = {review.get('severity')}")
    print(f"  stuck    = {review.get('stuck')}")
    print(f"  summary  = {review.get('summary')}")
    challenges = review.get("challenges") or []
    print(f"  challenges ({len(challenges)} 条):")
    for i, ch in enumerate(challenges, 1):
        print(f"   [{i}] target={ch.get('target_agent')}")
        print(f"       issue    : {ch.get('issue')}")
        print(f"       required : {ch.get('required_action')}")

    print("=== 最终路径 ===")
    rounds = result.get("revision_round", 0)
    print(f"  返工轮次 = {rounds} / 上限 {MAX_REVISION_ROUNDS}")
    print(f"  round_at_last_review = {result.get('round_at_last_review')}")
    print(f"  LLM 总调用次数 = {llm.calls}（1 planner + 1 财务 + 风控复核次数）")
    print(f"  final_report = {result.get('final_report')!r}（成稿节点仍是占位，第六步实现）")
    errors = {k: v for k, v in result.items() if k.endswith("_error")}
    print(f"  降级信息 = {errors or '无'}")

    # 密钥不落地检查
    key = os.environ.get("DEEPSEEK_API_KEY") or ""
    blob = json.dumps(result, ensure_ascii=False, default=str)
    assert not key or key not in blob, "密钥明文出现在结果里"
    print("[OK] 结果中不含 key 明文")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
