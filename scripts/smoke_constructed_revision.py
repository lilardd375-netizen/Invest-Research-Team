"""构造式真实链路冒烟（第六步收尾）：证明 high -> revise -> severity 下降 成立。

思路（不放松任何判罚标准，只是"钓鱼"）：
- planner / 风控 用真实 DeepSeekClient；
- 财务分析师用 SeededFirstResponseLLM 包装真实客户端：**第一次 chat() 返回
  硬编码的 flawed 分析 JSON**（基于 300750 真实数据写就、能通过
  FinancialAnalysisModel 校验，但含 2 处"已覆盖维度"的口径/逻辑错误），
  第二次起透传真实客户端 —— 于是 revise 之后是真的模型在修订；
- 数据源用真实 akshare（同一快照只取一次，避免接口抖动造成假失败）。

两处 flaw（都不是覆盖类问题，覆盖不足只会触发 medium）：
1. 逻辑/口径：净利增速 45.31% < 营收增速 54.80%（净利率在收缩），
   却写成"盈利与收入同步扩张、增长质量较好"；
2. 数据自相矛盾：同一份 JSON 里 ROE 先写 18.60%（近年高位）、
   后写 11.41%（同比下降），引用值与口径自相矛盾。

跑法：.\.venv\Scripts\python.exe scripts\smoke_constructed_revision.py
输出中出现 key 明文视为失败。
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # Windows 控制台中文
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

from src.graph.builder import build_graph  # noqa: E402
from src.llm.client import DeepSeekClient  # noqa: E402
from src.tools.financial_data import AkshareProvider  # noqa: E402

QUERY = "分析宁德时代（300750）的投资价值"

# 硬编码的 flawed 分析（数字取自 300750 真实财报快照，报告期 2026-06-30）
FLAWED_ANALYSIS = {
    "agent": "financial_analyst",
    "conclusion": "增持",
    "confidence": "高",
    "key_points": [
        {
            "finding": "2026 年中报营业收入同比 +54.80%、净利润同比 +45.31%，"
            "盈利与收入同步扩张，增长质量较好。",
            "data_source": "akshare:stock_financial_analysis_indicator (2026-06-30)",
        },
        {
            "finding": "净资产收益率 18.60%，处于近年高位，股东回报能力突出。",
            "data_source": "akshare:stock_financial_analysis_indicator (2026-06-30)",
        },
        {
            "finding": "净资产收益率 11.41%，同比下降 3.2 个百分点，"
            "主要受总资产周转率下滑拖累。",
            "data_source": "akshare:stock_financial_analysis_indicator (2026-06-30)",
        },
        {
            "finding": "资产负债率 63.65%，处于制造业合理区间，杠杆水平可控。",
            "data_source": "akshare:stock_financial_analysis_indicator (2026-06-30)",
        },
    ],
    "concerns": ["销售毛利率数据缺失，毛利率趋势无法验证。"],
    "assumptions": ["假设动力电池行业需求维持当前增速。"],
}


class SeededFirstResponseLLM:
    """真实客户端的薄包装：第一次 chat() 返回种子分析，其后全部透传。"""

    def __init__(self, real, seeded_json: str):
        self.real = real
        self.seeded_json = seeded_json
        self.calls = 0
        self.responses: list[str] = []

    def chat(self, system: str, user: str) -> str:
        self.calls += 1
        if self.calls == 1:
            out = self.seeded_json
        else:
            out = self.real.chat(system, user)
        self.responses.append(out)
        return out

    def plan_outline(self, query: str) -> str:
        return self.real.plan_outline(query)


class RecordingRiskLLM:
    """真实客户端的薄包装：透传，同时记录每轮风控的 severity / 质询数。"""

    def __init__(self, real):
        self.real = real
        self.calls = 0
        self.rounds: list[dict] = []

    def chat(self, system: str, user: str) -> str:
        self.calls += 1
        raw = self.real.chat(system, user)
        try:
            data = json.loads(raw)
            self.rounds.append(
                {
                    "round": self.calls,
                    "severity": data.get("severity"),
                    "challenges": len(data.get("challenges") or []),
                    "summary": data.get("summary", ""),
                }
            )
        except Exception:
            self.rounds.append(
                {"round": self.calls, "severity": "?", "challenges": None, "summary": raw[:200]}
            )
        return raw

    def plan_outline(self, query: str) -> str:
        return self.real.plan_outline(query)


class SnapshotProvider:
    """包裹真实 provider：整轮冒烟只取一次数据快照（前后两轮口径一致）。"""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.payload = None

    def get_financials(self, ticker: str) -> dict:
        if self.payload is None:
            self.payload = self.inner.get_financials(ticker)
        self.calls += 1
        return self.payload


def _dump(label: str, obj) -> None:
    print(f"--- {label} ---")
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def main() -> int:
    key = os.environ.get("DEEPSEEK_API_KEY") or ""
    real = DeepSeekClient()
    print(f"DEEPSEEK_API_KEY 已配置: {bool(key)}（长度 {len(key)}，不打印明文）")

    seeded = SeededFirstResponseLLM(real, json.dumps(FLAWED_ANALYSIS, ensure_ascii=False))
    risk = RecordingRiskLLM(real)
    provider = SnapshotProvider(AkshareProvider())

    app = build_graph(
        planner_llm=real,
        financial_llm=seeded,
        risk_llm=risk,
        data_provider=provider,
    )
    result = app.invoke({"query": QUERY, "revision_round": 0})

    print("\n=== 1. 种子分析（分析师首轮，构造式 flawed）===")
    print(f"conclusion={FLAWED_ANALYSIS['conclusion']} confidence={FLAWED_ANALYSIS['confidence']}")
    for i, kp in enumerate(FLAWED_ANALYSIS["key_points"], 1):
        print(f"  {i}. {kp['finding']}")

    print("\n=== 2. 风控每轮输出 ===")
    for r in risk.rounds:
        print(f"  第 {r['round']} 轮: severity={r['severity']} challenges={r['challenges']}")
        print(f"           summary: {r['summary']}")

    final_analysis = (result.get("analyses") or {}).get("financial_analyst") or {}
    print("\n=== 3. 修订前后 diff（分析师真实输出）===")
    print(f"  conclusion : {FLAWED_ANALYSIS['conclusion']} -> {final_analysis.get('conclusion')}")
    print(f"  confidence : {FLAWED_ANALYSIS['confidence']} -> {final_analysis.get('confidence')}")
    before = [kp["finding"] for kp in FLAWED_ANALYSIS["key_points"]]
    after = [kp.get("finding", "") for kp in final_analysis.get("key_points") or []]
    print(f"  key_points : {len(before)} 条 -> {len(after)} 条")
    print("  [修订后要点]")
    for i, f in enumerate(after, 1):
        print(f"    {i}. {f}")
    print(f"  revised_in_round = {final_analysis.get('revised_in_round')}")
    print(f"  剩余待办质询 pending_challenges = {result.get('pending_challenges')}")
    print(f"  concerns = {json.dumps(final_analysis.get('concerns'), ensure_ascii=False)}")

    severities = [r["severity"] for r in risk.rounds]
    first_high = bool(severities) and severities[0] == "high"
    recovered = len(severities) > 1 and severities[-1] in ("medium", "low")
    print("\n=== 4. 闭环判定 ===")
    print(f"  首轮 severity=high        : {first_high}（实际首轮 {severities[0] if severities else 'N/A'}）")
    print(f"  复审降级为 medium/low     : {recovered}（实际末轮 {severities[-1] if severities else 'N/A'}）")
    print(f"  severity 序列             : {severities}")
    print(f"  revision_round            : {result.get('revision_round')}")
    print(f"  final_report              : {result.get('final_report')!r}")
    print(f"  数据源 get_financials 调用: {provider.calls} 次（快照 provider 只真实取数 1 次）")
    print(f"  分析师 LLM 调用次数       : {seeded.calls}（首轮为种子，其余真实）")
    print(f"  风控 LLM 调用次数         : {risk.calls}")
    print(f"  planner_error/analyst_error/risk_error: "
          f"{result.get('planner_error')!r} / {result.get('analyst_error')!r} / {result.get('risk_error')!r}")

    blob = json.dumps(result, ensure_ascii=False, default=str) + "".join(seeded.responses)
    leaked = bool(key) and key in blob
    print("\n=== 5. 密钥检查 ===")
    print(f"  输出/返回值中出现 key 明文: {leaked}（必须为 False）")

    ok = first_high and recovered and not leaked
    print(f"\n结论: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
