"""Pydantic v2 结构化输出模型。"""

from typing import Literal

from pydantic import BaseModel, Field


class Section(BaseModel):
    """研究大纲的一个章节。"""

    title: str
    research_questions: list[str]


class OutlineModel(BaseModel):
    """主编 Agent 的输出结构。"""

    sections: list[Section] = Field(default_factory=list)


class KeyPoint(BaseModel):
    """财务分析的一条核心发现，必须能追溯到具体数据来源。"""

    finding: str
    data_source: str


class AnalysisBase(BaseModel):
    """四个分析师 Agent 的公共输出结构。

    conclusion 三档跨 Agent 统一，是为了第八步报告合成时能把四家结论聚合比较；
    各子类只固定 agent 字段，不得改变其余字段的语义。
    """

    conclusion: Literal["增持", "中性", "减持"]
    confidence: Literal["高", "中", "低"]
    key_points: list[KeyPoint] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class FinancialAnalysisModel(AnalysisBase):
    """财务分析 Agent 的输出结构。"""

    agent: Literal["financial_analyst"] = "financial_analyst"


class IndustryAnalysisModel(AnalysisBase):
    """行业分析 Agent 的输出结构（板块归属、板块表现、同业对比）。"""

    agent: Literal["industry_analyst"] = "industry_analyst"


class SentimentAnalysisModel(AnalysisBase):
    """舆情分析 Agent 的输出结构（近期新闻与市场情绪）。"""

    agent: Literal["sentiment_analyst"] = "sentiment_analyst"


class TechnicalAnalysisModel(AnalysisBase):
    """技术面分析 Agent 的输出结构（均线、区间位置、量能）。"""

    agent: Literal["technical_analyst"] = "technical_analyst"


class Challenge(BaseModel):
    """风控对某个 Agent 输出的一条具体质询。"""

    target_agent: str  # 被质询的 Agent，如 financial_analyst
    issue: str  # 问题是什么
    required_action: str  # 要求对方补充/修正什么


class RiskReviewModel(BaseModel):
    """风控 Agent 的评审结论。

    severity 语义（决定是否返工，必须按这个标准打分）：

    - high   = 存在**数据错误**或**重大逻辑漏洞**，结论不可采信，**应返工**；
    - medium = 假设偏激进但可接受，**报告中披露即可**，不必返工；
    - low    = 常规提示，**并入风险章节**即可。

    判定要克制：medium / low 不是失败，把可接受的问题报成 high 会让流程空转。
    拿不准时不要武断扣分，写进 challenges 要求补充数据。
    """

    severity: Literal["high", "medium", "low"]
    challenges: list[Challenge] = Field(default_factory=list)
    summary: str = ""  # 一句话总评


class AgentSummary(BaseModel):
    """报告里对单个分析师的逐家呈现。

    特意做成 list 而不是「四家合成一句话」：分歧本身是研究结论的一部分，
    谁看多谁看空必须留在报告里，不允许被综合结论吞掉。
    """

    agent: str
    status: str = "ok"  # ok / unavailable（数据源或 LLM 降级）
    conclusion: str = ""  # 增持/中性/减持；unavailable 时为空
    confidence: str = ""  # 高/中/低
    summary: str = ""  # 一句话要点


class RevisionRound(BaseModel):
    """返工记录的一轮。

    degraded=True 表示**该轮风控复核本身降级了**（网络/服务端故障），
    此时 severity 是兜底写死的 low，不是真实评审结论——报告必须如实披露，
    否则读者会把「没评」误当成「没问题」。
    """

    round: int = 0
    severity: str = ""
    degraded: bool = False
    challenge_targets: dict[str, int] = Field(default_factory=dict)
    revised_agents: list[str] = Field(default_factory=list)
    note: str = ""


class ReportModel(BaseModel):
    """报告合成 Agent 的输出结构（第八步）。

    约束：
    - per_agent_summaries 必须逐家呈现（四家都出现），不许只给综合结论；
    - revision_history 与 disclosures 是独立字段，渲染时也会独立成章；
    - 不得引入四家结论之外的外部评级数据。
    """

    title: str
    stock_code: str
    report_date: str  # YYYY-MM-DD
    overall_conclusion: Literal["增持", "中性", "减持"]
    overall_confidence: Literal["高", "中", "低"]
    per_agent_summaries: list[AgentSummary] = Field(default_factory=list)
    key_thesis: list[str] = Field(default_factory=list)  # 3-5 条支撑逻辑
    key_risks: list[str] = Field(default_factory=list)
    revision_history: list[RevisionRound] = Field(default_factory=list)
    disclosures: list[str] = Field(default_factory=list)
    generated_at: str = ""
