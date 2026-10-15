"""图状态定义。"""

import operator
from typing import Annotated, NotRequired, TypeAlias, TypedDict, Union

_MergeDict: TypeAlias = Annotated[dict, operator.or_]


class ResearchState(TypedDict):
    """投资研究流水线在节点之间传递的状态。"""

    query: str
    outline: dict
    analyses: Annotated[dict, operator.or_]  # 并行节点合并用
    risk_review: dict  # {"severity": str, "stuck": bool}
    revision_round: int
    final_report: str
    # 防死循环保险丝：上一次风控复核时的 revision_round
    round_at_last_review: int
    # 仅降级路径出现（各 Agent 调 LLM / 校验失败时）
    planner_error: NotRequired[str]
    analyst_error: NotRequired[str]
    risk_error: NotRequired[str]
    # revise 分发下来的待办质询：target_agent -> list[dict]。
    # 并行分析师各自清自己的 key 会在同一步写这个字段，所以必须是 or_ reducer。
    pending_challenges: NotRequired[_MergeDict]
    # 三个新分析 Agent 的降级摘要：agent -> 错误字符串。
    # 财务 Agent 的单数 analyst_error 是 LastValue 且全图只有财务节点写它，
    # 不会冲突，保持不动；新 Agent 并行降级时可能同一步写多份，必须用 reducer，
    # 否则报 "Can receive only one value per step"。
    analyst_errors: NotRequired[_MergeDict]


def merge_dict(left: Union[dict, None], right: Union[dict, None]) -> dict:
    """dict 合并 reducer（等价于 operator.or_，用于并行节点合并 analyses）。"""
    return operator.or_(left or {}, right or {})
