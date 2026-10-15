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
    # revise 分发下来的待办质询：target_agent -> list[dict]
    pending_challenges: NotRequired[dict]


def merge_dict(left: Union[dict, None], right: Union[dict, None]) -> dict:
    """dict 合并 reducer（等价于 operator.or_，用于并行节点合并 analyses）。"""
    return operator.or_(left or {}, right or {})
