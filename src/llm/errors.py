"""LLM 调用异常分类。

目的是把「可以重试的坏输出」和「重试也没用的硬故障」分开，
避免网络/鉴权错误被当成 JSON 校验失败反复重试、并把真实原因掩盖掉。
"""


class LLMTransientError(RuntimeError):
    """可重试：JSON 解析失败、Pydantic 校验失败、限流、空响应。

    这类错误重试一次有实际意义。
    """


class LLMFatalError(RuntimeError):
    """不可重试：网络不可达 / 鉴权失败（401）/ 其他未预期错误。

    重试只会浪费时间并放大故障，直接降级并把原始信息带进 planner_error。
    """
