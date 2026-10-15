"""openai 异常 -> 可重试 / 不可重试 的归类断言。

不联网：用真实 openai SDK 异常类构造实例（本地内省 openai 3.31.0 确认构造方式）。
"""

import httpx
import openai

from src.llm.client import classify_openai_error
from src.llm.errors import LLMFatalError, LLMTransientError

_REQUEST = httpx.Request("POST", "https://api.deepseek.com/chat/completions")


def _status_error(cls, status_code):
    """构造真实 SDK 状态异常实例（APIStatusError 需要 httpx.Response）。"""
    return cls("boom", response=httpx.Response(status_code, request=_REQUEST), body=None)


def test_500_is_transient():
    """500 InternalServerError 必须可重试（第六步真实冒烟撞到过，当时被误判为 fatal）。"""
    exc = _status_error(openai.InternalServerError, 500)
    assert exc.status_code == 500
    assert isinstance(classify_openai_error(exc), LLMTransientError)


def test_other_5xx_is_transient():
    """SDK 未具名子类的 5xx（如 503）同样按 status_code 归为可重试。"""
    exc = _status_error(openai.APIStatusError, 503)
    assert isinstance(classify_openai_error(exc), LLMTransientError)


def test_401_is_fatal():
    """401 鉴权失败重试多少次都一样，必须立刻暴露。"""
    exc = _status_error(openai.AuthenticationError, 401)
    assert isinstance(classify_openai_error(exc), LLMFatalError)


def test_api_timeout_is_transient():
    """APITimeoutError 归可重试：服务端排队/网络抖动，非配置错误，重试代价低于降级。"""
    exc = openai.APITimeoutError(request=_REQUEST)
    assert isinstance(classify_openai_error(exc), LLMTransientError)


def test_rate_limit_stays_transient():
    """429 必须命中限流分支，不能落到「4xx -> 不可重试」的兜底里。"""
    exc = _status_error(openai.RateLimitError, 429)
    assert isinstance(classify_openai_error(exc), LLMTransientError)
