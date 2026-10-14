"""DeepSeek 客户端封装（DeepSeek 兼容 OpenAI SDK）。"""

import os

import openai
from dotenv import load_dotenv
from openai import OpenAI

from src.llm.errors import LLMFatalError, LLMTransientError

# .env -> os.environ（不覆盖已存在的环境变量，即 环境变量 > .env 文件）
load_dotenv()

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
CHAT_COMPLETIONS_TIMEOUT = 60.0
PLANNER_TEMPERATURE = 0.2


def _named_errors(*names):
    """按名字从 openai 包内取异常类，跨 SDK 版本位置变化时不至于直接 ImportError。"""
    found = []
    for name in names:
        exc = getattr(openai, name, None)
        if isinstance(exc, type) and issubclass(exc, BaseException):
            found.append(exc)
    return tuple(found)


_NETWORK_ERRORS = _named_errors("APIConnectionError", "APITimeoutError")
_AUTH_ERRORS = _named_errors("AuthenticationError", "PermissionDeniedError")
_RATE_LIMIT_ERRORS = _named_errors("RateLimitError")

# 优先级：鉴权 > 限流 > 网络。因为 RateLimitError 在这版 SDK 里是
# APIConnectionError 的子类，顺序反了会把限流误判成硬故障。
_FATAL_KINDS = "、".join(e.__name__ for e in _AUTH_ERRORS) or "AuthError"
_TRANSIENT_KINDS = "、".join(
    e.__name__ for e in (*_RATE_LIMIT_ERRORS, *_NETWORK_ERRORS)
) or "NetworkError"

# 限流归为「可重试」的理由：RateLimitError 是服务端过载/配额瞬时打满，
# 不是配置错误——重试 1 次 + 稍后重跑就能恢复；而 401 (AuthenticationError)
# 重试多少次都一样，必须立刻暴露。
_TEMPORARY_MARKERS = ("rate limit", "429", "overloaded", "timeout", "timed out")
_PERMANENT_MARKERS = ("401", "403", "invalid api key", "authentication", "unauthorized")


def classify_openai_error(exc: BaseException) -> "LLMTransientError | LLMFatalError":
    """把 openai 异常 / 空响应 / 未知异常映射成我们自己的两类错误。"""
    if isinstance(exc, LLMTransientError | LLMFatalError):
        return exc
    if _AUTH_ERRORS and isinstance(exc, _AUTH_ERRORS):
        return LLMFatalError(f"鉴权失败（不可重试）: {type(exc).__name__}")
    if _RATE_LIMIT_ERRORS and isinstance(exc, _RATE_LIMIT_ERRORS):
        return LLMTransientError(f"触发限流（可重试）: {type(exc).__name__}")
    if _NETWORK_ERRORS and isinstance(exc, _NETWORK_ERRORS):
        text = str(exc).lower()
        if any(m in text for m in _PERMANENT_MARKERS):
            return LLMFatalError(f"网络/鉴权类硬故障（不可重试）: {type(exc).__name__}")
        return LLMTransientError(f"网络异常（可重试）: {type(exc).__name__}")
    text = str(exc).lower()
    if any(m in text for m in _TEMPORARY_MARKERS):
        return LLMTransientError(f"临时性错误（可重试）: {type(exc).__name__}")
    return LLMFatalError(f"未预期的 LLM 错误（不可重试）: {type(exc).__name__}")


class DeepSeekClient:
    """极薄的客户端封装：只负责发请求、返回原始 JSON 字符串。

    本类不做任何 JSON 解析或校验——解析/校验属于各 Agent 节点的职责。
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
    ):
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError(
                "缺少 DEEPSEEK_API_KEY：请设置环境变量或写入项目根目录的 .env "
                '（PowerShell 临时设置：$env:DEEPSEEK_API_KEY = "sk-..."），'
                "也可显式传入 DeepSeekClient(api_key=...)。"
            )
        self.model = model
        self.client = OpenAI(api_key=key, base_url=base_url)

    def chat(self, system: str, user: str) -> str:
        """通用对话：强制 JSON 输出，返回原始 JSON 字符串（不解析）。"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},  # 强制 JSON 输出
                temperature=PLANNER_TEMPERATURE,
                timeout=CHAT_COMPLETIONS_TIMEOUT,
            )
        except Exception as exc:
            raise classify_openai_error(exc) from exc

        content = resp.choices[0].message.content
        if not content or not content.strip():
            raise LLMTransientError("模型返回空内容（可重试）")
        return content

    def plan_outline(self, query: str) -> str:
        """请求研究大纲，返回原始 JSON 字符串（不解析）。行为与接入 chat 前一致。"""
        from src.graph.nodes import PLANNER_SYSTEM_PROMPT

        return self.chat(PLANNER_SYSTEM_PROMPT, query)
