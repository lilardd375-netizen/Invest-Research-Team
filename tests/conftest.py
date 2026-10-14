"""pytest 全局配置：保证测试确定性 —— 不加载 .env 的真实 key。"""

import os

import pytest
from dotenv import load_dotenv


@pytest.fixture(autouse=True)
def _no_env_key(monkeypatch):
    """清空 DEEPSEEK_API_KEY，并把 load_dotenv 变成 no-op。

    否则客户端模块导入时的 load_dotenv() 会把 .env 里的真实 key 带进测试环境，
    使 test_client_requires_api_key 这类「缺 key」语义失效。
    """
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("src.llm.client.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    os.environ.pop("DEEPSEEK_API_KEY", None)
    yield
