"""pytest 全局配置：保证测试确定性 —— 不加载 .env 的真实 key、不做 CA 引导。"""

import os

import pytest
from dotenv import load_dotenv

# 必须在任何 src.tools 模块被导入之前设置：utils 导入时会做 ASCII CA 引导
# （中文路径下把 certifi 的 CA 复制到 ASCII 目录并设环境变量），单元测试不该
# 在仓库外落文件，故整体跳过；需要验证该机制时由测试自己 monkeypatch 打开。
os.environ["IR_SKIP_CA_BOOTSTRAP"] = "1"


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
