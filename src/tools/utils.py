"""数据工具层公共清洗工具（纯函数，除 pandas 对象外不引入额外依赖）。

各 Provider 共用的事：
1. clean_value：NaN / inf / 空值 -> None，数字字符串 -> float，其余原样返回；
2. period_key：日期 / 日期字符串归一化成可比较的 YYYYMMDD；
3. pick_latest_row：akshare 多数接口按日期**升序**返回（首行最老），必须自己挑最新一期；
4. clean_metrics：按 {中文列名: 输出字段名} 逐项清洗，被丢弃的字段记入 missing；
5. normalize_ticker：从任意文本里取 6 位股票代码；
6. fetch_with_retry + call_with_timeout：外部接口抖动（对端断连）时的重试包装，
   以及**硬超时**——akshare 多数接口不暴露 timeout 参数（stock_zh_a_hist 默认
   timeout=None 即永不超时），网络半死会无限期挂住，必须由本地线程兜底；
7. ensure_ascii_ca：把 certifi 的 CA 复制到 ASCII 路径并设置 curl / requests 的
   CA 环境变量——libcurl（curl_cffi）无法从含中文的路径加载 CAfile（本机实测抛
   SSLError: curl: (77) error adding trust anchors），而本工作区路径正是中文路径，
   不处理的话 akshare 里所有走 curl_cffi 的接口（如 stock_news_em）必然失败。
"""

import math
import os
import re
import shutil
import threading
import time
from datetime import date, datetime
from pathlib import Path

_TICKER_RE = re.compile(r"\d{6}")

#: 单次外部调用允许的最长等待（秒）。所有 Provider 的真实调用都走这个默认值。
DEFAULT_TIMEOUT = 20.0

#: 设 IR_SKIP_CA_BOOTSTRAP=1 可整体跳过 CA 引导（单元测试用，避免在仓库外落文件）。
CA_SKIP_ENV = "IR_SKIP_CA_BOOTSTRAP"
#: 设 IR_ASCII_CA_DIR=<ASCII 目录> 可指定 CA 副本的存放位置。
CA_DIR_ENV = "IR_ASCII_CA_DIR"
_CA_ENV_VARS = ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")
_CA_READY = False
CA_ASCII_PATH: str | None = None


def _ascii_ca_dirs(source: Path) -> list[Path]:
    """候选存放目录（必须是 ASCII 路径）：环境变量 > 原 CA 所在盘根 > 系统盘根 > ProgramData。"""
    candidates: list[Path] = []
    override = os.environ.get(CA_DIR_ENV)
    if override:
        candidates.append(Path(override))
    candidates.append(Path(source.anchor or "C:\\") / "irt_ca")
    candidates.append(Path(os.environ.get("SystemDrive", "C:") + "\\") / "irt_ca")
    program_data = os.environ.get("ProgramData")
    if program_data:
        candidates.append(Path(program_data) / "irt_ca")
    unique: list[Path] = []
    for item in candidates:
        if str(item).isascii() and item not in unique:
            unique.append(item)
    return unique


def ensure_ascii_ca() -> str | None:
    """让 libcurl / curl_cffi 能加载 CA 证书，返回实际使用的 ASCII CA 路径（无法处理时 None）。

    为什么需要：curl_cffi 底层是 libcurl，CAfile 路径含中文时加载失败并抛
    SSLError: curl: (77) error adding trust anchors from locations: CAfile: ...
    而本工作区路径是 D:\\桌面\\校招\\...（含中文）。同一份 cacert.pem 放到 ASCII
    路径后，同样的请求返回 200、ak.stock_news_em 恢复 (10,6)——即问题在路径编码，
    不在证书本身。

    做法：把 certifi 的 cacert.pem 复制到 ASCII 目录，并用 CURL_CA_BUNDLE /
    REQUESTS_CA_BUNDLE / SSL_CERT_FILE 指过去（已存在同名环境变量时不动它）。
    该函数在模块导入时就调用一次，保证环境变量先于 curl_cffi 首次发请求生效；
    重复调用是幂等的。任何一步失败都静默返回 None，绝不影响主流程。
    """
    global _CA_READY, CA_ASCII_PATH
    if _CA_READY:
        return CA_ASCII_PATH
    _CA_READY = True
    if os.environ.get(CA_SKIP_ENV):
        return None
    try:
        import certifi

        source = Path(certifi.where())
    except Exception:  # noqa: BLE001 - 没有 certifi 就与 libcurl 无关
        return None
    if not source.is_file():
        return None
    if str(source).isascii():
        CA_ASCII_PATH = str(source)
        return CA_ASCII_PATH
    for directory in _ascii_ca_dirs(source):
        target = directory / source.name
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if not target.is_file() or target.stat().st_size != source.stat().st_size:
                shutil.copyfile(source, target)
            if target.stat().st_size != source.stat().st_size:
                continue
        except OSError:
            continue
        CA_ASCII_PATH = str(target)
        for name in _CA_ENV_VARS:
            os.environ.setdefault(name, CA_ASCII_PATH)
        return CA_ASCII_PATH
    return None


def clean_value(value):
    """numpy / NaN 清洗：把数字（含数字字符串）归一成纯 Python float，NaN、inf 返回 None。

    必要性：
    1. akshare 返回 numpy 标量，NaN 经 json.dumps(default=str) 会变成字符串
       "nan"——那是**非法 JSON**，会让 LLM 拿到脏上下文并解析失败；
    2. 有的接口把数字给成字符串（如 "63.6525"），统一成 float 才能比较与计算。
    清洗不掉的非数字文本按原样返回（例如 "--"），由上游决定怎么用。
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def period_key(value) -> str:
    """把日期归一化成可比较的 YYYYMMDD 字符串（空值 -> ""）。"""
    if isinstance(value, datetime | date):
        return value.strftime("%Y%m%d")
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def pick_latest_row(df, date_col: str = "日期"):
    """按日期列挑出最新一期，返回 (row, date_raw)。

    akshare 各接口的行序不统一（有升序也有降序），所以一律按解析出的
    period_key 取最大者，而不是假设首行/末行是最新。
    """
    if date_col not in df.columns:
        raise ValueError(f"缺少日期列: {date_col}")
    periods = df[date_col].map(period_key)
    if periods.eq("").all():
        raise ValueError(f"日期列无法解析: {date_col}")
    latest = df.loc[periods.idxmax()]
    return latest, str(latest.get(date_col)).strip()


def clean_metrics(row, field_map: dict) -> tuple[dict, list[str]]:
    """按 {中文列名: 输出字段名} 逐项清洗，返回 (values, missing)。

    missing 是被丢弃（None）的输出字段名列表——上游要么写进 missing_metrics，
    要么据此判定数据不可用，绝不能把 None 混进喂给 LLM 的 JSON。
    """
    values: dict = {}
    missing: list[str] = []
    for column, field in field_map.items():
        value = clean_value(row.get(column))
        if value is None:
            missing.append(field)
        else:
            values[field] = value
    return values, missing


def normalize_ticker(text) -> str:
    """从文本里取出 6 位股票代码（取首个匹配）；找不到时返回去空白的原文本。"""
    match = _TICKER_RE.search(str(text or ""))
    return match.group(0) if match else str(text or "").strip()


def call_with_timeout(fn, *, seconds: float | None = None, label: str = "数据接口"):
    """在 daemon 线程里执行 fn，超过 seconds 还没返回就抛 TimeoutError。

    为什么用线程而不是 socket 超时：akshare 的多数接口不接收 timeout 参数
    （实测 stock_zh_a_hist 的 timeout 默认 None = 永不超时），带重试的封装只能
    覆盖"快速失败"，覆盖不了"卡住不动"。走线程兜底后，调用方一定能拿到异常并
    按各自的降级路径走 unavailable，不会挂死。
    超时后不等待线程收尾（daemon）：底层 socket 可能仍在后台慢慢烂掉，但调用方
    已经可以继续。
    """
    limit = DEFAULT_TIMEOUT if seconds is None else seconds
    result: list = []
    failure: list = []

    def _target() -> None:
        try:
            result.append(fn())
        except BaseException as exc:  # noqa: BLE001 - 原样带回给调用方
            failure.append(exc)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(limit)
    if thread.is_alive():
        raise TimeoutError(f"{label} 超过 {limit}s 未返回（已放弃等待）")
    if failure:
        raise failure[0]
    return result[0] if result else None


def fetch_with_retry(
    fn,
    *,
    attempts: int = 3,
    delay: float = 1.0,
    delays: tuple[float, ...] | None = None,
    label: str = "数据接口",
    timeout: float | None = None,
):
    """调用易抖动的外部接口：每次尝试都套硬超时，失败重试；全部失败时抛异常。

    为什么需要这层：本机实测 eastmoney 的 push2 域名经常在**首次连接**就被对端断连
    （requests.exceptions.ConnectionError: RemoteDisconnected('Remote end closed
    connection without response')），同一进程内重试往往就能成功。行业/舆情/技术面
    三个 Provider 都依赖这类接口，不加重试会随机失败。

    退避：默认每次固定等 delay 秒；传 delays（如 (1.5, 3.0, 6.0, 12.0)）则按序列
    递增等待，用于实测会突发限流的接口（push2his 的日线：实测连抛失败后，等 ~1 分钟
    新进程首次调用即恢复，固定 1.5s 的耐心兜不住）。

    末次失败若是超时则抛 TimeoutError（节点据此走 unavailable），否则抛 RuntimeError
    并在消息里带上最后一次的原因。
    """
    ensure_ascii_ca()  # 中文路径下 libcurl 读不了 certifi 的 CA，必须在首次请求前兜底
    limit = DEFAULT_TIMEOUT if timeout is None else timeout
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call_with_timeout(fn, seconds=limit, label=label)
        except Exception as exc:  # noqa: BLE001 - 外部接口抛什么异常不可控
            last_exc = exc
            if attempt < attempts:
                if delays:
                    time.sleep(delays[min(attempt - 1, len(delays) - 1)])
                else:
                    time.sleep(delay)
    if isinstance(last_exc, TimeoutError):
        raise TimeoutError(
            f"{label} 连续 {attempts} 次超时（每次上限 {limit}s）"
        ) from last_exc
    raise RuntimeError(
        f"{label} 连续 {attempts} 次调用失败: {type(last_exc).__name__}: {last_exc}"
    ) from last_exc


ensure_ascii_ca()  # 导入即生效：保证环境变量先于任何 Provider 的 curl_cffi 请求
