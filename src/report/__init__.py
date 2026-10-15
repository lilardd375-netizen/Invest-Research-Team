"""报告层：规则式组装 + Markdown 渲染 + 落盘（纯函数，不依赖 LLM）。"""

from src.report.render import (
    AGENT_ORDER,
    DEFAULT_REPORT_DIR,
    build_rule_based_report,
    render_report,
    report_path,
    write_report,
)

__all__ = [
    "AGENT_ORDER",
    "DEFAULT_REPORT_DIR",
    "build_rule_based_report",
    "render_report",
    "report_path",
    "write_report",
]
