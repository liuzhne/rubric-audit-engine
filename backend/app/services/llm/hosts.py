"""模型服务的主机族：同一协议下各厂商的差异按主机族区分（docs/模型思考能力表方案.md 第 3.2 节）。

鉴权头（``auth_headers``）与思考能力表（``thinking_capabilities``）都按主机判断，识别集中在这里，
不再各写一份。只看主机名，不看路径与端口；未知主机归为 ``other``。
"""

from __future__ import annotations

from urllib.parse import urlsplit


VERTEX = "vertex"
GEMINI_API = "gemini_api"
ZHIPU = "zhipu"
OPENROUTER = "openrouter"
DASHSCOPE = "dashscope"
DEEPSEEK = "deepseek"
MOONSHOT = "moonshot"
OPENAI = "openai"
ANTHROPIC = "anthropic"
BEDROCK = "bedrock"
OTHER = "other"

GOOGLE_FAMILIES = frozenset({VERTEX, GEMINI_API})

_EXACT_HOSTS = {
    "aiplatform.googleapis.com": VERTEX,
    "generativelanguage.googleapis.com": GEMINI_API,
    "open.bigmodel.cn": ZHIPU,
    "api.z.ai": ZHIPU,
    "openrouter.ai": OPENROUTER,
    "dashscope.aliyuncs.com": DASHSCOPE,
    "dashscope-intl.aliyuncs.com": DASHSCOPE,
    "api.deepseek.com": DEEPSEEK,
    "api.moonshot.cn": MOONSHOT,
    "api.moonshot.ai": MOONSHOT,
    "api.openai.com": OPENAI,
    "api.anthropic.com": ANTHROPIC,
}


def hostname(base_url) -> str:
    return (urlsplit(str(base_url or "")).hostname or "").rstrip(".").casefold()


def host_family(base_url) -> str:
    """``base_url`` 所属的主机族；认不出的一律 ``other``。"""

    host = hostname(base_url)
    if not host:
        return OTHER
    exact = _EXACT_HOSTS.get(host)
    if exact:
        return exact
    # Vertex 区域端点：<区域>-aiplatform.googleapis.com
    if host.endswith("-aiplatform.googleapis.com"):
        return VERTEX
    # Bedrock：bedrock-runtime.<区域>.amazonaws.com 与 bedrock-mantle.<区域>.api.aws
    first = host.split(".", 1)[0]
    if first in {"bedrock-runtime", "bedrock-mantle"}:
        return BEDROCK
    return OTHER


def is_vertex_host(base_url) -> bool:
    return host_family(base_url) == VERTEX


def is_google_host(base_url) -> bool:
    """Vertex AI（全局与区域端点）或 Gemini API 的 OpenAI 兼容端点。"""

    return host_family(base_url) in GOOGLE_FAMILIES


__all__ = [
    "ANTHROPIC",
    "BEDROCK",
    "DASHSCOPE",
    "DEEPSEEK",
    "GEMINI_API",
    "GOOGLE_FAMILIES",
    "MOONSHOT",
    "OPENAI",
    "OPENROUTER",
    "OTHER",
    "VERTEX",
    "ZHIPU",
    "host_family",
    "hostname",
    "is_google_host",
    "is_vertex_host",
]
