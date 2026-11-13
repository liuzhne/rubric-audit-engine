"""Chat 协议评分请求的输出约束：发不发 Schema，以及按主机族把 Schema 改写成对方接受的写法。

方案见 ``docs/Chat评分请求严格Schema方案.md``。内核（``core_view``）为每条规则生成一份严格 Schema，
Responses 原样发送；Chat 的各家兼容端点对 JSON Schema 的支持不一样，所以发送前在适配器层改写，
内核的 Schema 不动。被放宽的约束仍由本地校验把关（``decode_core_response``），放宽只会让
「厂商拦不住、我们拦住」的回答多一些，不会让不合格的回答被采用。

改某个方言的改写规则要升它的版本号：方言标签写在 ``model_version`` 里，是运行身份的一部分。
"""

from __future__ import annotations

from backend.app.core.config import settings
from backend.app.services.llm import hosts


JSON_SCHEMA = "json_schema"
JSON_OBJECT = "json_object"
OFF = "off"
CHAT_STRUCTURED_OUTPUT_MODES = (JSON_SCHEMA, JSON_OBJECT, OFF)

PLAIN = "plain"
GOOGLE = "google"
_VERSIONS = {PLAIN: 1, GOOGLE: 1}

#: 没有显式设置时默认发 Schema 的主机族。只收对真实接口核实过的（记来源与日期）；
#: 目前为空——Vertex 的官方文档写明兼容接口支持 json_schema，但我们的 Schema 能否被接受还没有实测。
VERIFIED_STRICT_SCHEMA_FAMILIES: frozenset = frozenset()

_SUBSCHEMA_KEYS = ("items", "additionalProperties")
_SUBSCHEMA_LIST_KEYS = ("anyOf", "allOf", "oneOf", "prefixItems")
_SUBSCHEMA_MAP_KEYS = ("properties", "$defs", "definitions")


def strict_schema_enabled() -> bool:
    return bool(settings.SCORING_CHAT_STRICT_SCHEMA_ENABLED)


def resolve_chat_response_format(base_url, *, structured_output=None, response_format_json=False) -> str:
    """Core 评分请求的输出约束：``json_schema`` / ``json_object`` / ``none``。

    总开关关闭时与改动前完全相同（只看 ``response_format_json``），连接上的 ``structured_output``
    存着但不生效。打开后：连接显式设置优先，其次是旧的 ``response_format_json``，再其次是核实过
    的主机族清单。
    """

    legacy = JSON_OBJECT if response_format_json else "none"
    if not strict_schema_enabled():
        return legacy
    mode = str(structured_output or "").strip()
    if mode == JSON_SCHEMA:
        return JSON_SCHEMA
    if mode == JSON_OBJECT:
        return JSON_OBJECT
    if mode == OFF:
        return "none"
    if response_format_json:
        return JSON_OBJECT
    if hosts.host_family(base_url) in VERIFIED_STRICT_SCHEMA_FAMILIES:
        return JSON_SCHEMA
    return "none"


def chat_dialect(base_url) -> str:
    return GOOGLE if hosts.host_family(base_url) in hosts.GOOGLE_FAMILIES else PLAIN


def enum_limit() -> int:
    return int(settings.SCORING_CHAT_SCHEMA_ENUM_LIMIT)


def dialect_label(dialect: str) -> str:
    """写进 ``model_version`` 的标签；Google 方言带上枚举上限（它决定实际发出的 Schema）。"""

    label = "%s@%d" % (dialect, _VERSIONS[dialect])
    return label + "/e%d" % enum_limit() if dialect == GOOGLE else label


def model_version_suffix(response_format: str, base_url) -> str:
    if response_format != JSON_SCHEMA:
        return ""
    return ";schema=" + dialect_label(chat_dialect(base_url))


def apply(dialect: str, schema):
    """返回改写后的新 Schema；不修改传入的对象。"""

    if dialect == GOOGLE:
        return _google(schema, enum_limit())
    if dialect == PLAIN:
        return _copy(schema)
    raise ValueError("unknown schema dialect")


def _copy(node):
    if isinstance(node, dict):
        return {key: _copy(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_copy(item) for item in node]
    return node


def _google(node, limit: int):
    """Vertex / Gemini API：依据见方案第 3 节（官方文档，2026-11-13）。

    - 可空类型写成 ``nullable: true``，不用类型数组；
    - 枚举只允许非空字符串：去掉 ``null``，值太多时整个去掉（「枚举值很多」会被拒绝）；
    - ``minLength`` 不在支持列表里，删掉；
    - 必须为空的数组不带空对象的条目定义。
    """

    if isinstance(node, list):
        return [_google(item, limit) for item in node]
    if not isinstance(node, dict):
        return node
    result = {}
    for key, value in node.items():
        if key == "minLength":
            continue
        if key in _SUBSCHEMA_MAP_KEYS and isinstance(value, dict):
            result[key] = {name: _google(item, limit) for name, item in value.items()}
        elif key in _SUBSCHEMA_KEYS or key in _SUBSCHEMA_LIST_KEYS:
            result[key] = _google(value, limit)
        else:
            result[key] = _copy(value)
    kind = result.get("type")
    if isinstance(kind, list):
        concrete = [item for item in kind if item != "null"]
        if len(concrete) == 1:
            result["type"] = concrete[0]
            if "null" in kind:
                result["nullable"] = True
    enum = result.get("enum")
    if isinstance(enum, list):
        values = [item for item in enum if isinstance(item, str) and item]
        if None in enum:
            result["nullable"] = True
        if not values or len(values) > limit:
            del result["enum"]
        else:
            result["enum"] = values
    if result.get("type") == "array" and result.get("maxItems") == 0:
        result["items"] = {"type": "string"}
    return result


def chat_response_format(base_url, *, name: str, schema) -> dict:
    """Chat Completions 的 ``response_format`` 值（严格 Schema）。"""

    return {
        "type": JSON_SCHEMA,
        "json_schema": {"name": name, "strict": True, "schema": apply(chat_dialect(base_url), schema)},
    }


#: 测试连接用的小 Schema：形状与评分的回答一致（必填的枚举字段 + 可空字段），不含材料内容。
PROBE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "level_code"],
    "properties": {
        "status": {"type": "string", "enum": ["triggered", "not_triggered"]},
        "level_code": {"type": ["string", "null"], "enum": ["A", None]},
    },
}


__all__ = [
    "CHAT_STRUCTURED_OUTPUT_MODES",
    "GOOGLE",
    "JSON_OBJECT",
    "JSON_SCHEMA",
    "OFF",
    "PLAIN",
    "PROBE_SCHEMA",
    "VERIFIED_STRICT_SCHEMA_FAMILIES",
    "apply",
    "chat_dialect",
    "chat_response_format",
    "dialect_label",
    "enum_limit",
    "model_version_suffix",
    "resolve_chat_response_format",
    "strict_schema_enabled",
]
