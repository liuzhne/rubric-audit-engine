"""模型思考能力表：按「协议 + 主机族 + 模型名前缀」决定思考怎么发（docs/模型思考能力表方案.md）。

用户规则（2026-11-10）：
- 能关闭思考的模型直接关闭，输出上限沿用现有值；
- 关不掉的发最低思考强度，不给思考设 token 上限，只限制正式输出；
- 由适配器层识别能不能关、怎么关、最低强度怎么发。

解析顺序：连接的显式配置 → 能力表（具体前缀 → 族默认记录）→ 未知模型策略（不发参数，按可能在思考处理）。
结果是一个不可变的 ``ThinkingPlan``，适配器在构造时算好，每次请求只合并它的 ``fields()``。

作用范围由 ``THINKING_CAPABILITY_SCOPE`` 决定：``off`` 全部旧行为；``auxiliary``（默认）AI 任务与评分助手
按能力表、评分路径旧行为；``all`` 评分路径也按能力表（QWK 复核、维护者批准后再切）。

**改了某条记录的请求写法，必须同时改它的身份值（换强度名）或升 ``CORE_PROVIDER_PROMPT_VERSION``**：
运行身份只记开关与强度，不记写法（方案第 6.2 节）；``test_thinking_capabilities`` 的写法快照会提醒。
"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from types import MappingProxyType
from typing import Mapping

from backend.app.services.llm import hosts


logger = logging.getLogger(__name__)

CHAT = "chat"
RESPONSES = "responses"
MESSAGES = "messages"
PROTOCOLS = (CHAT, RESPONSES, MESSAGES)

#: 思考开着、连接没写输出上限时的请求上限：取 min(模型最大输出, 这个值)（Q2，2026-11-10 确认）。
#: 改动走代码评审，不做成环境变量。
THINKING_ON_MAX_TOKENS_CAP = 32_768
#: 思考开着时 AI 任务的缺省单次等待（秒）；连接显式超时仍优先（最多 300）。
THINKING_ON_TIMEOUT_SECONDS = 240

PROVIDER_DEFAULT = "provider-default"
DISABLED_VALUES = frozenset({"disabled", "none", "false", "off"})

SOURCE_EXPLICIT = "explicit"
SOURCE_ADJUSTED = "explicit_adjusted"
SOURCE_TABLE = "table"
SOURCE_FAMILY_DEFAULT = "family_default"
SOURCE_UNKNOWN = "unknown"
SOURCE_LEGACY = "legacy"
#: 能力表做出的「可以关闭」判断：观测到思考时记一行 thinking_capability_mismatch。
TABLE_SOURCES = frozenset({SOURCE_TABLE, SOURCE_FAMILY_DEFAULT})

SCOPE_OFF = "off"
SCOPE_AUXILIARY = "auxiliary"
SCOPE_ALL = "all"
SCOPES = (SCOPE_OFF, SCOPE_AUXILIARY, SCOPE_ALL)


def _frozen(value):
    if value is None:
        return None
    return MappingProxyType(deepcopy(dict(value)))


@dataclass(frozen=True)
class ThinkingCapability:
    """能力表的一条记录。请求写法是合并进请求体的 JSON 片段。"""

    protocols: tuple[str, ...]
    families: tuple[str, ...]
    #: 规整后的模型名前缀；空字符串是这个主机族的默认记录，优先级最低。
    model_prefix: str
    can_disable: bool
    #: 关闭思考的写法；``{}`` 表示「省略即关闭」。
    off_request: Mapping | None
    #: 最低强度的名字（进运行身份的 ``;effort=`` 后缀）；关不掉又没有强度参数时为 None。
    lowest_effort: str | None
    effort_request: Mapping | None
    #: 模型的最大输出 token；文档没写明时为 None（思考开着时取项目封顶）。
    max_output_tokens: int | None
    sources: tuple[str, ...]
    verified_on: str
    note: str = ""

    def __post_init__(self):
        object.__setattr__(self, "off_request", _frozen(self.off_request))
        object.__setattr__(self, "effort_request", _frozen(self.effort_request))

    @property
    def key(self) -> str:
        return "%s:%s:%s" % ("+".join(self.protocols), "+".join(self.families), self.model_prefix or "*")


@dataclass(frozen=True)
class ThinkingPlan:
    """一次解析的结果；适配器构造时算一次，之后不变。"""

    #: 可能在思考：决定运行身份的 enabled、请求上限与超时。
    thinking_on: bool
    #: 运行身份里的 ``thinking.type``（只表示开关，见方案第 6.2 节）。
    identity_type: str | None
    #: 发出的强度；没发为 None。进 ``model_version`` 的 ``;effort=`` 后缀。
    effort: str | None
    request_fields: Mapping = field(default_factory=dict)
    source: str = SOURCE_LEGACY
    entry: str | None = None
    model_max_output_tokens: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "request_fields", _frozen(self.request_fields) or MappingProxyType({}))

    def fields(self) -> dict:
        """要合并进请求体的思考字段（每次返回新副本）。"""

        return deepcopy(dict(self.request_fields))

    @property
    def identity(self) -> dict:
        return {"enabled": bool(self.thinking_on), "type": self.identity_type}

    @property
    def legacy(self) -> bool:
        return self.source == SOURCE_LEGACY

    @property
    def raises_output_limit(self) -> bool:
        """思考开着、按能力表处理的计划：请求上限放到封顶值（旧行为不放）。"""

        return self.thinking_on and not self.legacy

    def raised_max_tokens(self) -> int:
        limit = self.model_max_output_tokens or THINKING_ON_MAX_TOKENS_CAP
        return max(1, min(int(limit), THINKING_ON_MAX_TOKENS_CAP))

    def log_label(self) -> str:
        return "on" if self.thinking_on else "off"


def _cap(protocols, families, prefix, *, can_disable, off=None, effort=None, effort_request=None,
         max_output=None, sources, verified_on, note=""):
    return ThinkingCapability(
        protocols=tuple(protocols),
        families=tuple(families),
        model_prefix=prefix,
        can_disable=can_disable,
        off_request=off,
        lowest_effort=effort,
        effort_request=effort_request,
        max_output_tokens=max_output,
        sources=tuple(sources),
        verified_on=verified_on,
        note=note,
    )


# --- 来源 ----------------------------------------------------------------------------------

_GEMINI_OPENAI = "https://aistudio.google.com/docs/openai"
_GEMINI_THINKING = "https://aistudio.google.com/docs/thinking"
_VERTEX_OPENAI = "https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/openai"
_VERTEX_GEMINI3 = "https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/get-started-with-gemini-3"
_VERTEX_25_FLASH = "https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash"
_PRODUCTION_2026_10_20 = "生产验证 2026-10-20（DECISIONS「Vertex 关闭思考改发 thinking_budget 0」）"
_ZHIPU_THINKING = "https://docs.bigmodel.cn/cn/guide/capabilities/thinking.md"
_ZAI_MIGRATE = "https://docs.z.ai/guides/overview/migrate-to-glm-new"
_OPENROUTER_REASONING = "https://openrouter.ai/docs/use-cases/reasoning-tokens"
_CLAUDE_THINKING = "https://platform.claude.com/docs/en/build-with-claude/thinking.md"
_CLAUDE_EFFORT = "https://platform.claude.com/docs/en/build-with-claude/effort.md"
_CLAUDE_MODELS = "https://platform.claude.com/docs/en/about-claude/models/overview.md"
_OPENAI_GPT51 = "https://developers.openai.com/api/docs/models/gpt-5.1"
_OPENAI_GPT5 = "https://developers.openai.com/api/docs/models/gpt-5"
_OPENAI_REASONING = "https://developers.openai.com/api/docs/guides/reasoning.md"
_OPENAI_GPT41 = "https://developers.openai.com/api/docs/models/gpt-4.1"
_OPENAI_GPT41_MINI = "https://developers.openai.com/api/docs/models/gpt-4.1-mini"
_DEEPSEEK_THINKING = "https://api-docs.deepseek.com/guides/thinking_mode"
_BAILIAN_THINKING = "https://help.aliyun.com/zh/model-studio/deep-thinking"
_KIMI_THINKING = "https://platform.kimi.com/docs/guide/use-kimi-k2-thinking-model"

_GOOGLE = (hosts.VERTEX, hosts.GEMINI_API)
_CLAUDE_HOSTS = (hosts.ANTHROPIC, hosts.BEDROCK)
_OPENAI_PROTOCOLS = (CHAT, RESPONSES)


def _google_low(prefix, effort, *, max_output, sources, families=_GOOGLE, note=""):
    return _cap((CHAT,), families, prefix, can_disable=False, effort=effort,
                effort_request={"reasoning_effort": effort}, max_output=max_output,
                sources=sources, verified_on="2026-11-10", note=note)


def _claude_off(prefix, off, *, max_output=128_000, note=""):
    return _cap((MESSAGES,), _CLAUDE_HOSTS, prefix, can_disable=True, off=off, max_output=max_output,
                sources=(_CLAUDE_THINKING, _CLAUDE_MODELS), verified_on="2026-11-10", note=note)


def _claude_forced(prefix, *, note=""):
    # Bedrock 是否接受 output_config.effort 尚未核实：那里不发强度（按模型默认），身份如实记开启。
    return (
        _cap((MESSAGES,), (hosts.ANTHROPIC,), prefix, can_disable=False, effort="low",
             effort_request={"output_config": {"effort": "low"}}, max_output=128_000,
             sources=(_CLAUDE_THINKING, _CLAUDE_EFFORT), verified_on="2026-11-10", note=note),
        _cap((MESSAGES,), (hosts.BEDROCK,), prefix, can_disable=False, max_output=128_000,
             sources=(_CLAUDE_THINKING,), verified_on="2026-11-10",
             note="Bedrock 是否接受 output_config.effort 待核实，暂不发强度"),
    )


def _bailian_hybrid(prefix, note=""):
    return _cap((CHAT, RESPONSES), (hosts.DASHSCOPE,), prefix, can_disable=True,
                off={"enable_thinking": False}, sources=(_BAILIAN_THINKING,), verified_on="2026-11-11",
                note=note)


def _bailian_forced(prefix, note=""):
    # 百炼文档只给了 thinking_budget（思考 token 上限），用户规则不给思考设上限：不发参数。
    return _cap((CHAT, RESPONSES), (hosts.DASHSCOPE,), prefix, can_disable=False,
                sources=(_BAILIAN_THINKING,), verified_on="2026-11-11",
                note=note or "仅思考模式，没有强度参数")


CAPABILITIES: tuple[ThinkingCapability, ...] = (
    # --- Google（Vertex / Gemini API 的 OpenAI 兼容端点）------------------------------------
    _cap((CHAT,), (hosts.VERTEX,), "gemini-2.5-flash", can_disable=True,
         off={"extra_body": {"google": {"thinking_config": {"thinking_budget": 0}}}},
         max_output=65_535, sources=(_PRODUCTION_2026_10_20, _VERTEX_25_FLASH, _GEMINI_OPENAI),
         verified_on="2026-11-10",
         note="Vertex 的 reasoning_effort 不收 none（2026-10-20 生产 400），改用 Gemini 专属参数"),
    _cap((CHAT,), (hosts.GEMINI_API,), "gemini-2.5-flash", can_disable=True,
         off={"reasoning_effort": "none"}, max_output=65_535,
         sources=(_GEMINI_OPENAI, _VERTEX_25_FLASH), verified_on="2026-11-10"),
    _google_low("gemini-2.5-pro", "low", max_output=None,
                sources=(_GEMINI_OPENAI, _VERTEX_OPENAI),
                note="关不掉；low 在两家都映射到 1024 思考预算"),
    _google_low("gemini-3-pro", "low", max_output=65_536,
                sources=(_GEMINI_THINKING, _VERTEX_GEMINI3), note="只有 low / high"),
    _google_low("gemini-3.1-pro", "low", max_output=65_536,
                sources=(_GEMINI_THINKING, _VERTEX_GEMINI3), note="low / medium / high"),
    _google_low("gemini-3-flash", "minimal", max_output=65_536,
                sources=(_GEMINI_THINKING, _VERTEX_GEMINI3, _VERTEX_OPENAI),
                note="MINIMAL 仍会思考（尽量接近零预算）"),
    _google_low("gemini-3.5-flash-lite", "minimal", max_output=None,
                sources=(_GEMINI_THINKING, _VERTEX_OPENAI)),
    _google_low("gemini-3.6-flash", "minimal", max_output=None, sources=(_GEMINI_THINKING, _VERTEX_OPENAI)),
    _google_low("gemini-3.8-flash", "low", max_output=None, sources=(_GEMINI_THINKING, _VERTEX_OPENAI)),
    _google_low("gemini-4-argon", "low", max_output=None, sources=(_GEMINI_THINKING, _GEMINI_OPENAI),
                families=(hosts.GEMINI_API,), note="Vertex 写法待核实"),
    # --- 智谱 --------------------------------------------------------------------------------
    _cap((CHAT,), (hosts.ZHIPU,), "glm-5.2", can_disable=True, off={"thinking": {"type": "disabled"}},
         sources=(_ZHIPU_THINKING,), verified_on="2026-11-10", note="禁用思考仅 GLM-5.2 支持"),
    _cap((CHAT,), (hosts.ZHIPU,), "glm-5.3", can_disable=False, effort="low",
         effort_request={"reasoning_effort": "low"}, max_output=128_000,
         sources=(_ZHIPU_THINKING, _ZAI_MIGRATE), verified_on="2026-11-10",
         note="强制思考，传 disabled 报错；reasoning_effort 只收 max / high / low，默认 max"),
    # --- OpenRouter：沿用 2026-09-14 起草路径的做法（族默认）------------------------------------
    _cap((CHAT,), (hosts.OPENROUTER,), "", can_disable=True, off={"reasoning": {"enabled": False}},
         sources=(_OPENROUTER_REASONING,), verified_on="2026-11-10",
         note="强制推理的模型 OpenRouter 会拒绝关闭；遇到时补具体记录"),
    # --- Claude ------------------------------------------------------------------------------
    *_claude_forced("claude-opus-5-5", note="thinking: disabled 在任何强度下都 400"),
    *_claude_forced("claude-fable-5"),
    *_claude_forced("claude-mythos-5"),
    _claude_off("claude-sonnet-5-5", {"thinking": {"type": "between_tools"}},
                note="disabled 会 400；between_tools 只在强度 ≤ high 时可用（不发强度时默认 high）"),
    _claude_off("claude-opus-5", {"thinking": {"type": "disabled"}}, note="disabled 只在强度 ≤ high 时可用"),
    _claude_off("claude-sonnet-5", {"thinking": {"type": "disabled"}}),
    _claude_off("claude-haiku-5-5", {"thinking": {"type": "disabled"}}, note="disabled 只在强度 ≤ high 时可用"),
    _claude_off("claude-opus-4-8", {}, note="省略 thinking 即不思考"),
    _claude_off("claude-opus-4-7", {}, note="省略 thinking 即不思考"),
    _claude_off("claude-opus-4-6", {}, note="省略 thinking 即不思考"),
    _claude_off("claude-sonnet-4-6", {}, note="省略 thinking 即不思考"),
    _claude_off("claude-haiku-4-5", {}, max_output=64_000, note="省略 thinking 即不思考"),
    # --- OpenAI（api.openai.com）---------------------------------------------------------------
    _cap((RESPONSES,), (hosts.OPENAI,), "gpt-5.1", can_disable=True, off={"reasoning": {"effort": "none"}},
         max_output=128_000, sources=(_OPENAI_GPT51, _OPENAI_REASONING), verified_on="2026-11-11",
         note="effort 支持 none（默认）/ low / medium / high"),
    _cap((CHAT,), (hosts.OPENAI,), "gpt-5.1", can_disable=True, off={"reasoning_effort": "none"},
         max_output=128_000, sources=(_OPENAI_GPT51, _OPENAI_REASONING), verified_on="2026-11-11"),
    _cap((RESPONSES,), (hosts.OPENAI,), "gpt-5", can_disable=False, effort="minimal",
         effort_request={"reasoning": {"effort": "minimal"}}, max_output=128_000,
         sources=(_OPENAI_GPT5, _OPENAI_REASONING), verified_on="2026-11-11",
         note="effort 支持 minimal / low / medium / high，没有 none"),
    _cap((CHAT,), (hosts.OPENAI,), "gpt-5", can_disable=False, effort="minimal",
         effort_request={"reasoning_effort": "minimal"}, max_output=128_000,
         sources=(_OPENAI_GPT5, _OPENAI_REASONING), verified_on="2026-11-11"),
    # 非推理模型（含 -mini / -nano）：本来就不思考，什么都不发即关闭。
    _cap((CHAT, RESPONSES), (hosts.OPENAI,), "gpt-4.1", can_disable=True, off={}, max_output=32_768,
         sources=(_OPENAI_GPT41, _OPENAI_GPT41_MINI), verified_on="2026-11-11",
         note="官方页：non-reasoning / without a reasoning step"),
    # --- DeepSeek ------------------------------------------------------------------------------
    *(
        _cap((CHAT,), (hosts.DEEPSEEK,), prefix, can_disable=True, off={"thinking": {"type": "disabled"}},
             sources=(_DEEPSEEK_THINKING,), verified_on="2026-11-11", note="默认开启思考，强度 high")
        for prefix in ("deepseek-flash", "deepseek-v4-pro")
    ),
    # --- Kimi（月之暗面自营）---------------------------------------------------------------------
    _cap((CHAT,), (hosts.MOONSHOT,), "kimi-k3.1-pro", can_disable=False, effort="low",
         effort_request={"reasoning_effort": "low"}, sources=(_KIMI_THINKING,), verified_on="2026-11-11",
         note="始终思考；reasoning_effort 只收 low / high / max，默认 max"),
    _cap((CHAT,), (hosts.MOONSHOT,), "kimi-k3", can_disable=False, effort="low",
         effort_request={"reasoning_effort": "low"}, sources=(_KIMI_THINKING,), verified_on="2026-11-11",
         note="始终思考；reasoning_effort 只收 low / high / max，默认 max"),
    _cap((CHAT,), (hosts.MOONSHOT,), "kimi-k2.7-code", can_disable=False, sources=(_KIMI_THINKING,),
         verified_on="2026-11-11", note="始终思考，disabled 报错，不支持 reasoning_effort"),
    _cap((CHAT,), (hosts.MOONSHOT,), "kimi-k2.6", can_disable=True, off={"thinking": {"type": "disabled"}},
         sources=(_KIMI_THINKING,), verified_on="2026-11-11"),
    # --- 阿里云百炼（混合思考：enable_thinking=false 关闭；仅思考模式关不掉）--------------------------
    *(
        _bailian_hybrid(prefix)
        for prefix in (
            "qwen3.8-max", "qwen3.8-flash", "qwen3.8-27b", "qwen3.8-omni-flash",
            "qwen3.7-max", "qwen3.7-plus", "qwen3.7-flash",
            "qwen3.6-plus", "qwen3.6-flash",
            "qwen3.5-plus", "qwen3.5-flash",
            "qwen-plus", "qwen-flash",
            "kimi-k3.1-pro", "kimi-k3", "kimi-k2.6", "kimi-k2.5",
            "deepseek-v4.1-flash", "deepseek-v4-pro",
            "glm-5.2", "glm-5.1", "glm-5",
        )
    ),
    *(
        _bailian_forced(prefix)
        for prefix in (
            "qwen3.8-2.4t-a95b", "qwen3.7-max-preview", "qwen3.7-max-2026-05-17",
            "kimi-k2.7-code", "glm-5.3", "deepseek-r1",
        )
    ),
)


# --- 规整与匹配 ---------------------------------------------------------------------------------

#: 前缀之后紧跟版本号的延续（数字、`.数字`、`-一到两位数字`）就不算命中；日期快照与变体后缀算命中。
_VERSION_CONTINUATION = re.compile(r"(?:\d|\.\d|-\d{1,2}(?!\d))")
_BEDROCK_ANTHROPIC = "anthropic."


def normalize_model_name(model_name) -> str:
    """转小写、去掉 ``/`` 前缀与 Bedrock 的推理配置文件前缀。"""

    name = str(model_name or "").strip().casefold()
    name = name.rsplit("/", 1)[-1]
    index = name.find(_BEDROCK_ANTHROPIC)
    if index == 0 or (index > 0 and name[index - 1] == "."):
        name = name[index + len(_BEDROCK_ANTHROPIC):]
    return name


def prefix_matches(name: str, prefix: str) -> bool:
    if not prefix:
        return True
    if not name.startswith(prefix):
        return False
    return _VERSION_CONTINUATION.match(name, len(prefix)) is None


def lookup(protocol: str, base_url, model_name) -> ThinkingCapability | None:
    """同一协议与主机族里前缀命中、且最长的那条记录。"""

    family = hosts.host_family(base_url)
    name = normalize_model_name(model_name)
    best = None
    for capability in CAPABILITIES:
        if protocol not in capability.protocols or family not in capability.families:
            continue
        if not prefix_matches(name, capability.model_prefix):
            continue
        if best is None or len(capability.model_prefix) > len(best.model_prefix):
            best = capability
    return best


# --- 解析 -------------------------------------------------------------------------------------


def current_scope() -> str:
    from backend.app.core.config import settings

    scope = str(getattr(settings, "THINKING_CAPABILITY_SCOPE", SCOPE_AUXILIARY) or SCOPE_AUXILIARY)
    return scope if scope in SCOPES else SCOPE_AUXILIARY


def _is_disabled(value) -> bool:
    return value is None or str(value).strip().casefold() in DISABLED_VALUES


def _source_of(capability: ThinkingCapability) -> str:
    return SOURCE_FAMILY_DEFAULT if not capability.model_prefix else SOURCE_TABLE


def _off_identity(capability: ThinkingCapability, family: str) -> str | None:
    if family in hosts.GOOGLE_FAMILIES:
        return "none"
    return "disabled" if capability.off_request else None


def _forced_plan(capability: ThinkingCapability, *, source: str, effort=None, extra=None) -> ThinkingPlan:
    """关不掉思考：发最低强度（或连接显式的强度），身份如实记开启。"""

    fields = {} if effort is not None else dict(deepcopy(dict(capability.effort_request or {})))
    if extra:
        fields.update(extra)
    return ThinkingPlan(
        thinking_on=True,
        identity_type=PROVIDER_DEFAULT,
        effort=effort if effort is not None else capability.lowest_effort,
        request_fields=fields,
        source=source,
        entry=capability.key,
        model_max_output_tokens=capability.max_output_tokens,
    )


def _log_adjusted(protocol, base_url, model_name, capability):
    logger.warning(
        "thinking_explicit_adjusted protocol=%s family=%s model=%s entry=%s effort=%s "
        "reason=model_cannot_disable_thinking",
        protocol, hosts.host_family(base_url), str(model_name or "")[:120], capability.key,
        capability.lowest_effort or "-",
    )


def resolve_chat(base_url, model_name, thinking_type) -> ThinkingPlan:
    """Chat Completions：``thinking_type`` 是连接的显式配置（空即没配置）。"""

    family = hosts.host_family(base_url)
    google = family in hosts.GOOGLE_FAMILIES
    capability = lookup(CHAT, base_url, model_name)
    max_output = capability.max_output_tokens if capability else None
    value = str(thinking_type or "").strip()
    if value:
        if value.casefold() in DISABLED_VALUES:
            if capability is not None and capability.can_disable:
                fields = dict(deepcopy(dict(capability.off_request or {})))
                if not google:
                    # 显式写法原样带上（与改动前相同）；能力表的写法同时发出。
                    fields["thinking"] = {"type": value}
                return ThinkingPlan(False, "none" if google else value, None, fields, SOURCE_EXPLICIT,
                                    capability.key, max_output)
            if capability is not None:
                # Q4：表里记着关不掉，显式「关闭」改按最低强度发，身份如实记开启。
                _log_adjusted(CHAT, base_url, model_name, capability)
                return _forced_plan(capability, source=SOURCE_ADJUSTED)
            if google:
                # Google 不认 thinking 字段；不认识的模型不发参数，如实记按模型默认。
                return ThinkingPlan(True, PROVIDER_DEFAULT, None, {}, SOURCE_EXPLICIT, None, None)
            return ThinkingPlan(False, value, None, {"thinking": {"type": value}}, SOURCE_EXPLICIT, None, None)
        fields = {} if google else {"thinking": {"type": value}}
        return ThinkingPlan(True, value, None, fields, SOURCE_EXPLICIT,
                            capability.key if capability else None, max_output)
    if capability is not None:
        if capability.can_disable:
            return ThinkingPlan(False, _off_identity(capability, family), None,
                                capability.off_request or {}, _source_of(capability), capability.key, max_output)
        return _forced_plan(capability, source=_source_of(capability))
    return ThinkingPlan(True, PROVIDER_DEFAULT, None, {}, SOURCE_UNKNOWN, None, None)


def legacy_chat_plan(base_url, model_name, thinking_type) -> ThinkingPlan:
    """改动前的 Chat 行为（``thinking_params`` 的换算），逐字节不变。"""

    from backend.app.services.llm.thinking_params import effective_thinking_type
    from backend.app.services.llm.thinking_params import thinking_request_fields

    identity = effective_thinking_type(base_url, model_name, thinking_type)
    identity = str(identity or "").strip() or None
    return ThinkingPlan(
        thinking_on=not _is_disabled(identity),
        identity_type=identity,
        effort=None,
        request_fields=thinking_request_fields(base_url, model_name, identity or ""),
        source=SOURCE_LEGACY,
    )


def resolve_messages(base_url, model_name, thinking_type=None, effort=None) -> ThinkingPlan:
    """Claude Messages：``thinking_type``（adaptive / disabled）与 ``effort`` 是连接的显式配置。"""

    capability = lookup(MESSAGES, base_url, model_name)
    max_output = capability.max_output_tokens if capability else None
    thinking_type = str(thinking_type or "").strip() or None
    effort = str(effort or "").strip() or None
    if thinking_type == "disabled" and capability is not None and not capability.can_disable:
        _log_adjusted(MESSAGES, base_url, model_name, capability)
        if effort is not None:
            return _forced_plan(capability, source=SOURCE_ADJUSTED, effort=effort,
                                extra={"output_config": {"effort": effort}})
        return _forced_plan(capability, source=SOURCE_ADJUSTED)
    if thinking_type is not None or effort is not None:
        fields = {}
        if thinking_type is not None:
            fields["thinking"] = {"type": thinking_type}
        if effort is not None:
            fields["output_config"] = {"effort": effort}
        if thinking_type is not None:
            on = thinking_type != "disabled"
            identity = thinking_type
        elif capability is not None and capability.can_disable and not capability.off_request:
            # 只配了强度、模型省略 thinking 即不思考（Opus 4.6–4.8 等）：与改动前相同。
            on, identity = False, None
        else:
            on, identity = True, PROVIDER_DEFAULT
        return ThinkingPlan(on, identity, effort, fields, SOURCE_EXPLICIT,
                            capability.key if capability else None, max_output)
    if capability is not None:
        if capability.can_disable:
            return ThinkingPlan(False, _off_identity(capability, hosts.host_family(base_url)), None,
                                capability.off_request or {}, _source_of(capability), capability.key, max_output)
        return _forced_plan(capability, source=_source_of(capability))
    return ThinkingPlan(True, PROVIDER_DEFAULT, None, {}, SOURCE_UNKNOWN, None, None)


def legacy_messages_plan(thinking_type=None, effort=None) -> ThinkingPlan:
    """改动前的 Claude 行为：只发显式配置；没配置的 thinking 记 null。"""

    thinking_type = str(thinking_type or "").strip() or None
    effort = str(effort or "").strip() or None
    fields = {}
    if thinking_type:
        fields["thinking"] = {"type": thinking_type}
    if effort:
        fields["output_config"] = {"effort": effort}
    return ThinkingPlan(
        thinking_on=thinking_type is not None and thinking_type != "disabled",
        identity_type=thinking_type,
        effort=effort,
        request_fields=fields,
        source=SOURCE_LEGACY,
    )


def resolve_responses(base_url, model_name) -> ThinkingPlan:
    """Responses：连接没有思考选项，只看能力表。"""

    capability = lookup(RESPONSES, base_url, model_name)
    if capability is not None:
        if capability.can_disable:
            return ThinkingPlan(False, "disabled" if capability.off_request else None, None,
                                capability.off_request or {}, _source_of(capability), capability.key,
                                capability.max_output_tokens)
        return _forced_plan(capability, source=_source_of(capability))
    return ThinkingPlan(True, PROVIDER_DEFAULT, None, {}, SOURCE_UNKNOWN, None, None)


def legacy_responses_plan() -> ThinkingPlan:
    return ThinkingPlan(False, None, None, {}, SOURCE_LEGACY)


def plans(protocol, base_url, model_name, *, thinking_type=None, effort=None, scope=None):
    """``(辅助计划, 评分计划)``：辅助给 AI 任务与评分助手，评分给 ``score_*``（方案第 4.3 节）。"""

    scope = scope or current_scope()
    if protocol == CHAT:
        legacy = legacy_chat_plan(base_url, model_name, thinking_type)
        new = (lambda: resolve_chat(base_url, model_name, thinking_type))
    elif protocol == MESSAGES:
        legacy = legacy_messages_plan(thinking_type, effort)
        new = (lambda: resolve_messages(base_url, model_name, thinking_type, effort))
    elif protocol == RESPONSES:
        legacy = legacy_responses_plan()
        new = (lambda: resolve_responses(base_url, model_name))
    else:
        raise ValueError("unsupported protocol")
    if scope == SCOPE_OFF:
        return legacy, legacy
    auxiliary = new()
    return auxiliary, (auxiliary if scope == SCOPE_ALL else legacy)


def model_version_suffix(plan: ThinkingPlan, *, max_tokens_explicit: bool) -> str:
    """强度与放开后的请求上限写进 ``model_version``（PromptEnvelope 的 provider 是封闭结构）。"""

    parts = []
    if plan.effort:
        parts.append(";effort=%s" % plan.effort)
    if plan.raises_output_limit and not max_tokens_explicit:
        parts.append(";max_tokens=%d" % plan.raised_max_tokens())
    return "".join(parts)


def request_max_tokens(plan: ThinkingPlan, *, configured: int, explicit: bool, answer_budget=None) -> int:
    """实发上限：思考关闭时沿用现有取法；思考开着时放到封顶值（连接显式上限仍优先）。"""

    configured = int(configured)
    if explicit:
        return configured
    if plan.raises_output_limit:
        return plan.raised_max_tokens()
    if answer_budget is None:
        return configured
    return max(configured, int(answer_budget))


def call_context(plan: ThinkingPlan, *, answer_budget=None, max_tokens=None) -> dict:
    """调用日志的上下文（只有设置，不含任何内容）。"""

    return {
        "thinking": plan.log_label(),
        "effort": plan.effort,
        "source": plan.source,
        "entry": plan.entry,
        "answer_budget": answer_budget,
        "max_tokens": max_tokens,
    }


__all__ = [
    "CAPABILITIES",
    "CHAT",
    "MESSAGES",
    "PROVIDER_DEFAULT",
    "RESPONSES",
    "SCOPES",
    "SCOPE_ALL",
    "SCOPE_AUXILIARY",
    "SCOPE_OFF",
    "SOURCE_ADJUSTED",
    "SOURCE_EXPLICIT",
    "SOURCE_FAMILY_DEFAULT",
    "SOURCE_LEGACY",
    "SOURCE_TABLE",
    "SOURCE_UNKNOWN",
    "TABLE_SOURCES",
    "THINKING_ON_MAX_TOKENS_CAP",
    "THINKING_ON_TIMEOUT_SECONDS",
    "ThinkingCapability",
    "ThinkingPlan",
    "call_context",
    "current_scope",
    "legacy_chat_plan",
    "legacy_messages_plan",
    "legacy_responses_plan",
    "lookup",
    "model_version_suffix",
    "normalize_model_name",
    "plans",
    "prefix_matches",
    "request_max_tokens",
    "resolve_chat",
    "resolve_messages",
    "resolve_responses",
]
