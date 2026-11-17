"""模型管理页的请求与响应（docs/模型管理页与评分思考修正方案.md 3.2）。"""

from __future__ import annotations

import json
import re
from datetime import date
from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from backend.app.services.llm import hosts

CapabilityProtocol = Literal["chat", "responses", "messages"]

HOST_FAMILIES = (hosts.VERTEX, hosts.GEMINI_API, hosts.ZHIPU, hosts.OPENROUTER, hosts.DASHSCOPE, hosts.DEEPSEEK,
                 hosts.MOONSHOT, hosts.OPENAI, hosts.ANTHROPIC, hosts.BEDROCK, hosts.OTHER)
#: 请求写法只收这几个顶层键：管理员不能借这里往请求里塞模型名、消息之类的字段。
REQUEST_KEYS = frozenset({"thinking", "reasoning_effort", "reasoning", "extra_body", "output_config"})
REQUEST_MAX_BYTES = 1024
REQUEST_MAX_DEPTH = 4

_PREFIX = re.compile(r"[a-z0-9][a-z0-9._:-]{0,119}")
_EFFORT = re.compile(r"[a-z][a-z0-9_-]{0,31}")


def _depth(value, level=1) -> int:
    if isinstance(value, dict):
        return max([level, *(_depth(item, level + 1) for item in value.values())])
    if isinstance(value, list):
        return max([level, *(_depth(item, level + 1) for item in value)])
    return level - 1


def validate_request_fields(value, *, name: str):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("%s 必须是 JSON 对象" % name)
    unknown = sorted(set(value) - REQUEST_KEYS)
    if unknown:
        raise ValueError("%s 只能包含 %s，不能包含 %s" % (name, "、".join(sorted(REQUEST_KEYS)), "、".join(unknown)))
    if len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > REQUEST_MAX_BYTES:
        raise ValueError("%s 不能超过 %d 字节" % (name, REQUEST_MAX_BYTES))
    if _depth(value) > REQUEST_MAX_DEPTH:
        raise ValueError("%s 的嵌套不能超过 %d 层" % (name, REQUEST_MAX_DEPTH))
    return value


class ModelCapabilityFields(BaseModel):
    can_disable: bool
    off_request: dict | None = None
    lowest_effort: str | None = Field(default=None, max_length=32)
    effort_request: dict | None = None
    max_output_tokens: int | None = Field(default=None, ge=256, le=1_000_000)
    enabled: bool = True
    note: str = Field(default="", max_length=500)
    source_url: str = Field(min_length=8, max_length=500)
    verified_on: date

    @field_validator("off_request")
    @classmethod
    def _off(cls, value):
        return validate_request_fields(value, name="关闭写法")

    @field_validator("effort_request")
    @classmethod
    def _effort_request(cls, value):
        return validate_request_fields(value, name="最低强度写法")

    @field_validator("lowest_effort")
    @classmethod
    def _effort(cls, value):
        value = (value or "").strip() or None
        if value is not None and not _EFFORT.fullmatch(value):
            raise ValueError("最低强度只能是小写字母、数字、下划线或短横线，例如 low、minimal")
        return value

    @field_validator("source_url")
    @classmethod
    def _source(cls, value):
        value = value.strip()
        if not value.startswith("https://"):
            raise ValueError("来源链接必须是 https:// 开头的厂商文档地址")
        return value

    @field_validator("note")
    @classmethod
    def _note(cls, value):
        return value.strip()

    @model_validator(mode="after")
    def _consistent(self):
        if self.can_disable and self.effort_request:
            raise ValueError("能关闭思考的模型不用填最低强度写法")
        if not self.can_disable and self.off_request:
            raise ValueError("关不掉思考的模型不用填关闭写法")
        return self


class ModelCapabilityCreate(ModelCapabilityFields):
    protocol: CapabilityProtocol
    host_family: str
    model_prefix: str = Field(min_length=1, max_length=120)

    @field_validator("host_family")
    @classmethod
    def _family(cls, value):
        if value not in HOST_FAMILIES:
            raise ValueError("主机族只能是 %s" % "、".join(HOST_FAMILIES))
        return value

    @field_validator("model_prefix")
    @classmethod
    def _prefix(cls, value):
        value = value.strip().casefold()
        if not _PREFIX.fullmatch(value):
            raise ValueError("模型名前缀只能是小写字母、数字和 . _ : -，例如 glm-5.3")
        return value


class ModelCapabilityUpdate(ModelCapabilityFields):
    revision: int = Field(ge=1)


class ModelCapabilityEntry(BaseModel):
    """列表里的一行：代码表或管理员记录。"""

    source: Literal["code", "admin"]
    id: str | None = None
    key: str
    protocol: str
    host_family: str
    model_prefix: str
    can_disable: bool
    off_request: dict | None = None
    lowest_effort: str | None = None
    effort_request: dict | None = None
    max_output_tokens: int | None = None
    enabled: bool = True
    note: str = ""
    sources: list[str] = Field(default_factory=list)
    verified_on: str
    #: 代码记录被同一键的启用管理员记录覆盖。
    overridden: bool = False
    #: 现在会命中这一条的启用连接数（全平台）。
    active_connections: int = 0
    revision: int | None = None
    updated_at: datetime | None = None


class ModelCapabilityList(BaseModel):
    items: list[ModelCapabilityEntry]
    host_families: list[str]
    #: 助手 Agent 评测达标的模型（只读：放行要跑真实评测，见方案 Q3）。
    agent_evaluated: list[str]
