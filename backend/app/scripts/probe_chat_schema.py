"""对一个 Chat Completions 端点实测：它接不接受评分用的严格 Schema，枚举多大开始被拒绝。

方案 ``docs/Chat评分请求严格Schema方案.md`` 第 4.5 节第 2 步。**会调用真实模型，需要维护者授权。**
默认只打印将要发送的请求（不联网）；加 ``--send`` 才发送。请求内容全部是合成的，不含任何材料。

    CHAT_SCHEMA_PROBE_BASE_URL=... CHAT_SCHEMA_PROBE_MODEL=... CHAT_SCHEMA_PROBE_API_KEY=... \\
        python -m backend.app.scripts.probe_chat_schema --send

每个枚举规模发 2 次请求（原样 Schema、主机族方言各 1 次），外加「提示词要求违反 Schema」的
请求（原样与方言各 1 次），看 ``strict`` 是否真的被执行。总次数不超过 ``--max-calls``（默认 16）。只输出状态码、厂商
错误类型、回答是否合规与 token 用量，不输出 Key，也不输出厂商返回的正文。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import httpx

from backend.app.core.config import settings
from backend.app.services.llm import schema_dialects
from backend.app.services.llm.auth_headers import openai_auth_headers
from backend.app.services.llm.thinking_params import thinking_request_fields


STATUSES = ("triggered", "not_triggered")
LEVELS = ("A", "B", "C")


def rule_schema(alias_count: int) -> dict:
    """与 ``core_view.build_core_request`` 同形状的 Schema，可引用的证据别名有 ``alias_count`` 个。"""

    aliases = ["e%d" % index for index in range(1, alias_count + 1)]
    evidence = {
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "evidence_ref", "quote"],
        "properties": {
            "type": {"type": "string", "enum": ["source_quote"]},
            "evidence_ref": {"type": "string", "enum": aliases},
            "quote": {"type": "string", "minLength": 1},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "rule_code", "status", "level_code", "occurrences"],
        "properties": {
            "schema_version": {"type": "string", "enum": ["semantic-rule-response@2"]},
            "rule_code": {"type": "string", "enum": ["probe.v1"]},
            "status": {"type": "string", "enum": list(STATUSES)},
            "level_code": {"type": ["string", "null"], "enum": list(LEVELS) + [None]},
            "occurrences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["finding_code", "evidence"],
                    "properties": {
                        "finding_code": {"type": "string", "enum": ["F1", "F2"]},
                        "evidence": {"type": "array", "minItems": 1, "items": evidence},
                    },
                },
            },
        },
    }


_FOLLOW = (
    "Return one JSON object: schema_version is semantic-rule-response@2, rule_code is probe.v1, "
    "status is not_triggered, level_code is null, occurrences is an empty array."
)
_VIOLATE = (
    "Return one JSON object: schema_version is semantic-rule-response@2, rule_code is probe.v1, "
    "status is maybe, level_code is Z, occurrences is an empty array, and add a field note set to x."
)


def build_payload(model: str, schema: dict, *, violate: bool = False, max_tokens: int = 1024,
                  base_url: str = "") -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": _VIOLATE if violate else _FOLLOW}],
        "temperature": 0,
        "max_tokens": max_tokens,
        # 与评分一样关掉思考：否则思考会先吃掉输出上限，回答被截断，看不出 Schema 有没有被执行。
        **thinking_request_fields(base_url, model, "none"),
        # schema 为 None 是对照：同一句提示词、不带任何约束，看模型自己会不会照着违反。
        **({} if schema is None else {"response_format": {
            "type": "json_schema",
            "json_schema": {"name": "semantic_rule_response", "strict": True, "schema": schema},
        }}),
    }


def conforms(answer) -> bool:
    """回答是否满足原始 Schema 里厂商应当执行的那几条（字段齐全、没有多余字段、枚举内）。"""

    if not isinstance(answer, dict):
        return False
    if set(answer) != {"schema_version", "rule_code", "status", "level_code", "occurrences"}:
        return False
    return (
        answer["schema_version"] == "semantic-rule-response@2"
        and answer["rule_code"] == "probe.v1"
        and answer["status"] in STATUSES
        and (answer["level_code"] is None or answer["level_code"] in LEVELS)
        and isinstance(answer["occurrences"], list)
    )


def plan(sizes, dialect: str):
    """``(名称, Schema, 是否要求违反)`` 的列表。"""

    cases = []
    for size in sizes:
        schema = rule_schema(size)
        cases.append(("aliases=%d raw" % size, schema, False))
        if dialect != schema_dialects.PLAIN:
            cases.append(("aliases=%d %s" % (size, dialect), schema_dialects.apply(dialect, schema), False))
    cases.extend(violate_cases(min(sizes), dialect))
    return cases


def violate_cases(size: int, dialect: str):
    smallest = rule_schema(size)
    cases = [("violate raw", smallest, True)]
    if dialect != schema_dialects.PLAIN:
        cases.append(("violate %s" % dialect, schema_dialects.apply(dialect, smallest), True))
    # 对照：不带 Schema。这一行 conforms=False，上面两行的 True 才说明是 Schema 在起作用。
    cases.append(("violate no-schema", None, True))
    # 没有可引用证据时评分发的是另一种写法（必须为空的数组），单独确认它被接受。
    empty = no_evidence_schema()
    cases.append(("no-evidence raw", empty, False))
    if dialect != schema_dialects.PLAIN:
        cases.append(("no-evidence %s" % dialect, schema_dialects.apply(dialect, empty), False))
    return cases


def no_evidence_schema() -> dict:
    """``core_view._occurrences_schema`` 在没有可引用证据时的写法。"""

    schema = rule_schema(1)
    schema["properties"]["occurrences"] = {
        "type": "array",
        "maxItems": 0,
        "items": {"type": "object", "additionalProperties": False, "properties": {}, "required": []},
    }
    return schema


def _completed(result) -> bool:
    return bool(result) and result["http"] == 200 and result["finish"] != "length" and result["conforms"] is not None


def enforcement_conclusion(outcomes: dict) -> str:
    """只有对照成立时才下结论：不带 Schema 时模型照着违反了，带 Schema 时回答合规。"""

    control = outcomes.get("violate no-schema")
    constrained = {
        name: result for name, result in outcomes.items()
        if name.startswith("violate ") and name != "violate no-schema"
    }
    if not constrained:
        return "结论：这次没有发「要求违反 Schema」的请求，不判断 Schema 是否被执行。"
    if not _completed(control):
        return "结论：没有结论。对照（violate no-schema）没有得到完整的回答，无法判断合规是不是 Schema 的作用。"
    if control["conforms"]:
        return "结论：没有结论。不带 Schema 时模型也没有照着违反，带 Schema 的合规说明不了问题。"
    lines = []
    for name, result in sorted(constrained.items()):
        if not _completed(result):
            lines.append("%s：没有结论（请求失败或被截断）" % name)
        elif result["conforms"]:
            lines.append("%s：Schema 被执行（对照违反了，带 Schema 时合规）" % name)
        else:
            lines.append("%s：Schema 没有被执行（只接受了参数）" % name)
    return "结论：" + "；".join(lines)


def _error_type(response) -> str:
    try:
        data = response.json()
    except ValueError:
        return "-"
    if isinstance(data, list) and data:
        data = data[0]
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return "-"
    return str(error.get("status") or error.get("type") or error.get("code") or "-")[:64]


def _send(client, endpoint, headers, payload) -> dict:
    response = client.post(endpoint, json=payload, headers=headers)
    result = {"http": response.status_code, "error": "-", "conforms": None, "finish": "-", "input": 0, "output": 0}
    if response.status_code != 200:
        result["error"] = _error_type(response)
        return result
    data = response.json()
    usage = data.get("usage") or {}
    result["input"] = int(usage.get("prompt_tokens") or 0)
    result["output"] = int(usage.get("completion_tokens") or 0)
    choice = (data.get("choices") or [{}])[0]
    result["finish"] = str(choice.get("finish_reason") or "-")
    try:
        answer = json.loads((choice.get("message") or {}).get("content") or "")
    except ValueError:
        answer = None
    # 被截断的回答说明不了问题：既不算合规，也不算违反。
    result["conforms"] = None if result["finish"] == "length" else conforms(answer)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--sizes", default="8,32,64,128,256", help="要试的证据别名个数，逗号分隔")
    parser.add_argument("--max-calls", type=int, default=16, help="最多发送多少次请求")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument(
        "--enum-limit", type=int, default=2000,
        help="方言保留枚举的上限。默认放到最大，这样方言那一列测的是厂商的上限，而不是我们自己的",
    )
    parser.add_argument("--only-violate", action="store_true", help="只发「要求违反 Schema」的请求")
    parser.add_argument("--send", action="store_true", help="真的发送（会调用真实模型）")
    args = parser.parse_args(argv)

    base_url = os.environ.get("CHAT_SCHEMA_PROBE_BASE_URL", "").rstrip("/")
    model = os.environ.get("CHAT_SCHEMA_PROBE_MODEL", "")
    api_key = os.environ.get("CHAT_SCHEMA_PROBE_API_KEY", "")
    if not base_url or not model:
        print("需要 CHAT_SCHEMA_PROBE_BASE_URL 与 CHAT_SCHEMA_PROBE_MODEL", file=sys.stderr)
        return 2
    sizes = sorted({int(item) for item in args.sizes.split(",") if item.strip()})
    settings.SCORING_CHAT_SCHEMA_ENUM_LIMIT = args.enum_limit
    dialect = schema_dialects.chat_dialect(base_url)
    cases = violate_cases(min(sizes), dialect) if args.only_violate else plan(sizes, dialect)
    if len(cases) > args.max_calls:
        print("计划 %d 次请求，超过 --max-calls=%d；请减少 --sizes" % (len(cases), args.max_calls), file=sys.stderr)
        return 2

    print("dialect=%s enum_limit=%d model=%s calls=%d" % (dialect, schema_dialects.enum_limit(), model, len(cases)))
    if not args.send:
        for name, schema, violate in cases:
            size = len(json.dumps(build_payload(model, schema, violate=violate, base_url=base_url), ensure_ascii=False))
            print("  [不发送] %-24s request_bytes=%d" % (name, size))
        print("加 --send 才会发送。")
        return 0
    if not api_key:
        print("需要 CHAT_SCHEMA_PROBE_API_KEY", file=sys.stderr)
        return 2

    endpoint = base_url + "/chat/completions"
    headers = openai_auth_headers(base_url, api_key)
    total_input = total_output = 0
    outcomes = {}
    with httpx.Client(timeout=args.timeout, follow_redirects=False) as client:
        for name, schema, violate in cases:
            try:
                result = _send(client, endpoint, headers, build_payload(model, schema, violate=violate, base_url=base_url))
            except httpx.HTTPError as exc:
                print("  %-24s transport_error=%s" % (name, type(exc).__name__))
                continue
            outcomes[name] = result
            total_input += result["input"]
            total_output += result["output"]
            print("  %-24s http=%s error=%s finish=%s conforms=%s input=%d output=%d" % (
                name, result["http"], result["error"], result["finish"], result["conforms"],
                result["input"], result["output"],
            ))
    print("total input_tokens=%d output_tokens=%d" % (total_input, total_output))
    print(enforcement_conclusion(outcomes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
