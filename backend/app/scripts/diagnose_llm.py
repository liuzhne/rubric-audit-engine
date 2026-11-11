import json
import sys

import httpx

from backend.app.core.config import settings
from backend.app.services.llm.auth_headers import openai_auth_headers
from backend.app.services.llm.thinking_capabilities import CHAT
from backend.app.services.llm.thinking_capabilities import plans


def _thinking_config(payload):
    return ((payload.get("extra_body") or {}).get("google") or {}).get("thinking_config")


def main():
    if not settings.OPENAI_COMPATIBLE_API_KEY:
        print_json({"ok": False, "stage": "config", "error": "OPENAI_COMPATIBLE_API_KEY is not configured"})
        return 1

    url = "%s/chat/completions" % settings.OPENAI_COMPATIBLE_BASE_URL.rstrip("/")
    payload = {
        "model": settings.OPENAI_COMPATIBLE_MODEL,
        "messages": [
            {"role": "system", "content": "只返回一个 JSON 对象，不要返回 Markdown。"},
            {"role": "user", "content": '请严格返回 {"ok": true, "message": "pong"}'},
        ],
        "temperature": 0.2,
        "max_tokens": 200,
    }
    # 与 AI 任务同一份思考计划（模型能力表，docs/模型思考能力表方案.md）：能关的发各家的关闭写法，
    # 关不掉的发最低强度。
    plan, _scoring = plans(
        CHAT, url, settings.OPENAI_COMPATIBLE_MODEL, thinking_type=settings.OPENAI_COMPATIBLE_THINKING_TYPE,
    )
    payload.update(plan.fields())
    if settings.OPENAI_COMPATIBLE_RESPONSE_FORMAT_JSON:
        payload["response_format"] = {"type": "json_object"}

    try:
        response = httpx.post(
            url,
            headers={
                **openai_auth_headers(url, settings.OPENAI_COMPATIBLE_API_KEY),
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=settings.OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
        )
    except httpx.TransportError as exc:
        print_json(
            {
                "ok": False,
                "stage": "transport",
                "url": url,
                "model": settings.OPENAI_COMPATIBLE_MODEL,
                "thinking": payload.get("thinking"),
                "reasoning_effort": payload.get("reasoning_effort"),
                "thinking_config": _thinking_config(payload),
                "response_format": payload.get("response_format"),
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "hint": "请求未到达可解析的 HTTP 响应阶段，请优先检查代理、DNS、TLS 拦截或网络出口。",
            }
        )
        return 2

    body_preview = response.text[:1200]
    result = {
        "ok": response.is_success,
        "stage": "http",
        "url": url,
        "model": settings.OPENAI_COMPATIBLE_MODEL,
        "thinking": payload.get("thinking"),
        # Google 主机关闭思考不发 thinking：Gemini API 发 reasoning_effort，Vertex 发
        # extra_body.google.thinking_config，三者都报出来便于核对。
        "reasoning_effort": payload.get("reasoning_effort"),
        "thinking_config": _thinking_config(payload),
        "thinking_plan": {"on": plan.thinking_on, "effort": plan.effort, "source": plan.source, "entry": plan.entry},
        "response_format": payload.get("response_format"),
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type"),
        "body_preview": body_preview,
    }
    try:
        data = response.json()
    except ValueError:
        print_json(result)
        return 3

    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    result.update(
        {
            "choices_len": len(choices),
            "finish_reason": choice.get("finish_reason"),
            "choice_keys": sorted(choice.keys()),
            "message_keys": sorted(message.keys()),
            "content_chars": len(content or "") if isinstance(content, str) else None,
            "content_preview": str(content or "")[:500],
            "reasoning_content_chars": len(reasoning or "") if isinstance(reasoning, str) else None,
        }
    )
    print_json(result)
    return 0 if response.is_success and bool(content) else 4


def print_json(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
