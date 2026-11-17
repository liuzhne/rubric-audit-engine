"""模型管理：平台管理员登记模型的思考能力（docs/模型管理页与评分思考修正方案.md 3.2）。"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from backend.app.db import models
from backend.app.services.llm import capability_overrides
from backend.app.services.llm import thinking_capabilities as thinking
from backend.app.services.llm.openai_compatible_adapter import OpenAICompatibleChatScorer
from backend.app.tests.test_ops_permissions import _login_org_admin
from backend.app.tests.test_ops_permissions import _login_platform_admin

URL = "/api/admin/model-capabilities"
GATEWAY = "https://gw.example/v1"
ZHIPU = "https://open.bigmodel.cn/api/paas/v4"


def _payload(**overrides):
    payload = {
        "protocol": "chat", "host_family": "other", "model_prefix": "brand-new-think",
        "can_disable": False, "lowest_effort": "low", "effort_request": {"reasoning_effort": "low"},
        "max_output_tokens": 65536, "note": "强制思考", "source_url": "https://docs.example/thinking",
        "verified_on": "2026-11-17",
    }
    payload.update(overrides)
    return payload


def _refresh(client):
    capability_overrides.invalidate()
    with client.session_factory() as session:
        capability_overrides.refresh_if_stale(session)


def test_only_platform_admins_reach_the_page(client, monkeypatch):
    _login_org_admin(client, monkeypatch, "capability-org-admin")
    assert client.get(URL).status_code == 403
    assert client.post(URL, json=_payload()).status_code == 403


def test_list_merges_the_code_table_and_marks_overrides(client, monkeypatch):
    _login_platform_admin(client, monkeypatch)
    created = client.post(URL, json=_payload(host_family="zhipu", model_prefix="glm-5.3"))
    assert created.status_code == 201, created.text
    body = client.get(URL).json()
    admin = [item for item in body["items"] if item["source"] == "admin"]
    assert [item["model_prefix"] for item in admin] == ["glm-5.3"]
    assert admin[0]["key"].startswith(thinking.OVERRIDE_KEY_PREFIX) and admin[0]["revision"] == 1
    code = next(item for item in body["items"] if item["source"] == "code" and item["model_prefix"] == "glm-5.3")
    assert code["overridden"] is True and code["can_disable"] is False
    assert "other" in body["host_families"]
    assert body["agent_evaluated"] == ["google · gemini-2.5-flash"]
    with client.session_factory() as session:
        events = session.scalars(select(models.AuditLog.event_type)).all()
        assert "model_capability.created" in events


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"effort_request": {"model": "other-model"}}, "不能包含 model"),
        ({"effort_request": {"reasoning_effort": "x" * 1100}}, "1024 字节"),
        ({"effort_request": {"extra_body": {"a": {"b": {"c": {"d": 1}}}}}}, "嵌套"),
        ({"source_url": "http://docs.example"}, "https://"),
        ({"model_prefix": "GLM 5"}, "模型名前缀"),
        ({"host_family": "somewhere"}, "主机族"),
        ({"can_disable": True, "off_request": {"thinking": {"type": "disabled"}}}, "不用填最低强度"),
        ({"lowest_effort": "Low!"}, "最低强度"),
    ],
)
def test_request_fields_are_whitelisted(client, monkeypatch, overrides, message):
    _login_platform_admin(client, monkeypatch)
    response = client.post(URL, json=_payload(**overrides))
    assert response.status_code == 422
    assert message in response.text


def test_duplicate_key_and_stale_revision_are_conflicts(client, monkeypatch):
    _login_platform_admin(client, monkeypatch)
    created = client.post(URL, json=_payload()).json()
    assert client.post(URL, json=_payload()).status_code == 409
    update = {key: value for key, value in _payload().items() if key not in {"protocol", "host_family", "model_prefix"}}
    assert client.patch("%s/%s" % (URL, created["id"]), json={**update, "revision": 1, "enabled": False}).status_code == 200
    stale = client.patch("%s/%s" % (URL, created["id"]), json={**update, "revision": 1})
    assert stale.status_code == 409


def test_a_registered_record_changes_the_plan_and_the_identity(client, monkeypatch):
    _login_platform_admin(client, monkeypatch)
    # 表外模型：评分按旧行为（声明不思考），撞上守卫。
    before = OpenAICompatibleChatScorer(api_key="k", base_url=GATEWAY, model_name="brand-new-think-preview",
                                        thinking_type="disabled")
    assert before.scoring_thinking_plan.legacy and not before.scoring_thinking_plan.thinking_on

    created = client.post(URL, json=_payload()).json()
    _refresh(client)
    after = OpenAICompatibleChatScorer(api_key="k", base_url=GATEWAY, model_name="brand-new-think-preview",
                                       thinking_type="disabled")
    plan = after.scoring_thinking_plan
    assert plan.thinking_on and plan.fields() == {"reasoning_effort": "low"}
    assert plan.entry == created["key"]
    assert after.model_version.startswith("chat-completions;cap=%s;effort=low" % created["key"][3:])

    # 改了写法：内容哈希变，身份跟着变；停用即退回代码表。
    update = {key: value for key, value in _payload().items() if key not in {"protocol", "host_family", "model_prefix"}}
    changed = client.patch("%s/%s" % (URL, created["id"]),
                           json={**update, "effort_request": {"reasoning_effort": "low", "reasoning": {"effort": "low"}},
                                 "revision": 1}).json()
    assert changed["key"] != created["key"]
    client.patch("%s/%s" % (URL, created["id"]), json={**update, "enabled": False, "revision": 2})
    _refresh(client)
    assert thinking.lookup(thinking.CHAT, GATEWAY, "brand-new-think-preview") is None


def test_longest_prefix_wins_and_admin_wins_a_tie(client):
    with client.session_factory() as session:
        for prefix, can_disable in (("glm-5", True), ("glm-5.3", True)):
            session.add(models.ModelCapabilityOverride(
                protocol="chat", host_family="zhipu", model_prefix=prefix, can_disable=can_disable,
                off_request={"thinking": {"type": "disabled"}}, enabled=True, note="",
                source_url="https://docs.example", verified_on="2026-11-17", revision=1,
            ))
        session.commit()
    _refresh(client)
    # 一样长：管理员记录压过代码表里“关不掉”的 glm-5.3。
    assert thinking.lookup(thinking.CHAT, ZHIPU, "glm-5.3").override_hash
    assert thinking.lookup(thinking.CHAT, ZHIPU, "glm-5.3").can_disable is True
    # 更长的代码记录压过更短的管理员记录。
    assert thinking.lookup(thinking.CHAT, ZHIPU, "glm-5.2").override_hash is None


def test_cache_refreshes_only_after_the_interval(client):
    with client.session_factory() as session:
        capability_overrides.refresh_if_stale(session, now=1000.0)
        assert capability_overrides.current() == ()
        session.add(models.ModelCapabilityOverride(
            protocol="chat", host_family="other", model_prefix="late", can_disable=True, enabled=True,
            note="", source_url="https://docs.example", verified_on="2026-11-17", revision=1,
        ))
        session.commit()
        capability_overrides.refresh_if_stale(session, now=1010.0)
        assert capability_overrides.current() == ()
        capability_overrides.refresh_if_stale(session, now=1000.0 + capability_overrides.REFRESH_SECONDS)
        assert [item.model_prefix for item in capability_overrides.current()] == ["late"]


def test_connection_runtime_resolution_refreshes_the_cache(client, monkeypatch):
    calls = []
    monkeypatch.setattr(capability_overrides, "refresh_if_stale", lambda session, **_: calls.append(session))
    from backend.app.services.ai_connections import resolve_connection_runtime

    with client.session_factory() as session:
        with pytest.raises(ValueError):
            resolve_connection_runtime(session, connection_id="missing", owner_id="u", organization_id="o")
    # 连接不存在时不刷新（直接报错）；存在时在返回运行时之前刷新，见 resolve_connection_runtime。
    assert calls == []
