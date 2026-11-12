"""评分时应用适用范围的三个判断，由 Profile 注入 Core 规则执行器（docs/评分细则适用范围方案.md 第 8 节）。

Core 包不依赖检索与词表模块（``test_m2_import_boundary``），所以执行器只认 Profile 上的
``rule_scope_policy``：

- ``section_role_map``：一篇论文算一次章节角色归类（词表 + 模型归类对不上的一级标题）；
- ``missing_sections``：目标章节在论文里一段都没有 → 执行器不调用模型、转人工；
- ``citations_cover_targets``：多章对照判 triggered 时引用是否覆盖至少两个目标章节。
"""

from __future__ import annotations

from collections.abc import Mapping

from backend.app.services.scoring.retrieval.judge_scope import CROSS_SECTION
from backend.app.services.scoring.retrieval.judge_scope import missing_targets
from backend.app.services.scoring.retrieval.judge_scope import rule_scope
from backend.app.services.scoring.retrieval.section_roles import RoleResolver
from backend.app.services.scoring.retrieval.section_roles import section_role_map


def _mapping(envelope):
    return envelope.to_mapping() if hasattr(envelope, "to_mapping") else envelope


def _scoped(rule) -> bool:
    scope = rule.get("scope")
    return isinstance(scope, Mapping) and scope.get("type") in {"section", "cross_section"}


class RuleScopePolicy:
    """适用范围的判断；无状态，所有 Profile 共用一个实例。"""

    def section_role_map(self, *, value, rules, llm_runtime):
        """有语义规则要看章节（规则快照 @5 带 scope）时返回本篇的归类记录，否则返回 None。"""

        if not any(rule.get("judge_type") == "semantic" and _scoped(rule) for rule in rules):
            return None
        profile_key = value["runtime_identity"]["profile_key"]
        classify_runtime = getattr(llm_runtime, "classify_section_headings", None)
        classify = None
        if callable(classify_runtime):
            def classify(headings):
                return classify_runtime(headings=headings, profile_key=profile_key)
        return section_role_map(value["document"]["evidence_units"], profile_key, classify)

    def missing_sections(self, envelope) -> list[str]:
        return missing_targets(_mapping(envelope))

    def citations_cover_targets(self, decision, envelope, rule) -> bool:
        """不是多章对照、或引用覆盖了至少两个目标章节时为 True。"""

        value = _mapping(envelope)
        if value.get("coverage_mode") != CROSS_SECTION:
            return True
        targets = set(rule_scope(rule)["targets"])
        resolver = RoleResolver.from_envelope(value)
        units = {unit["evidence_unit_id"]: unit for unit in value.get("evidence_units") or ()}
        cited = {
            resolver.unit_role(units[ref["evidence_unit_id"]])
            for ref in decision.get("evidence_refs") or ()
            if ref.get("evidence_unit_id") in units
        }
        return len(cited & targets) >= 2


RULE_SCOPE_POLICY = RuleScopePolicy()

__all__ = ["RULE_SCOPE_POLICY", "RuleScopePolicy"]
