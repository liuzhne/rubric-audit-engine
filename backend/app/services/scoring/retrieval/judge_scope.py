"""判定范围：一条语义规则该看论文的哪一部分（docs/评分细则适用范围方案.md 第 8 节）。

范围只来自细则的适用范围：``atomic-rule-snapshot@5`` 的 ``scope``，由模型在起草或范围识别时
给出，人不指定；没有范围按全文。旧的按 ``applies_to``、人工判定范围与规则文字推断的做法删除。
论文段落属于哪个章节角色由 ``section_roles`` 决定。范围只决定证据怎么取，不出分；
推断规则的版本号 ``JUDGE_SCOPE_VERSION`` 进入证据选择身份。
"""

from __future__ import annotations

from collections.abc import Mapping

from backend.app.services.scoring.retrieval.section_roles import RoleResolver


JUDGE_SCOPE_VERSION = "judge-scope@2"

SECTION = "section"
CROSS_SECTION = "cross_section"
DOCUMENT = "document"
SCOPED = (SECTION, CROSS_SECTION)
# 一章或几章对照时最多给多少段：整章放得下就全部给出（coverage=complete），
# 超过这个数或放不进预算才在章内排序截取（partial）。
MAX_SCOPED_UNITS = 60


def rule_scope(rule) -> dict:
    """``{"scope": ..., "targets": [...]}``；targets 是章节角色名。"""

    if rule.get("judge_type") == "deterministic":
        return {"scope": DOCUMENT, "targets": []}
    scope = rule.get("scope")
    if not isinstance(scope, Mapping) or scope.get("type") not in SCOPED:
        return {"scope": DOCUMENT, "targets": []}
    targets = [str(item) for item in scope.get("sections") or () if str(item)]
    if not targets:
        return {"scope": DOCUMENT, "targets": []}
    return {"scope": scope["type"], "targets": targets}


infer_judge_scope = rule_scope


def shared_rule(rules) -> dict:
    """共用一次证据选择的规则里编号最小的那条：互斥组各档、共用选择的评分项规则都按它。"""

    return min(rules, key=lambda item: item["rule_code"])


def select_scoped_units(units, scope, *, rank, top_k, resolver: RoleResolver):
    """只在目标章节里选段；一章都找不到时按全文选（执行器随后不调用模型，转人工）。

    ``rank`` 给一组单元排序（BM25 + 章节多样性）。多章对照时各章交替取，保证每章先有代表。
    """

    limit = max(int(top_k), MAX_SCOPED_UNITS)
    pools = []
    for target in scope["targets"]:
        pool = [unit for unit in units if resolver.unit_role(unit) == target]
        if pool:
            pools.append(rank(pool))
    if not pools:
        return rank(list(units))[:top_k]
    selected = []
    seen = set()
    depth = 0
    while len(selected) < limit:
        added = False
        for ranked in pools:
            if depth < len(ranked):
                unit = ranked[depth]
                if unit["evidence_unit_id"] not in seen:
                    seen.add(unit["evidence_unit_id"])
                    selected.append(unit)
                    if len(selected) >= limit:
                        break
                added = True
        if not added:
            break
        depth += 1
    return selected


def _coverage_by_target(value, targets, resolver):
    """每个目标章节在论文里有哪些一级标题、给出的证据是否是整章。

    依据证据选择身份的 ``section_coverage``：开了范围时它列出全文每个标题路径的可选与已选段数。
    """

    entries = ((value.get("evidence_selection_identity") or {}).get("section_coverage")) or []
    result = {target: {"headings": [], "complete": True, "available": 0} for target in targets}
    for entry in entries:
        path = list(entry.get("section_path") or [])
        role = resolver.unit_role({"section_path": path})
        if role not in result or not entry.get("available_unit_count"):
            continue
        item = result[role]
        item["available"] += int(entry["available_unit_count"])
        if path and path[0] not in item["headings"]:
            item["headings"].append(path[0])
        if not entry.get("complete"):
            item["complete"] = False
    return result


def missing_targets(value) -> list[str]:
    """目标章节里在论文中一段都没有的（第 8.3 节）；全文范围或没开范围时为空。"""

    if value.get("coverage_mode") not in SCOPED:
        return []
    scope = rule_scope(value["atomic_rule_snapshot"])
    if scope["scope"] not in SCOPED:
        return []
    coverage = _coverage_by_target(value, scope["targets"], RoleResolver.from_envelope(value))
    return [target for target, item in coverage.items() if not item["available"]]


def scope_view(rules, value) -> dict:
    """告诉模型的范围：类型、每个目标章节对应的论文标题，以及给出的是整章还是部分。"""

    scope = rule_scope(shared_rule(rules))
    if value.get("coverage_mode") not in SCOPED or scope["scope"] not in SCOPED:
        return {"type": DOCUMENT}
    coverage = _coverage_by_target(value, scope["targets"], RoleResolver.from_envelope(value))
    return {
        "type": scope["scope"],
        "sections": [
            {
                "role": target,
                "headings": coverage[target]["headings"],
                "coverage": (
                    "missing" if not coverage[target]["available"]
                    else "complete" if coverage[target]["complete"] else "partial"
                ),
            }
            for target in scope["targets"]
        ],
    }


__all__ = [
    "CROSS_SECTION",
    "DOCUMENT",
    "JUDGE_SCOPE_VERSION",
    "MAX_SCOPED_UNITS",
    "SECTION",
    "infer_judge_scope",
    "missing_targets",
    "rule_scope",
    "scope_view",
    "select_scoped_units",
    "shared_rule",
]
