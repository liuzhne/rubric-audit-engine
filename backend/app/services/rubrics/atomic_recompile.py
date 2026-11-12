"""Prepare a successor without flattening imported atomic rules into criteria."""
from copy import deepcopy
from decimal import Decimal, InvalidOperation

from sqlalchemy import select

from backend.app.db import models
from backend.app.services.rubric_import.deduction_caps import (
    ai_mutex_identity,
    normalize_ai_group_caps,
    suffixed_mutex_groups,
)
from backend.app.services.rubric_import import pipeline
from backend.app.services.rubrics import lifecycle
from backend.app.services.rubrics.review_workspace import content_token, number_text
from backend.app.services.scoring.core.policy import retarget_policy_total


def _fields(obj, names):
    return {name: deepcopy(getattr(obj, name)) for name in names.split()}


def prepare_atomic_recompile(session, version, command, edits):
    rules = session.scalars(select(models.AtomicRule).where(
        models.AtomicRule.rubric_version_id == version.id)).all()
    by_id = {rule.id: rule for rule in rules}
    if any(not isinstance(edit, dict) for edit in edits) or len(edits) != len(rules) or {edit.get("id") for edit in edits} != set(by_id):
        raise ValueError("必须完整保留当前草稿的每条原子规则，不允许通过评分项投影删除或合并")
    values = []
    expected = {}
    for edit in edits:
        rule = by_id[edit["id"]]
        if edit.get("content_token") != content_token(rule):
            raise ValueError("条款内容已变化，请刷新后重新核对")
        expected[rule.id] = edit["content_token"]
        changes = edit.get("changes", {})
        if not isinstance(changes, dict) or set(changes) - lifecycle._EDITABLE_RULE_FIELDS - {"levels"}:
            raise ValueError("原子规则包含不可编辑字段")
        try:
            normalized = lifecycle._normalize_rule_changes(rule.rule_code,
                {key: value for key, value in changes.items() if key != "levels"})
        except lifecycle.RubricLifecycleError as exc:
            raise ValueError(str(exc)) from exc
        value = _fields(rule, "rule_code name rule_text direction effect_type max_points repeat_policy cap_points judge_type checker_key checker_params evidence_policy positive_example negative_example boundary_example strictness applies_to mutex_group depends_on_rule_codes creation_method ai_origin ai_model scope")
        if "evidence_policy" in normalized:
            normalized["evidence_policy"] = lifecycle.keep_existing_judge_scope(
                rule.evidence_policy, normalized["evidence_policy"]
            )
        value.update(normalized)
        for key in ("max_points", "cap_points"):
            value[key] = number_text(value[key])
        value["criterion_code"] = rule.criterion.code
        value["source_rule_codes"] = [source.id for source in rule.source_rules]
        levels = changes.get("levels", [
            {"code": level.level_code, **_fields(level, "points descriptor positive_example negative_example display_order")}
            for level in rule.levels])
        if not isinstance(levels, list):
            raise ValueError("分档必须为列表")
        value["levels"] = []
        seen = set()
        for index, level in enumerate(levels):
            if not isinstance(level, dict):
                raise ValueError("分档必须为对象")
            code = level.get("code")
            if not isinstance(code, str) or not code.strip() or code in seen:
                raise ValueError("分档编号必须非空且唯一")
            seen.add(code)
            try:
                points = Decimal(str(level.get("points")))
            except InvalidOperation as exc:
                raise ValueError("分档分值必须为数值") from exc
            if not points.is_finite() or points < 0 or not isinstance(level.get("descriptor"), str) or not level["descriptor"].strip():
                raise ValueError("分档需要非负有限分值和说明")
            value["levels"].append({"level_code": code, "points": number_text(points),
                "descriptor": level["descriptor"], "positive_example": level.get("positive_example"),
                "negative_example": level.get("negative_example"), "display_order": index})
        if value["direction"] == "band" and not value["levels"]:
            raise ValueError("分档规则至少需要一个档位")
        values.append(value)

    artifacts = session.scalars(select(models.SourceArtifact).where(
        models.SourceArtifact.compilation_id == version.compilation_id)).all()
    source_values, template_values, artifact_values = [], [], []
    for artifact in artifacts:
        artifact_values.append({"token": artifact.id, **_fields(artifact,
            "artifact_type file_name file_hash file_size_bytes")})
        for source in artifact.source_rules:
            source_values.append({"source_rule_code": source.source_rule_code, "source_token": source.id, "artifact_token": artifact.id,
                **_fields(source, "sheet_name row_number cell_locator raw_text")})
        for item in artifact.template_items:
            template_values.append({"item_code": item.item_code, "item_token": item.id, "artifact_token": artifact.id,
                **_fields(item, "kind section_path raw_text normalized_constraint strictness source_locator source_hash"),
                "parse_confidence": number_text(item.parse_confidence)})
    links = [{"rule_code": rule.rule_code, "template_item_code": link.template_item_id,
              **_fields(link, "relationship_type match_method rationale"),
              "match_confidence": number_text(link.match_confidence), "review_status": "pending"}
             for rule in rules for link in rule.template_links]
    rubric = deepcopy(command["rubric"])
    for criterion in rubric["criteria"]:
        criterion["deduction_rules_structured"] = normalize_ai_group_caps(
            criterion.get("deduction_rules_structured") or [],
            criterion_code=criterion["code"], maximum=criterion["max_score"])
    rubric["global_policy"] = retarget_policy_total(
        rubric["global_policy"], total_score=rubric["total_score"]).to_mapping()
    prepared = pipeline._base_graph(command=command, source_kind="atomic_edit",
        rubric=rubric, artifacts=artifact_values, source_rules=source_values,
        template_items=template_values, criteria=rubric["criteria"], atomic_rules=values,
        template_links=links, blockers=[])
    graph = prepared.to_mapping()
    graph["expected_rule_tokens"] = expected
    return pipeline.PreparedRubricGraph.from_mapping(graph)


def _sync_ai_mutex_groups(existing, generated, command):
    """已有的 AI 细则改取本次编译规整后的互斥组（方案 §4.1(2)）。

    追加路径从库里原样取已有细则，只追加新编号；不同步的话，先“仅应用此条”写入的
    单档细则在其余档追加后仍游离在组外，存量的单档阻断和撞组也永远修不好。

    只改机器能产生的值：起草原名、空、原名加指纹后缀。老师在原子规则编辑里改成别的
    名字的保持不动。改了的细则内容签名会变，回到待确认——判分语义确实变了。
    """
    generated_by_code = {rule["rule_code"]: rule for rule in generated}
    drafted = {}
    for criterion in command["rubric"]["criteria"]:
        for index, row in enumerate(criterion.get("deduction_rules_structured") or [], 1):
            # 只认编译时会规整的行：带指纹和行键、来源却不是 ai_ 的行编译时原样保留，
            # 这里也不能把老师清空的互斥组改回去。
            if ai_mutex_identity(row) is None:
                continue
            # 编译总按位置编号；行上写了编号时，它指的是库里已有的那条细则。
            compiled_code = f"manual.{criterion['code'].lower()}.deduct.{index}.v1"
            drafted[row.get("rule_code") or compiled_code] = (row, compiled_code)
    for value in existing:
        row, compiled_code = drafted.get(value["rule_code"], (None, None))
        if row is None or compiled_code not in generated_by_code:
            continue
        original, fingerprint = ai_mutex_identity(row)
        if value.get("mutex_group") in {None, "", original, *suffixed_mutex_groups(original, fingerprint)}:
            value["mutex_group"] = generated_by_code[compiled_code].get("mutex_group")


def prepare_atomic_ai_append(session, version, command):
    """Append explicitly fingerprinted suggestions without replacing imported rules."""
    rules = session.scalars(select(models.AtomicRule).where(models.AtomicRule.rubric_version_id == version.id)).all()
    edits = [{"id": rule.id, "content_token": content_token(rule), "changes": {}} for rule in rules]
    graph = prepare_atomic_recompile(session, version, command, edits).to_mapping()
    generated = pipeline.prepare_manual_json_recompile(command=command).to_mapping()
    codes = {rule.rule_code for rule in rules}
    eligible = set()
    for criterion in command["rubric"]["criteria"]:
        for index, row in enumerate(criterion.get("deduction_rules_structured") or [], 1):
            if row.get("generation_fingerprint") and row.get("draft_row_key"):
                eligible.add(row.get("rule_code") or f"manual.{criterion['code'].lower()}.deduct.{index}.v1")
    additions = [rule for rule in generated["atomic_rules"] if rule["rule_code"] in eligible - codes]
    if not additions:
        raise ValueError("导入原子规则必须使用完整原子规则编辑，不能由评分项投影重新编译")
    _sync_ai_mutex_groups(graph["atomic_rules"], generated["atomic_rules"], command)
    graph["atomic_rules"].extend(additions)
    graph["artifacts"].extend(generated["artifacts"])
    graph["source_rules"].extend(generated["source_rules"])
    return pipeline.PreparedRubricGraph.from_mapping(graph)
