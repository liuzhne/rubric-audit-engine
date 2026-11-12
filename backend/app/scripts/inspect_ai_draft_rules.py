"""只读核对 AI 起草细则的互斥组与区间拆档（AI起草互斥组规整与区间拆档方案 §4.1(4)）。

用法（生产在 app 容器里跑）::

    python -m backend.app.scripts.inspect_ai_draft_rules <rubric_id>
    python -m backend.app.scripts.inspect_ai_draft_rules --all

看评分标准当前的执行草稿（已发布的看发布版本），列出三类情况：

- ``lone_mutex_groups``：互斥组里参与评分的细则只有 1 条。``blocks_publish`` 与发布校验
  （``mutex_definition_invalid``）对单条组的判断一致：只因排除剩 1 条、且这条不是分档细则的不挡。
- ``invalid_mutex_groups``：2 条以上、却跨评分项或含分档细则的组，发布校验同样报
  ``mutex_definition_invalid``。和上一类合起来覆盖这个错误码的全部情况。
- ``collided_mutex_groups``：同一个互斥组里的细则来自不止一次起草，不同问题被并成一组。
- ``single_tier_ranges``：AI 起草的一组细则引用了写成分值区间的原文，却只有一档。只核对
  引用评分项“扣分规则”段落的来源，原文单元（``docx:…``）记在 ``unchecked_refs``。

只读：查完即回滚、不写库。只输出编号和数字，不输出规则正文和原文。
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

from sqlalchemy import select

from backend.app.db import models


_DEDUCTION_REF_RE = re.compile(r"^/(?:criteria/[^/]+|criterion)/deduction_rules/(\d+)$")


def _projection_rows(criteria):
    """细则编号 → (评分项, 投影行)。编号公式与 ``pipeline`` 编译结构化扣分时相同。"""

    rows = {}
    for criterion in criteria:
        for index, row in enumerate(criterion.deduction_rules_structured or [], 1):
            if isinstance(row, dict):
                code = row.get("rule_code") or f"manual.{criterion.code.lower()}.deduct.{index}.v1"
                rows[code] = (criterion, row)
    return rows


def inspect_rubric(db, rubric_id: str) -> dict:
    """核对一份评分标准当前执行草稿的 AI 细则，返回只含编号和数字的报告。"""

    from backend.app.services.rubric_import.compiler import point_range
    from backend.app.services.rubrics.draft_graph import read_execution_draft
    from backend.app.services.rubrics.rule_status import participates

    try:
        draft = read_execution_draft(session=db, rubric_id=rubric_id)
    except ValueError:
        return {"rubric_id": rubric_id, "found": False, "next_step": "没有这个评分标准：确认 ID。"}
    version = ((draft.get("active_compilation") or {}).get("version") or {})
    if not version.get("id"):
        return {"rubric_id": rubric_id, "found": True, "version": None,
                "next_step": "没有唯一的当前执行草稿，无从核对。"}

    rules = db.scalars(
        select(models.AtomicRule)
        .where(models.AtomicRule.rubric_version_id == version["id"])
        .order_by(models.AtomicRule.rule_code)
    ).all()
    criteria = db.scalars(
        select(models.RubricCriterion).where(models.RubricCriterion.rubric_id == rubric_id)
    ).all()
    criterion_code = {item.id: item.code for item in criteria}
    rows = _projection_rows(criteria)
    active = [rule for rule in rules if participates(rule)]
    excluded_groups = {rule.mutex_group for rule in rules if not participates(rule) and rule.mutex_group}

    by_mutex = defaultdict(list)
    for rule in active:
        if rule.mutex_group:
            by_mutex[rule.mutex_group].append(rule)

    lone, invalid, collided = [], [], []
    for name, members in sorted(by_mutex.items()):
        codes = [rule.rule_code for rule in members]
        criterion_ids = {rule.criterion_id for rule in members}
        has_band = any(rule.direction == "band" for rule in members)
        if len(members) < 2:
            lone.append({
                "criterion_code": criterion_code.get(members[0].criterion_id),
                "mutex_group": name,
                "rule_codes": codes,
                # 发布校验对含分档细则的组不看排除：分档细则写了互斥组本身就是错的。
                "blocks_publish": has_band or name not in excluded_groups,
            })
        elif len(criterion_ids) != 1 or has_band:
            # 发布校验另外两种 mutex_definition_invalid：同组跨评分项、组里有分档细则。
            invalid.append({
                "criterion_codes": sorted(str(criterion_code.get(item)) for item in criterion_ids),
                "mutex_group": name,
                "rule_codes": codes,
                "cross_criterion": len(criterion_ids) != 1,
                "has_band": has_band,
            })
        fingerprints = {
            rows[rule.rule_code][1].get("generation_fingerprint")
            for rule in members
            if rule.rule_code in rows and rows[rule.rule_code][1].get("generation_fingerprint")
        }
        if len(fingerprints) > 1:
            collided.append({
                "criterion_code": criterion_code.get(members[0].criterion_id),
                "mutex_group": name,
                "rule_codes": codes,
                "generations": len(fingerprints),
            })

    # 同一次起草的同一组：草稿行键是“评分项::组编号::序号”。
    drafted = defaultdict(list)
    for rule in active:
        criterion, row = rows.get(rule.rule_code, (None, None))
        key = str((row or {}).get("draft_row_key") or "").split("::")
        fingerprint = (row or {}).get("generation_fingerprint")
        if criterion is not None and len(key) == 3 and fingerprint:
            drafted[(criterion.code, key[1], fingerprint)].append((criterion, row, rule.rule_code))
    single_tier = []
    for (code, group, fingerprint), members in sorted(drafted.items()):
        criterion = members[0][0]
        segments = list(criterion.deduction_rules or [])
        ranged, unchecked = set(), set()
        for _, row, _ in members:
            for ref in row.get("source_refs") or []:
                match = _DEDUCTION_REF_RE.match(str(ref))
                if match is None:
                    if not str(ref).startswith("/criterion"):
                        unchecked.add(str(ref))
                    continue
                index = int(match.group(1))
                if index < len(segments) and point_range(segments[index]) is not None:
                    ranged.add(index)
        if ranged and len(members) < 2:
            single_tier.append({
                "criterion_code": code,
                "draft_group": group,
                "fingerprint_prefix": fingerprint[:8],
                "rule_codes": [rule_code for _, _, rule_code in members],
                "range_segments": sorted(ranged),
                "unchecked_refs": len(unchecked),
            })

    return {
        "rubric_id": rubric_id,
        "found": True,
        "version": {"id": version["id"], "version": version.get("version")},
        "counts": {
            "active_rules": len(active),
            "ai_rules": sum(1 for rule in active if rule.ai_origin),
            "lone_mutex_groups": len(lone),
            "lone_blocking_publish": sum(1 for item in lone if item["blocks_publish"]),
            "invalid_mutex_groups": len(invalid),
            "collided_mutex_groups": len(collided),
            "single_tier_ranges": len(single_tier),
        },
        "lone_mutex_groups": lone,
        "invalid_mutex_groups": invalid,
        "collided_mutex_groups": collided,
        "single_tier_ranges": single_tier,
    }


def inspect_all(db) -> dict:
    """逐份核对全部评分标准，只列出有问题的那些，并给出各类问题的总数。"""

    ids = db.scalars(select(models.Rubric.id).order_by(models.Rubric.created_at, models.Rubric.id)).all()
    reports = [inspect_rubric(db, rubric_id) for rubric_id in ids]
    totals = defaultdict(int)
    for report in reports:
        for key, value in (report.get("counts") or {}).items():
            totals[key] += value
    # 只列出有问题的评分标准，避免把整库的编号都打出来。
    flagged = [report for report in reports if any(
        (report.get("counts") or {}).get(key)
        for key in ("lone_mutex_groups", "invalid_mutex_groups", "collided_mutex_groups", "single_tier_ranges"))]
    return {"rubrics": len(reports), "totals": dict(totals), "flagged": flagged}


def main(argv=None) -> int:
    """命令行入口：打印 JSON 报告；找不到评分标准时返回 1，用法不对返回 2。"""

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("用法：python -m backend.app.scripts.inspect_ai_draft_rules <rubric_id>|--all", file=sys.stderr)
        return 2

    from backend.app.db.session import SessionLocal

    db = SessionLocal()
    try:
        report = inspect_all(db) if args[0] == "--all" else inspect_rubric(db, args[0])
    finally:
        # 只读：不管查到什么都回滚。生产连的是事务池，不能用会话级 SET 设只读。
        db.rollback()
        db.close()
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report.get("found", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
