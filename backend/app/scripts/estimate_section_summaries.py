"""只读估算：一份论文快照做分节摘要要调用几次模型、发多少字（docs/分节摘要定位方案.md）。

用法::

    python -m backend.app.scripts.estimate_section_summaries <document_snapshot_id>
    python -m backend.app.scripts.estimate_section_summaries <document_snapshot_id> --json

只读数据库，不调模型、不写任何东西。输出只有节点标题与计数，不含论文正文；
标题本身仍来自论文，真实论文的输出不要提交到仓库。
"""

from __future__ import annotations

import argparse
import json
import sys

from backend.app.core.config import settings
from backend.app.services.scoring.retrieval import section_summary


def report(units, *, char_budget) -> dict:
    nodes, _batches, _truncated = section_summary.summary_batches(units, char_budget=char_budget)
    return {
        **section_summary.estimate(units, char_budget=char_budget),
        "char_budget": char_budget,
        "outline": [
            {"id": node["id"], "title": node["title"], "level": node["level"], "chars": node["chars"]}
            for node in nodes
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("document_snapshot_id")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from backend.app.db.models import DocumentSnapshot
    from backend.app.db.session import SessionLocal

    db = SessionLocal()
    try:
        snapshot = db.get(DocumentSnapshot, args.document_snapshot_id)
        if snapshot is None:
            print("找不到论文快照：%s" % args.document_snapshot_id, file=sys.stderr)
            return 1
        units = list((snapshot.snapshot_payload or {}).get("evidence_units") or ())
    finally:
        db.close()
    result = report(units, char_budget=settings.SCORING_SECTION_SUMMARY_CHAR_BUDGET)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("节点 %d 个，模型调用 %d 次，发给模型约 %d 字（每次上限 %d 字）" % (
        result["nodes"], result["calls"], result["input_chars"], result["char_budget"]))
    if result["truncated_nodes"]:
        print("正文超过单次上限、只取开头与结尾的节点：%s" % "、".join(result["truncated_nodes"]))
    for node in result["outline"]:
        print("%s%s %s（%d 字）" % ("  " * (node["level"] - 1), node["id"], node["title"], node["chars"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
