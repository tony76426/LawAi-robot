"""Small offline regression baseline, NOT a legal reasoning accuracy score."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieval import load_records, rank_records  # noqa: E402

questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
records = load_records(ROOT)
passed = 0
failures = []
for row in questions:
    hits = rank_records(row["question"], records, limit=5)
    candidates = [hit["id"] for hit in hits]
    if row["expected_id"] is None:
        ok = not candidates
    else:
        ok = row["expected_id"] in candidates
    if ok:
        passed += 1
    else:
        failures.append({"question": row["question"], "expected": row["expected_id"],
                         "actual": candidates})
print(json.dumps({
    "test_cases": len(questions), "passed": passed,
    "recall_at_5_or_abstain": round(passed / len(questions), 4),
    "failures": failures, "warning": "FAQ-only fixture; not a legal accuracy benchmark.",
}, ensure_ascii=False, indent=2))
if failures:
    sys.exit(1)
