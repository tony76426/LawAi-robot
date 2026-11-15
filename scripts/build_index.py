"""Build a REAL OpenAI embedding index from FAQ questions and reviewed legal sources.

Offline and explicitly invoked only. Never runs on import or without a configured API key.
Usage: OPENAI_API_KEY=... python scripts/build_index.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieval import EMBEDDING_MODEL, fingerprint, load_records  # noqa: E402


def build_index(model: str, output: Path) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required to generate real embeddings")
    from openai import OpenAI
    records = load_records(ROOT)
    if not records:
        raise RuntimeError("No knowledge records to index")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60.0, max_retries=2)
    embeddings = {}
    dims = None
    for start in range(0, len(records), 40):
        group = records[start:start + 40]
        response = client.embeddings.create(
            model=model, input=[record.embedding_text for record in group])
        if len(response.data) != len(group):
            raise ValueError("Embedding API returned an unexpected number of vectors")
        for record, item in zip(group, sorted(response.data, key=lambda x: x.index)):
            vector = item.embedding
            if not isinstance(vector, list) or len(vector) < 10:
                raise ValueError(f"Invalid embedding shape for {record.id}")
            if any(not math.isfinite(value) for value in vector):
                raise ValueError(f"Non-finite embedding for {record.id}")
            norm = math.sqrt(sum(value * value for value in vector))
            if not (0.98 <= norm <= 1.02):
                raise ValueError(f"Expected unit vector for {record.id}")
            if not (min(vector) < 0 < max(vector)):
                raise ValueError(f"Suspicious one-sided vector for {record.id}")
            if dims is None:
                dims = len(vector)
            if len(vector) != dims:
                raise ValueError("Inconsistent embedding dimension")
            embeddings[record.id] = vector
    result = {
        "version": 1, "model": model, "dimensions": dims,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint(records), "embeddings": embeddings,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically, not to git. A deploy with ephemeral storage must supply this index.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                     dir=output.parent, delete=False) as fh:
        temp_path = Path(fh.name)
        json.dump(result, fh, ensure_ascii=False, separators=(",", ":"))
    temp_path.replace(output)
    return {"records": len(records), "dimensions": dims,
            "model": model, "path": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("LAWAI_EMBEDDING_MODEL", EMBEDDING_MODEL))
    parser.add_argument("--output", default=os.getenv(
        "LAWAI_INDEX_PATH", str(ROOT / "data" / "embedding_index.json")))
    args = parser.parse_args()
    print(json.dumps(build_index(args.model, Path(args.output)), ensure_ascii=False))
