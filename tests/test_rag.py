"""Offline tests: do not make paid OpenAI requests or use live legal case data."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import app
from retrieval import (Record, authoritative_hits, fingerprint, lexical_score,
                       load_index, load_records, rank_records)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.records = load_records()

    def test_legacy_answers_are_kept_but_vectors_are_ignored(self):
        self.assertEqual(30, len(self.records))
        self.assertEqual("faq", self.records[0].kind)
        self.assertFalse(self.records[0].approved_legal)
        self.assertFalse(hasattr(self.records[0], "vector"))

    def test_exact_and_partial_question(self):
        matches = rank_records("我要離婚", self.records)
        self.assertTrue(matches)
        self.assertEqual("我要離婚", matches[0]["question"])
        self.assertGreaterEqual(matches[0]["keyword_score"], 0.99)

    def test_unrelated_question_abstains(self):
        matches = rank_records("火星火箭燃料推進器試飛技術", self.records)
        self.assertEqual([], matches)

    def test_blank_question_abstains(self):
        self.assertEqual([], rank_records("請問", self.records))

    def test_outdated_or_fake_embedding_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "index.json"
            p.write_text(json.dumps({
                "version": 1, "model": "text-embedding-3-small",
                "fingerprint": fingerprint(self.records), "dimensions": 12,
                "embeddings": {r.id: [0.5] * 12 for r in self.records},
            }))
            vectors, status = load_index(self.records, p, "text-embedding-3-small")
            self.assertEqual({}, vectors)
            self.assertEqual("non_normalized_vectors", status)

    def test_approved_legal_sources_require_human_metadata(self):
        r = Record(id="law-1", title="測試法律", content="測試資料",
                   kind="law", source_url="https://example.org/test",
                   review_status="approved", reviewed_at="2026-11-16",
                   reviewed_by="reviewer")
        self.assertTrue(r.approved_legal)
        self.assertFalse(Record(id="x", title="x", content="y", kind="law",
                                review_status="approved").approved_legal)

    def test_retrieved_faq_is_not_authoritative(self):
        hits = rank_records("我要離婚", self.records)
        self.assertEqual([], authoritative_hits(hits))


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def test_health_without_paid_calls(self):
        response = self.client.get("/api/health")
        self.assertEqual(200, response.status_code)
        self.assertEqual(30, response.json["records"])
        self.assertEqual(0, response.json["approved_legal_sources"])

    def test_faq_response_is_clearly_disclaimed(self):
        with patch.dict(os.environ, {"LAWAI_API_TOKEN": ""}):
            response = self.client.post("/api/generate", json={"question": "我要離婚"})
        self.assertEqual(200, response.status_code)
        self.assertIn("不是個案法律意見", response.json["answer"])

    def test_no_forced_answer(self):
        with patch.dict(os.environ, {"LAWAI_API_TOKEN": ""}):
            response = self.client.post("/api/generate", json={
                "question": "火星火箭燃料推進器試飛技術"})
        self.assertEqual(200, response.status_code)
        self.assertEqual("no_verified_answer", response.json["status"])

    def test_non_json_rejected(self):
        with patch.dict(os.environ, {"LAWAI_API_TOKEN": ""}):
            response = self.client.post("/api/search", data="not json")
        self.assertEqual(415, response.status_code)

    def test_compare_disabled_by_default(self):
        with patch.dict(os.environ, {"LAWAI_ENABLE_COMPARISON": "0"}):
            response = self.client.post("/api/compare", json={"question": "我要離婚"})
        self.assertEqual(403, response.status_code)

    def test_compare_requires_token_even_if_enabled(self):
        with patch.dict(os.environ, {"LAWAI_ENABLE_COMPARISON": "1",
                                    "LAWAI_API_TOKEN": ""}):
            response = self.client.post("/api/compare", json={"question": "我要離婚"})
        self.assertEqual(503, response.status_code)

    def test_citation_check_only_verifies_ids(self):
        result = app._extract_citations("[S1] [S3]", {"[S1]", "[S2]"})
        self.assertEqual(["[S1]"], result["cited"])
        self.assertEqual(["[S3]"], result["invalid"])


if __name__ == "__main__":
    unittest.main()
