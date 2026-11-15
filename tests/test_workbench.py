"""Web UI and explicit origin CORS checks: no paid OpenAI calls."""
import os
import unittest
from unittest.mock import patch
import app


class WebFrontendTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def test_root_redirects_to_new_research_workbench(self):
        response = self.client.get("/")
        self.assertEqual(302, response.status_code)
        self.assertEqual("/web/", response.headers["Location"])
        page = self.client.get("/web/")
        self.assertEqual(200, page.status_code)
        self.assertIn("法律檢索研究工作臺".encode("utf-8"), page.data)
        self.assertIn(b"./workbench.css", page.data)
        self.assertIn(b"./workbench.js", page.data)

    def test_styles_and_js_are_served(self):
        self.assertEqual(200, self.client.get("/web/workbench.css").status_code)
        self.assertEqual(200, self.client.get("/web/workbench.js").status_code)
        self.assertEqual(200, self.client.get("/web/config.js").status_code)
        self.assertEqual(404, self.client.get("/web/../../app.py").status_code)

    def test_cors_allows_exact_explicit_origin(self):
        with patch.dict(os.environ, {"LAWAI_ALLOWED_ORIGINS": "https://lab.example.com"}):
            response = self.client.options("/api/search", headers={
                "Origin": "https://lab.example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            })
        self.assertEqual(200, response.status_code)
        self.assertEqual("https://lab.example.com",
                         response.headers.get("Access-Control-Allow-Origin"))
        self.assertIn("POST", response.headers.get("Access-Control-Allow-Methods", ""))
        self.assertIn("Content-Type", response.headers.get("Access-Control-Allow-Headers", ""))
        self.assertNotIn("Access-Control-Allow-Credentials", response.headers)
        self.assertIn("Origin", response.headers.get("Vary", ""))

    def test_cors_denies_unlisted_and_null_origins(self):
        with patch.dict(os.environ, {"LAWAI_ALLOWED_ORIGINS": "https://lab.example.com"}):
            for origin in ["https://attacker.example", "null",
                           "https://lab.example.com.attacker.example"]:
                response = self.client.options("/api/generate", headers={
                    "Origin": origin, "Access-Control-Request-Method": "POST"})
                self.assertNotIn("Access-Control-Allow-Origin", response.headers)

    def test_cors_does_not_expose_wildcard(self):
        with patch.dict(os.environ, {"LAWAI_ALLOWED_ORIGINS": "*"}):
            response = self.client.get("/api/health", headers={"Origin": "https://lab.example.com"})
        self.assertNotIn("Access-Control-Allow-Origin", response.headers)

    def test_successful_search_does_not_need_browser_cors(self):
        with patch.dict(os.environ, {"LAWAI_ALLOWED_ORIGINS": ""}):
            response = self.client.post("/api/search", json={"question": "我要離婚"})
        self.assertEqual(200, response.status_code)
        self.assertIn("hits", response.json)


if __name__ == "__main__":
    unittest.main()
