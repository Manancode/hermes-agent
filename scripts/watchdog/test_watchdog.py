#!/usr/bin/env python3
"""
Watchdog test suite with failure injection.

Tests the watchdog without modifying production systems.
Uses mock servers and temporary state files.

Usage:
    python test_watchdog.py              # Run all tests
    python test_watchdog.py -v           # Verbose output
    python test_watchdog.py TestGateway  # Run specific test class
"""

from __future__ import annotations

import ast
import http.server
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

import watchdog as wd


# ---------------------------------------------------------------------------
# Mock HTTP Server
# ---------------------------------------------------------------------------

class MockHealthHandler(http.server.BaseHTTPRequestHandler):
    health_response = {"status": "ok"}
    health_code = 200

    def do_GET(self):
        if self.path == "/health" or self.path == "/api/nous/recommended-models":
            self.send_response(self.health_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.health_response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def start_mock_server(port=18420):
    server = http.server.HTTPServer(("127.0.0.1", port), MockHealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def make_heartbeat(stale=False, pid=12345):
    now = datetime.now(timezone.utc)
    if stale:
        from datetime import timedelta
        now = now - timedelta(seconds=120)
    return {"pid": pid, "updated_at": now.isoformat(), "monotonic": 12345.67}


def write_heartbeat(path, stale=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_heartbeat(stale=stale)), encoding="utf-8")


def make_config(tmp_dir=None, gw_port=18420, vps_host="", hcnsc_port=18422):
    tmp = tmp_dir or tempfile.mkdtemp()
    return {
        "hermes_home": tmp,
        "gateway": {
            "heartbeat_file": os.path.join(tmp, "heartbeat"),
            "heartbeat_stale_secs": 90,
            "health_url": f"http://127.0.0.1:{gw_port}/health",
            "health_timeout_secs": 3,
        },
        "vps": {
            "ssh_host": vps_host,
            "ssh_user": "",
            "ssh_key": "",
            "ssh_timeout_secs": 3,
            "memorycore_health_url": "http://127.0.0.1:8420/health",
            "memorycore_health_timeout_secs": 3,
        },
        "hcnsc": {
            "passive_health_url": f"http://127.0.0.1:{hcnsc_port}/api/nous/recommended-models",
            "passive_health_timeout_secs": 3,
        },
        "alerts": {
            "ntfy": {"enabled": False, "server": "", "topic": "", "token": ""},
            "telegram": {"enabled": False, "bot_token": "", "chat_id": ""},
            "slack": {"enabled": False, "webhook_url": ""},
            "cooldown_secs": 5,
            "escalation_secs": 60,
            "max_alerts_per_hour": 10,
        },
        "state_file": os.path.join(tmp, "state.json"),
        "log_file": os.path.join(tmp, "watchdog.log"),
    }


# ---------------------------------------------------------------------------
# Test: No Synthetic Inference Audit
# ---------------------------------------------------------------------------

class TestNoSyntheticInference(unittest.TestCase):
    """Verify the watchdog NEVER makes LLM inference calls.

    This test parses the watchdog source code and checks:
    1. No URL contains /chat/completions, /completions, or /embeddings
    2. No function name suggests inference (generate, complete, chat, predict)
    3. The only outbound HTTP is to health/readiness endpoints
    """

    def setUp(self):
        self.source = Path(__file__).resolve().parent / "watchdog.py"
        self.tree = ast.parse(self.source.read_text(encoding="utf-8"))

    def test_no_chat_completions_url_in_source(self):
        """No hardcoded inference URLs in the watchdog source."""
        source_text = self.source.read_text(encoding="utf-8")
        for pattern in ("/chat/completions", "/completions", "/embeddings"):
            # Allow the FORBIDDEN_URL_PATTERNS constant which is a guard
            lines = source_text.split("\n")
            for i, line in enumerate(lines, 1):
                if pattern in line and "FORBIDDEN_URL_PATTERNS" not in line:
                    self.fail(f"Line {i} contains forbidden pattern '{pattern}': {line.strip()}")

    def test_no_inference_function_names(self):
        """No function names suggesting LLM inference."""
        inference_names = {"generate", "complete", "chat", "predict", "infer", "llm_call", "invoke_model"}
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.assertNotIn(node.name, inference_names,
                    f"Function '{node.name}' suggests LLM inference — not allowed in watchdog")

    def test_only_health_related_urls(self):
        """All outbound URLs are health/readiness endpoints."""
        source_text = self.source.read_text(encoding="utf-8")
        # Check that no URL in the source is an inference endpoint
        for pattern in ("api.openai.com", "inference-api.nousresearch.com/v1/chat",
                        "api.anthropic.com", "generativelanguage.googleapis.com"):
            self.assertNotIn(pattern, source_text,
                f"Found inference API URL in watchdog source: {pattern}")

    def test_forbidden_patterns_constant_exists(self):
        """The FORBIDDEN_URL_PATTERNS guard exists and blocks inference URLs."""
        self.assertTrue(hasattr(wd, "FORBIDDEN_URL_PATTERNS"))
        self.assertIn("/chat/completions", wd.FORBIDDEN_URL_PATTERNS)
        self.assertIn("/completions", wd.FORBIDDEN_URL_PATTERNS)
        self.assertIn("/embeddings", wd.FORBIDDEN_URL_PATTERNS)

    def test_hcnsc_check_uses_passive_endpoint_only(self):
        """The hcnsc check function fetches a non-inference endpoint."""
        source = self.source.read_text(encoding="utf-8")
        # The hcnsc function should only use GET on recommended-models or similar
        self.assertNotIn("chat/completions", source.split("def check_hcnsc")[1].split("\ndef ")[0]
            if "def check_hcnsc" in source else "")


# ---------------------------------------------------------------------------
# Test: Gateway Heartbeat
# ---------------------------------------------------------------------------

class TestGatewayHeartbeat(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.hb_file = os.path.join(self.tmp, "heartbeat")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fresh(self):
        write_heartbeat(Path(self.hb_file), stale=False)
        level, detail = wd.check_gateway_heartbeat(self.hb_file, stale_secs=90)
        self.assertEqual(level, wd.LEVEL_OK)
        self.assertIn("fresh", detail)

    def test_stale(self):
        write_heartbeat(Path(self.hb_file), stale=True)
        level, detail = wd.check_gateway_heartbeat(self.hb_file, stale_secs=90)
        self.assertEqual(level, wd.LEVEL_DOWN)
        self.assertIn("stale", detail)

    def test_missing(self):
        level, detail = wd.check_gateway_heartbeat("/nonexistent/path", stale_secs=90)
        self.assertEqual(level, wd.LEVEL_DOWN)
        self.assertIn("missing", detail)

    def test_corrupt(self):
        Path(self.hb_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.hb_file).write_text("not json", encoding="utf-8")
        level, detail = wd.check_gateway_heartbeat(self.hb_file, stale_secs=90)
        self.assertEqual(level, wd.LEVEL_DOWN)
        self.assertIn("unreadable", detail)


# ---------------------------------------------------------------------------
# Test: Gateway Health
# ---------------------------------------------------------------------------

class TestGatewayHealth(unittest.TestCase):
    def setUp(self):
        self.server = start_mock_server(18420)

    def tearDown(self):
        self.server.shutdown()

    def test_ok(self):
        MockHealthHandler.health_response = {"status": "ok", "version": "0.21.2"}
        level, detail = wd.check_gateway_health("http://127.0.0.1:18420/health", timeout=3)
        self.assertEqual(level, wd.LEVEL_OK)

    def test_degraded(self):
        MockHealthHandler.health_response = {"status": "degraded"}
        level, detail = wd.check_gateway_health("http://127.0.0.1:18420/health", timeout=3)
        self.assertEqual(level, wd.LEVEL_DEGRADED)

    def test_down(self):
        MockHealthHandler.health_response = {"status": "error"}
        level, detail = wd.check_gateway_health("http://127.0.0.1:18420/health", timeout=3)
        self.assertEqual(level, wd.LEVEL_DOWN)

    def test_unreachable(self):
        level, detail = wd.check_gateway_health("http://127.0.0.1:19999/health", timeout=1)
        self.assertEqual(level, wd.LEVEL_DOWN)


# ---------------------------------------------------------------------------
# Test: HCNSC Passive Reachability
# ---------------------------------------------------------------------------

class TestHcnscPassive(unittest.TestCase):
    def setUp(self):
        self.server = start_mock_server(18422)

    def tearDown(self):
        self.server.shutdown()

    def test_ok(self):
        MockHealthHandler.health_response = {"paidRecommendedModels": []}
        MockHealthHandler.health_code = 200
        level, detail = wd.check_hcnsc_passive({
            "passive_health_url": "http://127.0.0.1:18422/api/nous/recommended-models",
            "passive_health_timeout_secs": 3,
        })
        self.assertEqual(level, wd.LEVEL_OK)
        self.assertIn("reachable", detail)

    def test_server_error(self):
        MockHealthHandler.health_code = 503
        level, detail = wd.check_hcnsc_passive({
            "passive_health_url": "http://127.0.0.1:18422/api/nous/recommended-models",
            "passive_health_timeout_secs": 3,
        })
        self.assertEqual(level, wd.LEVEL_DEGRADED)

    def test_unreachable(self):
        level, detail = wd.check_hcnsc_passive({
            "passive_health_url": "http://127.0.0.1:19999/api/nous/recommended-models",
            "passive_health_timeout_secs": 1,
        })
        self.assertEqual(level, wd.LEVEL_DOWN)

    def test_forbidden_url_rejected(self):
        """Watchdog refuses to hit inference endpoints."""
        level, detail = wd.check_hcnsc_passive({
            "passive_health_url": "https://api.example.com/chat/completions",
            "passive_health_timeout_secs": 3,
        })
        self.assertEqual(level, wd.LEVEL_UNKNOWN)
        self.assertIn("forbidden", detail)


# ---------------------------------------------------------------------------
# Test: State Management
# ---------------------------------------------------------------------------

class TestStateManagement(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.state_path = os.path.join(self.tmp, "state.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_load_empty(self):
        state = wd.load_state("/nonexistent/state.json")
        self.assertIn("components", state)

    def test_save_and_load(self):
        state = {"components": {"gateway": {"level": "ok"}}}
        wd.save_state(state, self.state_path)
        loaded = wd.load_state(self.state_path)
        self.assertEqual(loaded["components"]["gateway"]["level"], "ok")

    def test_update_failure(self):
        state = {"components": {}, "alerts_sent": []}
        now = datetime.now(timezone.utc)
        comp = wd.update_component_state(state, "gateway", wd.LEVEL_DOWN, "test", now)
        self.assertEqual(comp["level"], wd.LEVEL_DOWN)
        self.assertEqual(comp["consecutive_failures"], 1)

    def test_update_recovery(self):
        state = {"components": {}, "alerts_sent": []}
        now = datetime.now(timezone.utc)
        wd.update_component_state(state, "gateway", wd.LEVEL_DOWN, "down", now)
        comp = wd.update_component_state(state, "gateway", wd.LEVEL_OK, "ok", now)
        self.assertEqual(comp["level"], wd.LEVEL_OK)
        self.assertEqual(comp["consecutive_failures"], 0)
        self.assertIsNone(comp["first_failure"])


# ---------------------------------------------------------------------------
# Test: Alert Formatting
# ---------------------------------------------------------------------------

class TestAlertFormatting(unittest.TestCase):
    def test_down(self):
        title, body = wd.format_alert("gateway", wd.ALERT_DOWN, "heartbeat stale", wd.LEVEL_DOWN)
        self.assertEqual(title, "gateway down")
        self.assertIn("heartbeat stale", body)

    def test_recovery(self):
        title, body = wd.format_alert("memorycore", wd.ALERT_RECOVERY, "ok", wd.LEVEL_OK, duration="5m 30s")
        self.assertEqual(title, "memorycore recovered")
        self.assertIn("5m 30s", body)

    def test_degraded(self):
        title, body = wd.format_alert("hcnsc", wd.ALERT_DEGRADED, "HTTP 503", wd.LEVEL_DEGRADED)
        self.assertEqual(title, "hcnsc degraded")


# ---------------------------------------------------------------------------
# Test: Alert Dedup
# ---------------------------------------------------------------------------

class TestAlertDedup(unittest.TestCase):
    def test_first_allowed(self):
        state = {"alerts_sent": []}
        now = datetime.now(timezone.utc)
        self.assertTrue(wd.should_alert(state, "gateway", wd.ALERT_DOWN, 300, 6, now))

    def test_cooldown_blocks(self):
        state = {"alerts_sent": [{"component": "gateway", "type": "down", "time": time.time()}]}
        now = datetime.now(timezone.utc)
        self.assertFalse(wd.should_alert(state, "gateway", wd.ALERT_DOWN, 300, 6, now))

    def test_cooldown_expires(self):
        state = {"alerts_sent": [{"component": "gateway", "type": "down", "time": time.time() - 400}]}
        now = datetime.now(timezone.utc)
        self.assertTrue(wd.should_alert(state, "gateway", wd.ALERT_DOWN, 300, 6, now))

    def test_rate_limit(self):
        now = time.time()
        state = {"alerts_sent": [{"component": "gateway", "type": "down", "time": now - i} for i in range(6)]}
        now_dt = datetime.now(timezone.utc)
        self.assertFalse(wd.should_alert(state, "gateway", wd.ALERT_DOWN, 0, 6, now_dt))

    def test_independent_components(self):
        state = {"alerts_sent": [{"component": "gateway", "type": "down", "time": time.time()}]}
        now = datetime.now(timezone.utc)
        self.assertTrue(wd.should_alert(state, "memorycore", wd.ALERT_DOWN, 300, 6, now))


# ---------------------------------------------------------------------------
# Test: Composite Evaluation (distinct incident types)
# ---------------------------------------------------------------------------

class TestCompositeEvaluation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = make_config(tmp_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_all_healthy_no_alerts(self):
        results = {
            "gateway_heartbeat": (wd.LEVEL_OK, "fresh"),
            "gateway_health": (wd.LEVEL_OK, "ok"),
            "vps": (wd.LEVEL_UNKNOWN, "not configured"),
            "hcnsc": (wd.LEVEL_UNKNOWN, "not configured"),
        }
        state = {"components": {}, "alerts_sent": []}
        now = datetime.now(timezone.utc)
        sent = wd.evaluate_and_alert(results, state, self.config, now, wd.setup_logging(None))
        self.assertEqual(sent, [])

    def test_vps_unreachable_separate_from_gateway(self):
        """VPS down does NOT affect gateway status — they are independent."""
        results = {
            "gateway_heartbeat": (wd.LEVEL_OK, "fresh"),
            "gateway_health": (wd.LEVEL_OK, "ok"),
            "vps": (wd.LEVEL_DOWN, "ssh failed"),
            "hcnsc": (wd.LEVEL_UNKNOWN, "not configured"),
        }
        state = {"components": {}, "alerts_sent": []}
        now = datetime.now(timezone.utc)
        sent = wd.evaluate_and_alert(results, state, self.config, now, wd.setup_logging(None))
        # Only VPS alert, not gateway
        self.assertTrue(any("vps" in t for t in sent))
        self.assertFalse(any("gateway" in t for t in sent))
        self.assertEqual(state["components"]["gateway"]["level"], wd.LEVEL_OK)

    def test_vps_ok_memorycore_down(self):
        """VPS reachable but MemoryCore down — distinct from VPS down."""
        results = {
            "gateway_heartbeat": (wd.LEVEL_OK, "fresh"),
            "gateway_health": (wd.LEVEL_OK, "ok"),
            "vps": (wd.LEVEL_OK, "reachable"),
            "vps_memorycore": (wd.LEVEL_DOWN, "unreachable"),
            "hcnsc": (wd.LEVEL_UNKNOWN, "not configured"),
        }
        state = {"components": {}, "alerts_sent": []}
        now = datetime.now(timezone.utc)
        sent = wd.evaluate_and_alert(results, state, self.config, now, wd.setup_logging(None))
        self.assertTrue(any("memorycore" in t for t in sent))
        # VPS should be marked ok
        self.assertEqual(state["components"]["vps"]["level"], wd.LEVEL_OK)

    def test_memorycore_ok_hcnsc_down(self):
        """MemoryCore healthy but HCNSC unreachable — distinct incident."""
        results = {
            "gateway_heartbeat": (wd.LEVEL_OK, "fresh"),
            "gateway_health": (wd.LEVEL_OK, "ok"),
            "vps": (wd.LEVEL_OK, "reachable"),
            "vps_memorycore": (wd.LEVEL_OK, "ok"),
            "hcnsc": (wd.LEVEL_DOWN, "unreachable"),
        }
        state = {"components": {}, "alerts_sent": []}
        now = datetime.now(timezone.utc)
        sent = wd.evaluate_and_alert(results, state, self.config, now, wd.setup_logging(None))
        self.assertTrue(any("hcnsc" in t for t in sent))
        self.assertFalse(any("memorycore" in t for t in sent))

    def test_gateway_down_independent_of_vps(self):
        """Gateway down does NOT depend on VPS status."""
        results = {
            "gateway_heartbeat": (wd.LEVEL_DOWN, "stale"),
            "gateway_health": (wd.LEVEL_DOWN, "unreachable"),
            "vps": (wd.LEVEL_OK, "reachable"),
            "vps_memorycore": (wd.LEVEL_OK, "ok"),
            "hcnsc": (wd.LEVEL_OK, "reachable"),
        }
        state = {"components": {}, "alerts_sent": []}
        now = datetime.now(timezone.utc)
        sent = wd.evaluate_and_alert(results, state, self.config, now, wd.setup_logging(None))
        self.assertTrue(any("gateway" in t for t in sent))
        # VPS + MC + HCNSC should NOT alert
        self.assertFalse(any("vps" in t for t in sent))
        self.assertFalse(any("memorycore" in t for t in sent))
        self.assertFalse(any("hcnsc" in t for t in sent))

    def test_recovery_after_failure(self):
        results_down = {
            "gateway_heartbeat": (wd.LEVEL_OK, "fresh"),
            "gateway_health": (wd.LEVEL_OK, "ok"),
            "vps": (wd.LEVEL_DOWN, "ssh failed"),
            "hcnsc": (wd.LEVEL_UNKNOWN, "not configured"),
        }
        state = {"components": {}, "alerts_sent": []}
        now = datetime.now(timezone.utc)
        wd.evaluate_and_alert(results_down, state, self.config, now, wd.setup_logging(None))

        results_ok = {
            "gateway_heartbeat": (wd.LEVEL_OK, "fresh"),
            "gateway_health": (wd.LEVEL_OK, "ok"),
            "vps": (wd.LEVEL_OK, "reachable"),
            "hcnsc": (wd.LEVEL_UNKNOWN, "not configured"),
        }
        sent = wd.evaluate_and_alert(results_ok, state, self.config, now, wd.setup_logging(None))
        self.assertTrue(any("recovered" in t for t in sent))

    def test_classify_composite_distinct_incidents(self):
        """All failure cases produce distinct incident types."""
        results_all_down = {
            "gateway_heartbeat": (wd.LEVEL_DOWN, "stale"),
            "gateway_health": (wd.LEVEL_DOWN, "down"),
            "vps": (wd.LEVEL_DOWN, "ssh failed"),
            "vps_memorycore": (wd.LEVEL_DOWN, "unreachable"),
            "hcnsc": (wd.LEVEL_DOWN, "unreachable"),
        }
        incidents = wd._classify_composite(results_all_down)
        components = [c for c, _, _ in incidents]
        # Each component produces its own incident
        self.assertIn("gateway", components)
        self.assertIn("vps", components)
        self.assertIn("hcnsc", components)
        # MemoryCore incident only when VPS is reachable
        self.assertNotIn("memorycore", components)

    def test_classify_composite_vps_down_hides_memorycore(self):
        """When VPS is down, MemoryCore check is irrelevant."""
        results = {
            "gateway_heartbeat": (wd.LEVEL_OK, "fresh"),
            "gateway_health": (wd.LEVEL_OK, "ok"),
            "vps": (wd.LEVEL_DOWN, "ssh failed"),
            "vps_memorycore": (wd.LEVEL_DOWN, "also down"),
            "hcnsc": (wd.LEVEL_OK, "reachable"),
        }
        incidents = wd._classify_composite(results)
        components = [c for c, _, _ in incidents]
        self.assertIn("vps", components)
        self.assertNotIn("memorycore", components, "MemoryCore incident should not appear when VPS is down")


# ---------------------------------------------------------------------------
# Test: End-to-End
# ---------------------------------------------------------------------------

class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.gw_server = start_mock_server(18420)
        self.hcnsc_server = start_mock_server(18422)

    def tearDown(self):
        self.gw_server.shutdown()
        self.hcnsc_server.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_check_run(self):
        MockHealthHandler.health_response = {"status": "ok", "version": "test"}
        MockHealthHandler.health_code = 200
        hb_file = os.path.join(self.tmp, "heartbeat")
        write_heartbeat(Path(hb_file), stale=False)

        config = make_config(tmp_dir=self.tmp, gw_port=18420, hcnsc_port=18422)
        logger = wd.setup_logging(None)
        results = wd.run_checks(config, logger)

        self.assertEqual(results["gateway_heartbeat"][0], wd.LEVEL_OK)
        self.assertEqual(results["gateway_health"][0], wd.LEVEL_OK)
        self.assertEqual(results["hcnsc"][0], wd.LEVEL_OK)


# ---------------------------------------------------------------------------
# Test: Slack Integration
# ---------------------------------------------------------------------------

class TestSlackIntegration(unittest.TestCase):
    def test_send_slack_disabled(self):
        """Slack disabled returns False without sending."""
        result = wd.send_slack("", "title", "body")
        self.assertFalse(result)

    def test_send_slack_enabled(self):
        """Slack enabled sends to webhook."""
        import urllib.request
        received = []

        class MockSlackHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                received.append(json.loads(body))
                self.send_response(200)
                self.end_headers()

            def log_message(self, format, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 18423), MockSlackHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            result = wd.send_slack(
                "http://127.0.0.1:18423/webhook",
                "Test Title",
                "Test Body",
            )
            self.assertTrue(result)
            self.assertEqual(len(received), 1)
            self.assertIn("Test Title", received[0]["text"])
        finally:
            server.shutdown()

    def test_send_slack_failure_isolation(self):
        """Slack failure does not raise exception."""
        result = wd.send_slack(
            "http://127.0.0.1:19999/webhook",
            "title",
            "body",
        )
        self.assertFalse(result)

    def test_slack_secret_protection(self):
        """Webhook URL is never logged."""
        import logging
        import io

        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        logger = wd.setup_logging(None)
        logger.addHandler(handler)

        webhook_url = "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
        wd.send_slack(webhook_url, "title", "body", logger=logger)

        log_output = log_stream.getvalue()
        self.assertNotIn("hooks.slack.com", log_output)
        self.assertNotIn("T00000000", log_output)
        self.assertNotIn("XXXXXXXXXXXXXXXXXXXXXXXX", log_output)

    def test_slack_in_config(self):
        """Config includes slack section."""
        config = make_config()
        self.assertIn("slack", config.get("alerts", {}))
        self.assertFalse(config["alerts"]["slack"]["enabled"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
