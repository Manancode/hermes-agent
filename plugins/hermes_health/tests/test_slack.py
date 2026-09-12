"""Tests for hermes_health Slack notification integration.

Covers:
    1. Slack disabled when no webhook URL configured
    2. Slack enabled when webhook URL configured
    3. Correct Slack payload structure and content
    4. Webhook failure isolation (Slack error never breaks Hermes)
    5. Webhook URL never logged
    6. Existing hermes_health tests still pass
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from hermes_health.notifiers import AlertNotifier
from hermes_health.notifiers_slack import (
    _build_slack_payload,
    _get_webhook_url,
    _post_webhook,
    make_slack_callback,
)
from hermes_health.state import HealthStatus


_FAKE_WEBHOOK_URL = "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"


# ---------------------------------------------------------------------------
# 1. Slack disabled when no webhook URL configured
# ---------------------------------------------------------------------------

class TestSlackDisabled:
    def test_no_env_var_returns_none(self, monkeypatch):
        monkeypatch.delenv("HERMES_HEALTH_SLACK_WEBHOOK_URL", raising=False)
        assert make_slack_callback() is None

    def test_empty_env_var_returns_none(self, monkeypatch):
        monkeypatch.setenv("HERMES_HEALTH_SLACK_WEBHOOK_URL", "")
        assert make_slack_callback() is None

    def test_whitespace_only_env_var_returns_none(self, monkeypatch):
        monkeypatch.setenv("HERMES_HEALTH_SLACK_WEBHOOK_URL", "   ")
        assert make_slack_callback() is None


# ---------------------------------------------------------------------------
# 2. Slack enabled when webhook URL configured
# ---------------------------------------------------------------------------

class TestSlackEnabled:
    def test_valid_url_returns_callback(self, monkeypatch):
        monkeypatch.setenv("HERMES_HEALTH_SLACK_WEBHOOK_URL", _FAKE_WEBHOOK_URL)
        cb = make_slack_callback()
        assert cb is not None
        assert callable(cb)

    def test_callback_fires_on_transition(self, monkeypatch):
        monkeypatch.setenv("HERMES_HEALTH_SLACK_WEBHOOK_URL", _FAKE_WEBHOOK_URL)
        cb = make_slack_callback()
        assert cb is not None

        with patch("hermes_health.notifiers_slack._post_webhook") as mock_post:
            cb("degraded", "Pipeline degraded")
            mock_post.assert_called_once()

    def test_callback_registered_in_notifier(self, monkeypatch):
        monkeypatch.setenv("HERMES_HEALTH_SLACK_WEBHOOK_URL", _FAKE_WEBHOOK_URL)

        notifier = AlertNotifier(cooldown_secs=0)
        cb = make_slack_callback()
        assert cb is not None
        notifier.add_callback(cb)

        with patch("hermes_health.notifiers_slack._post_webhook") as mock_post:
            notifier.on_transition(HealthStatus.HEALTHY, HealthStatus.DEGRADED, "degraded msg")
            mock_post.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Correct payload
# ---------------------------------------------------------------------------

class TestSlackPayload:
    def test_recovery_payload(self):
        payload = _build_slack_payload("recovery", "hermes recovered")
        assert "blocks" in payload
        blocks = payload["blocks"]

        header = blocks[0]
        assert header["type"] == "header"
        assert "recovered" in header["text"]["text"]
        assert "\u2705" in header["text"]["text"]

        fields = blocks[1]["fields"]
        assert any("hermes-pipeline" in f["text"] for f in fields)
        assert any("info" in f["text"] for f in fields)

        detail = blocks[2]
        assert "hermes recovered" in detail["text"]["text"]

        context = blocks[3]
        assert "Timestamp:" in context["elements"][0]["text"]

    def test_degraded_payload(self):
        payload = _build_slack_payload("degraded", "hermes degraded, 5 errors in 300s window")
        header_text = payload["blocks"][0]["text"]["text"]
        assert "degraded" in header_text
        assert "\u26a0\ufe0f" in header_text

        fields = payload["blocks"][1]["fields"]
        assert any("warning" in f["text"] for f in fields)

    def test_error_payload(self):
        payload = _build_slack_payload("error", "hermes down, 15 errors in 300s window")
        header_text = payload["blocks"][0]["text"]["text"]
        assert "down" in header_text
        assert "\U0001f534" in header_text

        fields = payload["blocks"][1]["fields"]
        assert any("critical" in f["text"] for f in fields)

    def test_payload_has_all_required_fields(self):
        payload = _build_slack_payload("error", "test detail")
        blocks = payload["blocks"]

        # Header block
        assert blocks[0]["type"] == "header"

        # Component + Severity fields
        field_texts = " ".join(f["text"] for f in blocks[1]["fields"])
        assert "Component:" in field_texts
        assert "Severity:" in field_texts

        # Detail section
        assert "Detail:" in blocks[2]["text"]["text"]

        # Timestamp context
        assert "Timestamp:" in blocks[3]["elements"][0]["text"]


# ---------------------------------------------------------------------------
# 4. Webhook failure isolation
# ---------------------------------------------------------------------------

class TestWebhookFailureIsolation:
    def test_post_webhook_does_not_raise_on_http_error(self):
        with patch("hermes_health.notifiers_slack.urllib.request.urlopen") as mock_open:
            mock_open.side_effect = urllib.error.URLError("connection refused")
            _post_webhook(_FAKE_WEBHOOK_URL, {"text": "test"})

    def test_post_webhook_does_not_raise_on_timeout(self):
        with patch("hermes_health.notifiers_slack.urllib.request.urlopen") as mock_open:
            mock_open.side_effect = TimeoutError("timed out")
            _post_webhook(_FAKE_WEBHOOK_URL, {"text": "test"})

    def test_callback_does_not_raise_on_webhook_failure(self, monkeypatch):
        monkeypatch.setenv("HERMES_HEALTH_SLACK_WEBHOOK_URL", _FAKE_WEBHOOK_URL)
        cb = make_slack_callback()
        assert cb is not None

        with patch("hermes_health.notifiers_slack.urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            cb("error", "test")

    def test_notifier_continues_after_slack_failure(self, monkeypatch):
        monkeypatch.setenv("HERMES_HEALTH_SLACK_WEBHOOK_URL", _FAKE_WEBHOOK_URL)

        notifier = AlertNotifier(cooldown_secs=0)
        cb = make_slack_callback()
        notifier.add_callback(cb)

        other_fired = []
        notifier.add_callback(lambda t, m: other_fired.append((t, m)))

        with patch("hermes_health.notifiers_slack._post_webhook", side_effect=RuntimeError("boom")):
            notifier.on_transition(HealthStatus.HEALTHY, HealthStatus.ERROR, "pipeline down")

        assert len(other_fired) == 1
        assert other_fired[0][0] == "error"


# ---------------------------------------------------------------------------
# 5. Webhook URL never logged
# ---------------------------------------------------------------------------

class TestSecretNotLogged:
    def test_webhook_url_not_in_log_output(self, monkeypatch, caplog):
        monkeypatch.setenv("HERMES_HEALTH_SLACK_WEBHOOK_URL", _FAKE_WEBHOOK_URL)

        with caplog.at_level(logging.DEBUG):
            _get_webhook_url()
            with patch("hermes_health.notifiers_slack.urllib.request.urlopen") as mock_open:
                mock_open.side_effect = urllib.error.URLError("refused")
                _post_webhook(_FAKE_WEBHOOK_URL, {"text": "test"})

        assert _FAKE_WEBHOOK_URL not in caplog.text

    def test_webhook_url_not_in_exception_traceback(self, monkeypatch, caplog):
        monkeypatch.setenv("HERMES_HEALTH_SLACK_WEBHOOK_URL", _FAKE_WEBHOOK_URL)

        with caplog.at_level(logging.DEBUG):
            with patch("hermes_health.notifiers_slack.urllib.request.urlopen") as mock_open:
                mock_open.side_effect = urllib.error.URLError("refused")
                _post_webhook(_FAKE_WEBHOOK_URL, {"text": "test"})

        assert "hooks.slack.com" not in caplog.text
        assert "XXXXXXXXXXXXXXXXXXXXXXXX" not in caplog.text
