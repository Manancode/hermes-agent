"""Slack notification callback for hermes_health state-transition alerts.

Sends alerts to Slack via Incoming Webhooks when configured.
Webhook URL is read from HERMES_HEALTH_SLACK_WEBHOOK_URL env var.
When unset, Slack notifications are silently disabled.

No LLM calls. No inference requests. Slack failure never breaks Hermes.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_ENV_WEBHOOK_URL = "HERMES_HEALTH_SLACK_WEBHOOK_URL"

_ALERT_TYPE_MAP = {
    "recovery": {"emoji": "\u2705", "severity": "info", "status": "recovered"},
    "degraded": {"emoji": "\u26a0\ufe0f", "severity": "warning", "status": "degraded"},
    "error": {"emoji": "\U0001f534", "severity": "critical", "status": "down"},
}

_COMPONENT = "hermes-pipeline"


def _get_webhook_url() -> Optional[str]:
    """Read Slack webhook URL from environment. Returns None if not set."""
    url = os.environ.get(_ENV_WEBHOOK_URL, "").strip()
    return url or None


def _build_slack_payload(alert_type: str, message: str) -> dict:
    """Build a Slack Block Kit payload for a health alert."""
    meta = _ALERT_TYPE_MAP.get(alert_type, {"emoji": "\u2753", "severity": "unknown", "status": alert_type.upper()})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    header_text = f"{meta['emoji']} hermes health: {meta['status']}"

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": header_text, "emoji": True},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Component:*\n{_COMPONENT}"},
                {"type": "mrkdwn", "text": f"*Severity:*\n{meta['severity']}"},
            ],
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Detail:*\n{message}"},
        },
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"Timestamp: {now}"},
            ],
        },
    ]

    return {"blocks": blocks}


def _post_webhook(url: str, payload: dict) -> None:
    """POST JSON payload to Slack webhook. Never raises."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status not in (200, 204):
                logger.warning("Slack webhook returned status %d", resp.status)
    except Exception:
        logger.warning("Slack webhook delivery failed", exc_info=True)


def make_slack_callback():
    """Return a callback fn(alert_type, message) suitable for AlertNotifier.add_callback().

    If the webhook URL is not configured, returns None.
    The callback never raises.
    """
    url = _get_webhook_url()
    if url is None:
        return None

    def _slack_callback(alert_type: str, message: str) -> None:
        payload = _build_slack_payload(alert_type, message)
        _post_webhook(url, payload)

    return _slack_callback
