#!/usr/bin/env python3
"""
Hermes External Watchdog

Runs independently of Hermes via launchd. Monitors the full stack:

  Mac:       Hermes Desktop, Gateway, Backend
  VPS:       Docker, tdai-memory-core (port 8420)
  External:  HCNSC API (passive reachability only)

This watchdog NEVER makes synthetic LLM/inference calls.
HCNSC health is checked via passive reachability only.
Real inference health comes from Hermes/MemoryCore event logs.

Topology:
  - Gateway heartbeat file (local, 30s write cycle)
  - Gateway HTTP health (local)
  - VPS reachability (SSH probe)
  - MemoryCore on VPS (SSH + curl to remote /health)
  - HCNSC reachability (GET recommended-models, no auth, no inference)

Usage:
    python watchdog.py                    # Run once (for launchd)
    python watchdog.py --loop             # Run in loop (for testing)
    python watchdog.py --check-only       # Run checks, print results, no alerts
    python watchdog.py --inject <component>  # Inject failure for testing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"

LEVEL_OK = "ok"
LEVEL_DEGRADED = "degraded"
LEVEL_DOWN = "down"
LEVEL_UNKNOWN = "unknown"

ALERT_RECOVERY = "recovery"
ALERT_DOWN = "down"
ALERT_DEGRADED = "degraded"

# These strings appear in urllib.request source — we use them for the
# no-inference audit test. They must NOT appear in any outbound request
# URL built by this watchdog.
FORBIDDEN_URL_PATTERNS = ("/chat/completions", "/completions", "/embeddings")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_CONFIG
    try:
        raw = cfg_path.read_text(encoding="utf-8")
        # Expand environment variables: ${VAR} or $VAR
        import re
        def _expand_env(match):
            var_name = match.group(1) or match.group(2)
            return os.environ.get(var_name, match.group(0))
        expanded = re.sub(r'\$\{(\w+)\}|\$(\w+)', _expand_env, raw)
        return json.loads(expanded)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def resolve_home(path_str: str, hermes_home: str) -> str:
    if path_str.startswith("~/"):
        return str(Path.home() / path_str[2:])
    if path_str.startswith("$HERMES_HOME/"):
        return os.path.join(hermes_home, path_str[len("$HERMES_HOME/"):])
    return path_str


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("hermes-watchdog")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    if log_file:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Health Check Functions
# ---------------------------------------------------------------------------

def check_gateway_heartbeat(heartbeat_file: str, stale_secs: int) -> tuple[str, str]:
    """Check if gateway heartbeat file is fresh."""
    hf = Path(heartbeat_file).expanduser()
    if not hf.exists():
        return LEVEL_DOWN, "heartbeat file missing"
    try:
        data = json.loads(hf.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return LEVEL_DOWN, f"heartbeat file unreadable: {e}"
    updated_at = data.get("updated_at")
    if not updated_at:
        return LEVEL_DOWN, "heartbeat file has no updated_at"
    try:
        last_update = datetime.fromisoformat(updated_at)
        now = datetime.now(timezone.utc)
        age_secs = (now - last_update).total_seconds()
    except (ValueError, TypeError):
        return LEVEL_UNKNOWN, f"heartbeat timestamp unparseable: {updated_at}"
    if age_secs > stale_secs:
        return LEVEL_DOWN, f"heartbeat stale for {int(age_secs)}s (threshold {stale_secs}s)"
    pid = data.get("pid", "?")
    return LEVEL_OK, f"heartbeat fresh (age {int(age_secs)}s, pid {pid})"


def check_gateway_health(url: str, timeout: int) -> tuple[str, str]:
    """Check gateway HTTP health endpoint."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
            status = body.get("status", "unknown")
            version = body.get("version", "?")
            if status in ("ok", "degraded"):
                level = LEVEL_OK if status == "ok" else LEVEL_DEGRADED
                return level, f"gateway {status} (v{version})"
            return LEVEL_DOWN, f"gateway status={status}"
    except urllib.error.HTTPError as e:
        return LEVEL_DOWN, f"gateway HTTP {e.code}"
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        return LEVEL_DOWN, f"gateway unreachable: {e}"


def check_vps_ssh(config: dict) -> tuple[str, str]:
    """Check VPS reachability via SSH."""
    host = config.get("ssh_host", "")
    user = config.get("ssh_user", "")
    key = config.get("ssh_key", "")
    timeout = config.get("ssh_timeout_secs", 10)

    if not host:
        return LEVEL_UNKNOWN, "vps ssh_host not configured"

    ssh_cmd = _build_ssh_cmd(host, user, key, timeout)
    ssh_cmd.append("echo ok")

    try:
        result = subprocess.run(
            ssh_cmd, capture_output=True, text=True, timeout=timeout + 5
        )
        if result.returncode == 0 and "ok" in result.stdout:
            return LEVEL_OK, f"vps reachable ({host})"
        return LEVEL_DOWN, f"vps ssh failed (rc={result.returncode})"
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return LEVEL_DOWN, f"vps ssh error: {e}"


def check_vps_memorycore(config: dict) -> tuple[str, str]:
    """Check MemoryCore on VPS via SSH + curl (no local port)."""
    host = config.get("ssh_host", "")
    user = config.get("ssh_user", "")
    key = config.get("ssh_key", "")
    timeout = config.get("ssh_timeout_secs", 10)
    mc_url = config.get("memorycore_health_url", "http://127.0.0.1:8420/health")
    mc_timeout = config.get("memorycore_health_timeout_secs", 5)

    if not host:
        return LEVEL_UNKNOWN, "vps ssh_host not configured"

    ssh_cmd = _build_ssh_cmd(host, user, key, timeout)
    ssh_cmd.extend(["curl", "-sf", "-m", str(mc_timeout), mc_url])

    try:
        result = subprocess.run(
            ssh_cmd, capture_output=True, text=True, timeout=timeout + mc_timeout + 5
        )
        if result.returncode == 0:
            try:
                body = json.loads(result.stdout)
                status = body.get("status", "unknown")
                if status in ("ok", "degraded"):
                    return LEVEL_OK, f"vps memorycore {status}"
                return LEVEL_DOWN, f"vps memorycore status={status}"
            except json.JSONDecodeError:
                return LEVEL_DOWN, "vps memorycore response not JSON"
        return LEVEL_DOWN, f"vps memorycore check failed (rc={result.returncode})"
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return LEVEL_DOWN, f"vps memorycore check error: {e}"


def check_hcnsc_passive(config: dict) -> tuple[str, str]:
    """Check HCNSC reachability via passive endpoint only.

    Uses GET /api/nous/recommended-models — public, no auth, no inference.
    This is a REACHABILITY check only. It does NOT test actual inference.
    Real inference health comes from Hermes/MemoryCore event logs.
    """
    url = config.get("passive_health_url", "")
    timeout = config.get("passive_health_timeout_secs", 10)

    if not url:
        return LEVEL_UNKNOWN, "hcnsc passive_health_url not configured"

    # Safety: never send inference requests from the watchdog
    for pattern in FORBIDDEN_URL_PATTERNS:
        if pattern in url:
            return LEVEL_UNKNOWN, f"hcnsc url contains forbidden pattern: {pattern}"

    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if 200 <= resp.status < 400:
                return LEVEL_OK, f"hcnsc reachable (HTTP {resp.status})"
            return LEVEL_DEGRADED, f"hcnsc HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        if e.code >= 500:
            return LEVEL_DEGRADED, f"hcnsc server error HTTP {e.code}"
        return LEVEL_DOWN, f"hcnsc HTTP {e.code}"
    except (urllib.error.URLError, OSError) as e:
        return LEVEL_DOWN, f"hcnsc unreachable: {e}"


def _build_ssh_cmd(host: str, user: str, key: str, timeout: int) -> list[str]:
    """Build SSH command with common options."""
    cmd = [
        "ssh",
        "-o", f"ConnectTimeout={timeout}",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
    ]
    if key:
        cmd.extend(["-i", key])
    target = f"{user}@{host}" if user else host
    cmd.append(target)
    return cmd


# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------

def load_state(path: str) -> dict[str, Any]:
    state_path = Path(path).expanduser()
    if not state_path.exists():
        return {"components": {}, "alerts_sent": [], "last_run": None}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"components": {}, "alerts_sent": [], "last_run": None}


def save_state(state: dict, path: str) -> None:
    state_path = Path(path).expanduser()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def update_component_state(
    state: dict, component: str, level: str, detail: str, now: datetime
) -> dict[str, Any]:
    comps = state.setdefault("components", {})
    comp = comps.setdefault(component, {
        "level": LEVEL_UNKNOWN,
        "detail": "",
        "first_failure": None,
        "last_ok": None,
        "last_failure": None,
        "consecutive_failures": 0,
    })
    prev_level = comp.get("level", LEVEL_UNKNOWN)

    if level in (LEVEL_OK, LEVEL_DEGRADED):
        comp["last_ok"] = now.isoformat()
        comp["level"] = level
        comp["detail"] = detail
        comp["consecutive_failures"] = 0
        if prev_level in (LEVEL_DOWN, LEVEL_UNKNOWN):
            comp["first_failure"] = None
    else:
        comp["last_failure"] = now.isoformat()
        comp["level"] = level
        comp["detail"] = detail
        comp["consecutive_failures"] = comp.get("consecutive_failures", 0) + 1
        if comp["first_failure"] is None:
            comp["first_failure"] = now.isoformat()
    return comp


# ---------------------------------------------------------------------------
# Alert Sending
# ---------------------------------------------------------------------------

def send_ntfy(server, topic, title, message, tags="warning", token="", logger=None):
    url = f"{server.rstrip('/')}/{topic}"
    headers = {"Content-Type": "application/json", "Tags": tags, "Title": title}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = json.dumps({"topic": topic, "message": message, "title": title, "tags": tags}).encode()
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status in (200, 201):
                if logger:
                    logger.info("ntfy alert sent: %s", title)
                return True
        return False
    except (urllib.error.URLError, OSError) as e:
        if logger:
            logger.error("ntfy send failed: %s", e)
        return False


def send_telegram(bot_token, chat_id, text, logger=None):
    if not bot_token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError) as e:
        if logger:
            logger.error("telegram send failed: %s", e)
        return False


def send_slack(webhook_url, title, body, logger=None):
    """Send alert via Slack Incoming Webhook.

    The webhook URL contains the secret token. Never log the URL.
    Slack failure must not break the watchdog.
    """
    if not webhook_url:
        return False
    payload = json.dumps({"text": f"*{title}*\n\n{body}"}).encode()
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status in (200, 201):
                if logger:
                    logger.info("slack alert sent: %s", title)
                return True
        return False
    except (urllib.error.URLError, OSError) as e:
        if logger:
            logger.error("slack send failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Alert Formatting & Dedup
# ---------------------------------------------------------------------------

def format_alert(component, alert_type, detail, level, duration="", now=None):
    ts = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M:%S UTC")
    comp = component.lower()

    if alert_type == ALERT_RECOVERY:
        title = f"{comp} recovered"
        body = f"{comp} recovered after {duration}.\nTime: {ts}"
    elif alert_type == ALERT_DEGRADED:
        title = f"{comp} degraded"
        body = f"{detail}\nTime: {ts}"
        if duration:
            body += f"\nDuration: {duration}"
    else:
        title = f"{comp} down"
        body = f"{detail}\nTime: {ts}"
        if duration:
            body += f"\nDuration: {duration}"
    return title, body


def should_alert(state, component, alert_type, cooldown, max_per_hour, now):
    alerts = state.setdefault("alerts_sent", [])
    cutoff = now.timestamp() - 3600
    recent = [a for a in alerts if a.get("component") == component and a.get("type") == alert_type and a.get("time", 0) > cutoff]
    if len(recent) >= max_per_hour:
        return False
    if recent:
        last_time = max(a.get("time", 0) for a in recent)
        if now.timestamp() - last_time < cooldown:
            return False
    return True


def record_alert(state, component, alert_type, now):
    alerts = state.setdefault("alerts_sent", [])
    alerts.append({"component": component, "type": alert_type, "time": now.timestamp()})
    cutoff = now.timestamp() - 7200
    state["alerts_sent"] = [a for a in alerts if a.get("time", 0) > cutoff]


# ---------------------------------------------------------------------------
# Composite State Evaluation
# ---------------------------------------------------------------------------

def _classify_composite(results):
    """Classify check results into distinct incident types.

    Returns list of (component, alert_type, detail) tuples.
    Does NOT collapse everything into generic "system down".
    """
    gw_hb = results.get("gateway_heartbeat", (LEVEL_UNKNOWN, ""))
    gw_hp = results.get("gateway_health", (LEVEL_UNKNOWN, ""))
    vps = results.get("vps", (LEVEL_UNKNOWN, ""))
    vps_mc = results.get("vps_memorycore", (LEVEL_UNKNOWN, ""))
    hcnsc = results.get("hcnsc", (LEVEL_UNKNOWN, ""))

    incidents = []

    # Gateway (local Mac)
    if gw_hb[0] == LEVEL_DOWN or gw_hp[0] == LEVEL_DOWN:
        detail = f"heartbeat: {gw_hb[1]}; health: {gw_hp[1]}"
        incidents.append(("gateway", ALERT_DOWN, detail))
    elif gw_hb[0] == LEVEL_OK or gw_hp[0] == LEVEL_OK:
        pass  # gateway healthy

    # VPS
    if vps[0] == LEVEL_DOWN:
        incidents.append(("vps", ALERT_DOWN, vps[1]))
    elif vps[0] == LEVEL_OK:
        # VPS reachable — check MemoryCore on it
        if vps_mc[0] == LEVEL_DOWN:
            incidents.append(("memorycore", ALERT_DOWN, f"VPS reachable; {vps_mc[1]}"))
        elif vps_mc[0] == LEVEL_DEGRADED:
            incidents.append(("memorycore", ALERT_DEGRADED, f"VPS reachable; {vps_mc[1]}"))
        elif vps_mc[0] == LEVEL_OK:
            pass  # memorycore healthy

    # HCNSC (passive reachability only)
    if hcnsc[0] == LEVEL_DOWN:
        detail = f"HCNSC unreachable; real inference health comes from Hermes events — {hcnsc[1]}"
        incidents.append(("hcnsc", ALERT_DOWN, detail))
    elif hcnsc[0] == LEVEL_DEGRADED:
        detail = f"HCNSC degraded; real inference health comes from Hermes events — {hcnsc[1]}"
        incidents.append(("hcnsc", ALERT_DEGRADED, detail))

    return incidents


def _compute_duration(comp_state, now):
    first = comp_state.get("first_failure")
    if not first:
        return ""
    try:
        first_dt = datetime.fromisoformat(first)
        delta = (now - first_dt).total_seconds()
        if delta < 60:
            return f"{int(delta)}s"
        if delta < 3600:
            return f"{int(delta // 60)}m {int(delta % 60)}s"
        return f"{int(delta // 3600)}h {int((delta % 3600) // 60)}m"
    except (ValueError, TypeError):
        return ""


def _send_alerts(ntfy_cfg, tg_cfg, slack_cfg, title, body, tags, logger, dry_run=False):
    if dry_run:
        logger.info("DRY RUN alert: %s\n%s", title, body)
        return
    if ntfy_cfg.get("enabled"):
        send_ntfy(
            server=ntfy_cfg.get("server", "https://ntfy.sh"),
            topic=ntfy_cfg.get("topic", "hermes-watchdog-alerts"),
            title=title, message=body, tags=tags,
            token=ntfy_cfg.get("token", ""), logger=logger,
        )
    if tg_cfg.get("enabled"):
        send_telegram(
            bot_token=tg_cfg.get("bot_token", ""),
            chat_id=tg_cfg.get("chat_id", ""),
            text=f"<b>{title}</b>\n\n{body}", logger=logger,
        )
    if slack_cfg.get("enabled"):
        send_slack(
            webhook_url=slack_cfg.get("webhook_url", ""),
            title=title,
            body=body,
            logger=logger,
        )


def evaluate_and_alert(results, state, config, now, logger, dry_run=False):
    """Evaluate check results and send alerts. Returns list of alert titles sent."""
    alerts_cfg = config.get("alerts", {})
    cooldown = alerts_cfg.get("cooldown_secs", 300)
    max_per_hour = alerts_cfg.get("max_per_hour", 6)
    ntfy_cfg = alerts_cfg.get("ntfy", {})
    tg_cfg = alerts_cfg.get("telegram", {})
    slack_cfg = alerts_cfg.get("slack", {})

    sent = []
    incidents = _classify_composite(results)

    # Capture previous state BEFORE updates (for recovery detection)
    prev_states = {}
    for comp_name in ("gateway", "vps", "memorycore", "hcnsc"):
        comp_state = state.get("components", {}).get(comp_name, {})
        prev_states[comp_name] = {
            "level": comp_state.get("level", LEVEL_UNKNOWN),
            "first_failure": comp_state.get("first_failure"),
        }

    # Track all components in state (gateway handled separately below)
    for comp_name in ("vps", "memorycore", "hcnsc"):
        level, detail = results.get(comp_name, (LEVEL_UNKNOWN, ""))
        update_component_state(state, comp_name, level, detail, now)

    # Handle gateway separately (local Mac, not composite)
    gw_hb = results.get("gateway_heartbeat", (LEVEL_UNKNOWN, ""))
    gw_hp = results.get("gateway_health", (LEVEL_UNKNOWN, ""))
    gw_level = LEVEL_DOWN if gw_hb[0] == LEVEL_DOWN or gw_hp[0] == LEVEL_DOWN else LEVEL_OK
    gw_detail = f"heartbeat: {gw_hb[1]}; health: {gw_hp[1]}"

    if gw_level == LEVEL_DOWN:
        comp_state = update_component_state(state, "gateway", gw_level, gw_detail, now)
        duration = _compute_duration(comp_state, now)
        if should_alert(state, "gateway", ALERT_DOWN, cooldown, max_per_hour, now):
            title, body = format_alert("gateway", ALERT_DOWN, gw_detail, gw_level, duration, now)
            _send_alerts(ntfy_cfg, tg_cfg, slack_cfg, title, body, "rotating_light", logger, dry_run)
            record_alert(state, "gateway", ALERT_DOWN, now)
            sent.append(title)
    else:
        # Check for gateway recovery using captured previous state
        prev_gw = prev_states.get("gateway", {})
        if prev_gw.get("level") == LEVEL_DOWN and prev_gw.get("first_failure"):
            comp_state = state.get("components", {}).get("gateway", {})
            duration = _compute_duration(comp_state, now)
            if should_alert(state, "gateway", ALERT_RECOVERY, cooldown, max_per_hour, now):
                title, body = format_alert("gateway", ALERT_RECOVERY, gw_detail, gw_level, duration, now)
                _send_alerts(ntfy_cfg, tg_cfg, slack_cfg, title, body, "white_check_mark", logger, dry_run)
                record_alert(state, "gateway", ALERT_RECOVERY, now)
                sent.append(title)
        update_component_state(state, "gateway", gw_level, gw_detail, now)

    # Handle other incidents from composite classification
    for comp_name, alert_type, detail in incidents:
        if comp_name == "gateway":
            continue  # already handled above

        if alert_type == ALERT_DOWN:
            comp_state = update_component_state(state, comp_name, LEVEL_DOWN, detail, now)
            duration = _compute_duration(comp_state, now)
            if should_alert(state, comp_name, ALERT_DOWN, cooldown, max_per_hour, now):
                title, body = format_alert(comp_name, ALERT_DOWN, detail, LEVEL_DOWN, duration, now)
                _send_alerts(ntfy_cfg, tg_cfg, slack_cfg, title, body, "rotating_light", logger, dry_run)
                record_alert(state, comp_name, ALERT_DOWN, now)
                sent.append(title)
        elif alert_type == ALERT_DEGRADED:
            comp_state = update_component_state(state, comp_name, LEVEL_DEGRADED, detail, now)
            duration = _compute_duration(comp_state, now)
            if should_alert(state, comp_name, ALERT_DEGRADED, cooldown, max_per_hour, now):
                title, body = format_alert(comp_name, ALERT_DEGRADED, detail, LEVEL_DEGRADED, duration, now)
                _send_alerts(ntfy_cfg, tg_cfg, slack_cfg, title, body, "warning", logger, dry_run)
                record_alert(state, comp_name, ALERT_DEGRADED, now)
                sent.append(title)

    # Check for recoveries (after all state updates, using captured previous state)
    for comp_name in ("vps", "memorycore", "hcnsc"):
        prev_comp = prev_states.get(comp_name, {})
        curr_comp = state.get("components", {}).get(comp_name, {})
        if curr_comp.get("level") == LEVEL_OK and prev_comp.get("level") in (LEVEL_DOWN, LEVEL_DEGRADED) and prev_comp.get("first_failure"):
            duration = _compute_duration(curr_comp, now)
            if should_alert(state, comp_name, ALERT_RECOVERY, cooldown, max_per_hour, now):
                title, body = format_alert(comp_name, ALERT_RECOVERY, curr_comp.get("detail", ""), LEVEL_OK, duration, now)
                _send_alerts(ntfy_cfg, tg_cfg, slack_cfg, title, body, "white_check_mark", logger, dry_run)
                record_alert(state, comp_name, ALERT_RECOVERY, now)
                sent.append(title)

    return sent


# ---------------------------------------------------------------------------
# Main Check Logic
# ---------------------------------------------------------------------------

def run_checks(config, logger):
    """Run all health checks. Returns {component: (level, detail)}."""
    results = {}
    gw = config.get("gateway", {})
    vps = config.get("vps", {})
    hcnsc = config.get("hcnsc", {})
    hermes_home = config.get("hermes_home", "~/.hermes")

    # 1. Gateway heartbeat (local)
    hb_file = resolve_home(gw.get("heartbeat_file", "~/.hermes/state/gateway.heartbeat"), hermes_home)
    results["gateway_heartbeat"] = check_gateway_heartbeat(hb_file, gw.get("heartbeat_stale_secs", 90))

    # 2. Gateway HTTP health (local)
    results["gateway_health"] = check_gateway_health(
        gw.get("health_url", "http://127.0.0.1:8642/health"),
        gw.get("health_timeout_secs", 5),
    )

    # 3. VPS reachability (SSH)
    results["vps"] = check_vps_ssh(vps)

    # 4. MemoryCore on VPS (SSH + curl to remote /health)
    if vps.get("ssh_host"):
        results["vps_memorycore"] = check_vps_memorycore(vps)

    # 5. HCNSC passive reachability (no inference)
    results["hcnsc"] = check_hcnsc_passive(hcnsc)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hermes External Watchdog")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inject", type=str, choices=("gateway", "vps", "memorycore", "hcnsc"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if not config:
        print("ERROR: config not found", file=sys.stderr)
        sys.exit(1)

    state_path = resolve_home(config.get("state_file", "~/.hermes/watchdog/state.json"), config.get("hermes_home", "~/.hermes"))
    log_file = resolve_home(config.get("log_file", "~/.hermes/logs/watchdog.log"), config.get("hermes_home", "~/.hermes"))

    logger = setup_logging(log_file)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logger.info("watchdog starting (pid=%d)", os.getpid())

    if args.loop:
        while True:
            try:
                _run_once(config, state_path, logger, args)
            except KeyboardInterrupt:
                break
            except Exception:
                logger.exception("watchdog error")
            time.sleep(args.interval)
    else:
        _run_once(config, state_path, logger, args)


def _run_once(config, state_path, logger, args):
    now = datetime.now(timezone.utc)
    state = load_state(state_path)
    logger.info("--- check run at %s ---", now.isoformat())

    results = run_checks(config, logger)

    # Failure injection (for testing)
    if args.inject:
        comp = args.inject
        if comp == "gateway":
            results["gateway_heartbeat"] = (LEVEL_DOWN, "injected failure")
            results["gateway_health"] = (LEVEL_DOWN, "injected failure")
        elif comp == "vps":
            results["vps"] = (LEVEL_DOWN, "injected failure")
        elif comp == "memorycore":
            results["vps_memorycore"] = (LEVEL_DOWN, "injected failure")
        elif comp == "hcnsc":
            results["hcnsc"] = (LEVEL_DOWN, "injected failure")

    for name, (level, detail) in sorted(results.items()):
        icon = "✓" if level == LEVEL_OK else "✗" if level == LEVEL_DOWN else "~" if level == LEVEL_DEGRADED else "?"
        logger.info("  %s %s: %s — %s", icon, name, level, detail)

    if args.check_only:
        print(json.dumps({k: {"level": v[0], "detail": v[1]} for k, v in results.items()}, indent=2))
        return

    sent = evaluate_and_alert(results, state, config, now, logger, dry_run=args.dry_run)
    if sent:
        logger.info("alerts sent: %s", sent)
    else:
        logger.info("no alerts sent")

    state["last_run"] = now.isoformat()
    save_state(state, state_path)


if __name__ == "__main__":
    main()
