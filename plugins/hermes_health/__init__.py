"""hermes_health — Pipeline health observer plugin.

Registers lifecycle hooks to observe LLM calls, tool execution, memory sync,
and computer-use across the Hermes → TencentDB → HCNSC stack.

Alerts fire ONLY on health-state transitions (healthy→degraded→error→healthy),
not on individual failures.

Provides:
  - `hermes health` CLI command
  - /api/plugins/hermes_health/ REST endpoint (dashboard)
  - State-based alert notification with cooldown safety net
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any, Optional

from .notifiers import AlertNotifier
from .observer import init as _init_observer
from .state import HealthState, HealthStatus

logger = logging.getLogger(__name__)

_state: Optional[HealthState] = None
_notifier: Optional[AlertNotifier] = None


def _get_state() -> HealthState:
    global _state
    if _state is None:
        _state = HealthState()
    return _state


def _get_notifier() -> AlertNotifier:
    global _notifier
    if _notifier is None:
        _notifier = AlertNotifier()
    return _notifier


# -- CLI ----------------------------------------------------------------------

def _format_health_text(data: dict) -> str:
    lines = [
        f"Status: {data['status'].upper()}",
        f"Uptime: {data['uptime_s']:.0f}s",
        f"Errors (5min): {data['errors_5min']}",
    ]
    if data["last_error"]:
        lines.append(f"Last error: {data['last_error']} ({data['last_error_age_s']:.0f}s ago)")
    if data["last_successful_llm_call_age_s"] is not None:
        lines.append(f"Last LLM success: {data['last_successful_llm_call_age_s']:.0f}s ago")
    if data["last_memory_sync_age_s"] is not None:
        mem_sync_line = f"Last memory sync: {data['last_memory_sync_age_s']:.0f}s ago"
        if data.get("consecutive_memory_sync_failures", 0) > 0:
            mem_sync_line += f" ({data['consecutive_memory_sync_failures']} consecutive failures)"
        lines.append(mem_sync_line)
    lines.append(f"Sessions: {data['sessions_active']} active / {data['sessions_total']} total")
    lines.append(f"Subagents: {data['subagents_total']} total")

    if data["providers"]:
        lines.append("")
        lines.append("Providers:")
        for name, stats in data["providers"].items():
            lines.append(f"  {name}: {stats['total_calls']} calls, {stats['errors']} errors, "
                         f"avg {stats['avg_duration_ms']:.0f}ms")

    if data["tools"]:
        lines.append("")
        lines.append("Tools:")
        for name, stats in data["tools"].items():
            lines.append(f"  {name}: {stats['total_calls']} calls, {stats['errors']} errors, "
                         f"avg {stats['avg_duration_ms']:.0f}ms")

    cu_failures = data.get("recent_computer_use_failures", [])
    cu_consecutive = data.get("consecutive_computer_use_failures", 0)
    if cu_failures:
        lines.append("")
        lines.append(f"Recent computer-use failures: {len(cu_failures)}")
        if cu_consecutive > 0:
            lines.append(f"  Consecutive failures: {cu_consecutive}")
        for f in cu_failures[-3:]:
            lines.append(f"  {f['action']}: {f['error_type']} - {f['error_message'][:80]}")

    if data["recent_events"]:
        lines.append("")
        lines.append(f"Recent events ({len(data['recent_events'])}):")
        for ev in data["recent_events"][-5:]:
            icon = "+" if ev.get("success", True) else "-"
            kind = ev.get("kind", "?")
            extra = ev.get("tool_name") or ev.get("provider") or ""
            lines.append(f"  [{icon}] {kind} {extra}")

    return "\n".join(lines)


def _health_command(args: argparse.Namespace) -> int:
    state = _get_state()
    data = state.to_dict()
    fmt = getattr(args, "format", "text")
    if fmt == "json":
        print(json.dumps(data, indent=2))
    else:
        print(_format_health_text(data))
    return 0


def _register_cli(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--json", "-j", dest="format", action="store_const", const="json",
        default="text", help="Output as JSON",
    )
    subparser.set_defaults(func=_health_command)


# -- Plugin registration ------------------------------------------------------

def register(ctx: Any) -> None:
    state = _get_state()
    notifier = _get_notifier()

    # Wire state transitions to notifier
    state.on_transition(notifier.on_transition)

    # Register Slack notification callback (optional, env-gated)
    from .notifiers_slack import make_slack_callback
    slack_cb = make_slack_callback()
    if slack_cb is not None:
        notifier.add_callback(slack_cb)
        logger.info("hermes_health: Slack notifications enabled")

    # Initialize observer module
    _init_observer(state, notifier)

    from . import observer

    hooks = [
        ("pre_api_request", observer.on_pre_api_request),
        ("post_api_request", observer.on_post_api_request),
        ("api_request_error", observer.on_api_request_error),
        ("pre_llm_call", observer.on_pre_llm_call),
        ("post_llm_call", observer.on_post_llm_call),
        ("pre_tool_call", observer.on_pre_tool_call),
        ("post_tool_call", observer.on_post_tool_call),
        ("on_session_start", observer.on_session_start),
        ("on_session_end", observer.on_session_end),
        ("on_session_finalize", observer.on_session_finalize),
        ("subagent_start", observer.on_subagent_start),
        ("subagent_stop", observer.on_subagent_stop),
        ("on_memory_sync", observer.on_memory_sync),
    ]
    for name, fn in hooks:
        ctx.register_hook(name, fn)

    ctx.register_cli_command(
        name="health",
        help="Show pipeline health status",
        setup_fn=_register_cli,
        handler_fn=_health_command,
        description="Display the current health state of the Hermes pipeline observer.",
    )

    ctx.on_unload(lambda: None)

    logger.info("hermes_health plugin registered (12 hooks, CLI: hermes health)")
