"""Hook handlers for the hermes_health plugin.

Each function is registered as a lifecycle hook observer. They receive
only the kwargs they declare (signature-inspected by the plugin dispatch).

Severity policy:
  - All events are recorded internally for diagnostics.
  - Alerts fire ONLY on health-state transitions (healthy→degraded→error→healthy).
  - Single failures are NORMAL (no alert). Repeated failures cross thresholds
    that trigger state transitions, which then alert.
  - No synthetic LLM calls are ever made.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from .state import HealthState, HealthStatus

logger = logging.getLogger(__name__)

# Module-level state, initialized by register()
_state: Optional[HealthState] = None
_notifier: Any = None  # AlertNotifier, set by __init__.py


def init(state: HealthState, notifier: Any) -> None:
    global _state, _notifier
    _state = state
    _notifier = notifier
    # Transition wiring is handled by __init__.py register() — not here,
    # to avoid duplicate callbacks when init() is called multiple times.


# -- LLM lifecycle -----------------------------------------------------------

def on_pre_api_request(
    *,
    task_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    session_id: str = "",
    model: str = "",
    provider: str = "",
    base_url: str = "",
    api_call_count: int = 0,
    retry_count: int = 0,
    approx_input_tokens: int = 0,
    **_: Any,
) -> None:
    if _state is None:
        return
    _state.record_event(
        "llm_request",
        success=True,
        provider=provider,
        model=model,
        session_id=session_id,
        extra={
            "api_call_count": api_call_count,
            "retry_count": retry_count,
            "approx_input_tokens": approx_input_tokens,
            "api_request_id": api_request_id,
        },
    )


def on_post_api_request(
    *,
    task_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    session_id: str = "",
    model: str = "",
    provider: str = "",
    base_url: str = "",
    api_call_count: int = 0,
    api_duration: float = 0.0,
    finish_reason: str = "",
    usage: Any = None,
    response_model: str = "",
    **_: Any,
) -> None:
    if _state is None:
        return
    duration_ms = api_duration * 1000 if api_duration < 1000 else api_duration
    _state.record_event(
        "llm_response",
        success=True,
        provider=provider,
        model=model,
        duration_ms=duration_ms,
        session_id=session_id,
        extra={
            "finish_reason": finish_reason,
            "response_model": response_model,
            "api_request_id": api_request_id,
        },
    )
    _state.last_llm_success_time = time.monotonic()


def on_api_request_error(
    *,
    task_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    session_id: str = "",
    model: str = "",
    provider: str = "",
    base_url: str = "",
    api_call_count: int = 0,
    api_duration: float = 0.0,
    status_code: int = 0,
    retry_count: int = 0,
    max_retries: int = 0,
    retryable: bool = False,
    reason: str = "",
    error: Any = None,
    **_: Any,
) -> None:
    if _state is None:
        return
    err = error if isinstance(error, dict) else {"type": "unknown", "message": str(error)}
    duration_ms = api_duration * 1000 if api_duration < 1000 else api_duration
    _state.record_event(
        "llm_error",
        success=False,
        provider=provider,
        model=model,
        duration_ms=duration_ms,
        error_type=err.get("type", "unknown"),
        error_message=err.get("message", reason),
        session_id=session_id,
        extra={
            "status_code": status_code,
            "retryable": retryable,
            "retry_count": retry_count,
            "max_retries": max_retries,
        },
    )
    # No per-failure alert — state transition handles it


def on_pre_llm_call(
    *,
    session_id: str = "",
    model: str = "",
    platform: str = "",
    **_: Any,
) -> None:
    pass  # Covered by pre_api_request


def on_post_llm_call(
    *,
    session_id: str = "",
    model: str = "",
    platform: str = "",
    assistant_response: str = "",
    **_: Any,
) -> None:
    pass  # Covered by post_api_request


# -- Tool lifecycle -----------------------------------------------------------

def on_pre_tool_call(
    *,
    tool_name: str = "",
    args: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    **_: Any,
) -> None:
    pass  # State tracked at post_tool_call for timing


def on_post_tool_call(
    *,
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    duration_ms: int = 0,
    status: str = "ok",
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    **_: Any,
) -> None:
    if _state is None:
        return
    success = status == "ok"
    _state.record_event(
        "tool_call",
        success=success,
        tool_name=tool_name,
        duration_ms=float(duration_ms),
        error_type=error_type,
        error_message=error_message,
        session_id=session_id,
        extra={"args_summary": _summarize_args(tool_name, args)},
    )

    # Computer-use failure tracking
    if tool_name == "computer_use" and not success:
        action = ""
        if isinstance(args, dict):
            action = args.get("action", "")
        failure = {
            "action": action,
            "status": status,
            "error_type": error_type,
            "error_message": (error_message or "")[:200],
            "duration_ms": duration_ms,
            "time": time.monotonic(),
        }
        with _state._lock:
            _state.recent_computer_use_failures.append(failure)
            _state.consecutive_computer_use_failures += 1
    elif tool_name == "computer_use" and success:
        with _state._lock:
            _state.consecutive_computer_use_failures = 0

    # No per-failure alert — state transition handles it


# -- Memory sync (ready for on_memory_sync hook) ------------------------------

def on_memory_sync(
    *,
    provider_name: str = "",
    success: bool = True,
    duration_ms: float = 0.0,
    error: Optional[str] = None,
    session_id: str = "",
    **_: Any,
) -> None:
    """Consume the on_memory_sync hook when Developer 2 adds it."""
    if _state is None:
        return
    _state.record_memory_sync(
        success=success,
        provider_name=provider_name,
        duration_ms=duration_ms,
        error=error,
        session_id=session_id,
    )


# -- Session lifecycle --------------------------------------------------------

def on_session_start(
    *,
    session_id: str = "",
    model: str = "",
    platform: str = "",
    **_: Any,
) -> None:
    if _state is None:
        return
    with _state._lock:
        _state.sessions_active += 1
        _state.sessions_total += 1
    _state.record_event("session_start", success=True, session_id=session_id)


def on_session_end(
    *,
    session_id: str = "",
    completed: bool = True,
    failed: bool = False,
    interrupted: bool = False,
    turn_exit_reason: str = "",
    **_: Any,
) -> None:
    if _state is None:
        return
    with _state._lock:
        _state.sessions_active = max(0, _state.sessions_active - 1)
    _state.record_event(
        "session_end",
        success=completed and not failed,
        session_id=session_id,
        extra={
            "completed": completed,
            "failed": failed,
            "interrupted": interrupted,
            "turn_exit_reason": turn_exit_reason,
        },
    )
    # Single session failure is NORMAL — no alert


def on_session_finalize(
    *,
    session_id: str = "",
    **_: Any,
) -> None:
    pass  # Covered by on_session_end


# -- Subagent lifecycle -------------------------------------------------------

def on_subagent_start(
    *,
    parent_session_id: str = "",
    child_session_id: str = "",
    child_role: str = "",
    child_goal: str = "",
    **_: Any,
) -> None:
    if _state is None:
        return
    with _state._lock:
        _state.subagents_total += 1
    _state.record_event(
        "subagent_start",
        success=True,
        session_id=child_session_id,
        extra={"child_role": child_role, "parent_session_id": parent_session_id},
    )


def on_subagent_stop(
    *,
    parent_session_id: str = "",
    child_session_id: str = "",
    child_role: str = "",
    child_summary: str = "",
    child_status: str = "",
    duration_ms: int = 0,
    **_: Any,
) -> None:
    if _state is None:
        return
    success = child_status not in ("failed", "crashed", "error", "timeout")
    _state.record_event(
        "subagent_stop",
        success=success,
        duration_ms=float(duration_ms),
        session_id=child_session_id,
        extra={
            "child_role": child_role,
            "child_status": child_status,
            "child_summary": (child_summary or "")[:200],
        },
    )
    # Single subagent failure is NORMAL — no alert


# -- Helpers ------------------------------------------------------------------

def _summarize_args(tool_name: str, args: Any) -> str:
    if not isinstance(args, dict):
        return ""
    if tool_name == "computer_use":
        return args.get("action", "")
    if tool_name == "terminal":
        cmd = args.get("command", "")
        return cmd[:80] + "..." if len(cmd) > 80 else cmd
    if tool_name in ("read_file", "write_file", "edit_file"):
        return args.get("path", "")[:100]
    return ""
