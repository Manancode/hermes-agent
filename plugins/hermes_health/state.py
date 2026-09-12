"""Health state tracking for the hermes_health plugin.

Maintains a sliding-window event log, per-provider and per-tool counters,
and computes overall health status. Alerts fire ONLY on state transitions
(healthy→degraded, degraded→error, any→healthy), not on individual failures.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ERROR = "error"


# Event severity classification
# CRITICAL: fires alert on state transition
# NORMAL: recorded internally, no immediate alert
class Severity(str, Enum):
    CRITICAL = "critical"
    NORMAL = "normal"


# Which event kinds are CRITICAL vs NORMAL
_SEVERITY_MAP: Dict[str, Severity] = {
    # LLM — single failure is NORMAL, repeated is CRITICAL (via state transition)
    "llm_request": Severity.NORMAL,
    "llm_response": Severity.NORMAL,
    "llm_error": Severity.NORMAL,
    # Tools — single failure is NORMAL
    "tool_call": Severity.NORMAL,
    # Sessions — single unexpected end is NORMAL
    "session_start": Severity.NORMAL,
    "session_end": Severity.NORMAL,
    # Subagents
    "subagent_start": Severity.NORMAL,
    "subagent_stop": Severity.NORMAL,
    # Memory sync
    "memory_sync": Severity.NORMAL,
}

# Events that are NEVER alert-worthy (always NORMAL regardless of count)
_NO_ALERT_EVENTS: set = {"llm_request", "llm_response", "session_start", "subagent_start"}


@dataclass
class ProviderStats:
    total_calls: int = 0
    errors: int = 0
    consecutive_errors: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None
    last_success_time: Optional[float] = None
    avg_duration_ms: float = 0.0
    _duration_sum: float = 0.0

    def record_success(self, duration_ms: float) -> None:
        self.total_calls += 1
        self.consecutive_errors = 0
        self.last_success_time = time.monotonic()
        self._duration_sum += duration_ms
        self.avg_duration_ms = self._duration_sum / self.total_calls

    def record_error(self, error_type: str) -> None:
        self.total_calls += 1
        self.errors += 1
        self.consecutive_errors += 1
        self.last_error = error_type
        self.last_error_time = time.monotonic()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "errors": self.errors,
            "consecutive_errors": self.consecutive_errors,
            "error_rate": round(self.errors / self.total_calls, 4) if self.total_calls else 0.0,
            "last_error": self.last_error,
            "avg_duration_ms": round(self.avg_duration_ms, 1),
        }


@dataclass
class ToolStats:
    total_calls: int = 0
    errors: int = 0
    consecutive_errors: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None
    last_success_time: Optional[float] = None
    avg_duration_ms: float = 0.0
    _duration_sum: float = 0.0

    def record_success(self, duration_ms: float) -> None:
        self.total_calls += 1
        self.consecutive_errors = 0
        self.last_success_time = time.monotonic()
        self._duration_sum += duration_ms
        self.avg_duration_ms = self._duration_sum / self.total_calls

    def record_error(self, error_type: str) -> None:
        self.total_calls += 1
        self.errors += 1
        self.consecutive_errors += 1
        self.last_error = error_type
        self.last_error_time = time.monotonic()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "errors": self.errors,
            "consecutive_errors": self.consecutive_errors,
            "error_rate": round(self.errors / self.total_calls, 4) if self.total_calls else 0.0,
            "last_error": self.last_error,
            "avg_duration_ms": round(self.avg_duration_ms, 1),
        }


@dataclass
class HealthState:
    """Thread-safe health state with sliding-window error tracking.

    Alerting is STATE-BASED: alerts fire only on status transitions,
    not on individual failures. The transition callback receives
    (old_status, new_status, message).
    """

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _events: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=200))
    _errors_5min: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    _providers: Dict[str, ProviderStats] = field(default_factory=dict)
    _tools: Dict[str, ToolStats] = field(default_factory=dict)
    _transition_callbacks: List[Callable[[HealthStatus, HealthStatus, str], None]] = field(
        default_factory=list, repr=False
    )

    # Current and previous status for transition detection
    _current_status: HealthStatus = HealthStatus.HEALTHY
    _previous_status: HealthStatus = HealthStatus.HEALTHY

    last_error: Optional[str] = None
    last_error_time: Optional[float] = None
    last_llm_success_time: Optional[float] = None
    last_memory_sync_time: Optional[float] = None
    last_memory_sync_error: Optional[str] = None
    consecutive_memory_sync_failures: int = 0
    recent_computer_use_failures: Deque[Dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=50)
    )
    consecutive_computer_use_failures: int = 0
    sessions_active: int = 0
    sessions_total: int = 0
    subagents_total: int = 0
    started_at: float = field(default_factory=time.monotonic)

    # Thresholds
    ERROR_WINDOW_SECS: float = 300.0
    DEGRADED_THRESHOLD: int = 5
    ERROR_THRESHOLD: int = 15
    # Consecutive failure thresholds for CRITICAL provider/tool alerts
    PROVIDER_CRITICAL_CONSECUTIVE: int = 5
    TOOL_CRITICAL_CONSECUTIVE: int = 3
    MEMORY_SYNC_CRITICAL_CONSECUTIVE: int = 3
    CU_CRITICAL_CONSECUTIVE: int = 3

    def on_transition(self, cb: Callable[[HealthStatus, HealthStatus, str], None]) -> None:
        """Register a callback for state transitions: (old, new, message)."""
        self._transition_callbacks.append(cb)

    def record_event(
        self,
        kind: str,
        *,
        success: bool = True,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        tool_name: Optional[str] = None,
        duration_ms: Optional[float] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        action: Optional[str] = None,
        session_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = time.monotonic()
        event: Dict[str, Any] = {
            "kind": kind,
            "success": success,
            "time": now,
        }
        if provider:
            event["provider"] = provider
        if model:
            event["model"] = model
        if tool_name:
            event["tool_name"] = tool_name
        if duration_ms is not None:
            event["duration_ms"] = round(duration_ms, 1)
        if error_type:
            event["error_type"] = error_type
        if error_message:
            event["error_message"] = error_message[:200]
        if action:
            event["action"] = action
        if session_id:
            event["session_id"] = session_id
        if extra:
            event.update(extra)

        try:
            with self._lock:
                self._events.append(event)

                if not success:
                    self._errors_5min.append(now)
                    self.last_error = error_message or error_type or kind
                    self.last_error_time = now

                if provider:
                    stats = self._providers.setdefault(provider, ProviderStats())
                    if success and duration_ms is not None:
                        stats.record_success(duration_ms)
                    elif not success:
                        stats.record_error(error_type or "unknown")

                if tool_name:
                    tstats = self._tools.setdefault(tool_name, ToolStats())
                    if success and duration_ms is not None:
                        tstats.record_success(duration_ms)
                    elif not success:
                        tstats.record_error(error_type or "unknown")
        except Exception:
            pass  # Never let recording failure break the agent

        # Check for state transition (outside lock to avoid deadlock in callbacks)
        try:
            self._check_transition()
        except Exception:
            pass

    def record_memory_sync(
        self,
        *,
        success: bool,
        provider_name: str = "",
        duration_ms: float = 0.0,
        error: Optional[str] = None,
        session_id: str = "",
    ) -> None:
        """Record a memory sync event. Called by on_memory_sync hook."""
        now = time.monotonic()
        event: Dict[str, Any] = {
            "kind": "memory_sync",
            "success": success,
            "time": now,
            "provider_name": provider_name,
            "duration_ms": round(duration_ms, 1),
            "session_id": session_id,
        }
        if error:
            event["error_type"] = error[:200]

        with self._lock:
            self._events.append(event)
            self.last_memory_sync_time = now

            if success:
                self.consecutive_memory_sync_failures = 0
                self.last_memory_sync_error = None
            else:
                self.consecutive_memory_sync_failures += 1
                self.last_memory_sync_error = error
                self._errors_5min.append(now)
                self.last_error = error or "memory_sync_failure"
                self.last_error_time = now

        self._check_transition()

    def _check_transition(self) -> None:
        """Evaluate current status and fire transition callbacks if changed."""
        new_status = self._compute_status()
        with self._lock:
            old_status = self._current_status
            if new_status == old_status:
                return
            self._previous_status = old_status
            self._current_status = new_status

        # Build transition message
        if new_status == HealthStatus.HEALTHY:
            msg = "Pipeline recovered to healthy"
        elif new_status == HealthStatus.DEGRADED:
            count = self.get_error_count_5min()
            msg = f"Pipeline degraded — {count} errors in {self.ERROR_WINDOW_SECS:.0f}s window"
        else:
            count = self.get_error_count_5min()
            msg = f"Pipeline ERROR — {count} errors in {self.ERROR_WINDOW_SECS:.0f}s window"

        for cb in self._transition_callbacks:
            try:
                cb(old_status, new_status, msg)
            except Exception:
                pass  # Never let callback failure break the agent

    def _compute_status(self) -> HealthStatus:
        count = self.get_error_count_5min()
        if count >= self.ERROR_THRESHOLD:
            return HealthStatus.ERROR
        if count >= self.DEGRADED_THRESHOLD:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def get_error_count_5min(self) -> int:
        now = time.monotonic()
        cutoff = now - self.ERROR_WINDOW_SECS
        with self._lock:
            while self._errors_5min and self._errors_5min[0] < cutoff:
                self._errors_5min.popleft()
            return len(self._errors_5min)

    def get_status(self) -> HealthStatus:
        new_status = self._compute_status()
        with self._lock:
            old_status = self._current_status
            if new_status != old_status:
                self._previous_status = old_status
                self._current_status = new_status
        return new_status

    def get_recent_events(self, n: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            events = list(self._events)
        return events[-n:]

    def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: v.to_dict() for k, v in self._providers.items()}

    def get_tool_stats(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: v.to_dict() for k, v in self._tools.items()}

    def to_dict(self) -> Dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            recent_cu = list(self.recent_computer_use_failures)
            cu_consecutive = self.consecutive_computer_use_failures
            mem_consecutive = self.consecutive_memory_sync_failures
        return {
            "status": self.get_status().value,
            "last_error": self.last_error,
            "last_error_age_s": round(now - self.last_error_time, 1) if self.last_error_time else None,
            "errors_5min": self.get_error_count_5min(),
            "providers": self.get_provider_stats(),
            "tools": self.get_tool_stats(),
            "last_successful_llm_call_age_s": (
                round(now - self.last_llm_success_time, 1) if self.last_llm_success_time else None
            ),
            "last_memory_sync_age_s": (
                round(now - self.last_memory_sync_time, 1) if self.last_memory_sync_time else None
            ),
            "last_memory_sync_error": self.last_memory_sync_error,
            "consecutive_memory_sync_failures": mem_consecutive,
            "recent_computer_use_failures": recent_cu[-10:],
            "consecutive_computer_use_failures": cu_consecutive,
            "sessions_active": self.sessions_active,
            "sessions_total": self.sessions_total,
            "subagents_total": self.subagents_total,
            "uptime_s": round(now - self.started_at, 1),
            "recent_events": self.get_recent_events(20),
        }
