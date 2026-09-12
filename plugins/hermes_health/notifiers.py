"""State-based alert notification.

Alerts fire ONLY on health-state transitions:
  healthy → degraded  (critical)
  degraded → error    (critical)
  degraded/error → healthy  (recovery)

Individual failures are recorded internally but do NOT trigger alerts.
The cooldown/dedup layer is a safety net, not the primary mechanism.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .state import HealthStatus

logger = logging.getLogger(__name__)


class AlertNotifier:
    """State-transition alert dispatcher with cooldown safety net."""

    DEFAULT_COOLDOWN_SECS: float = 300.0  # 5 minutes between same-type alerts

    def __init__(self, cooldown_secs: float = DEFAULT_COOLDOWN_SECS) -> None:
        self._cooldown = cooldown_secs
        self._last_alert: Dict[str, float] = {}
        self._alert_count: Dict[str, int] = {}
        self._callbacks: List[Callable[[str, str], None]] = []

    def add_callback(self, cb: Callable[[str, str], None]) -> None:
        """Register a notification callback: fn(alert_type, message)."""
        self._callbacks.append(cb)

    def on_transition(
        self,
        old_status: HealthStatus,
        new_status: HealthStatus,
        message: str,
    ) -> None:
        """Called by HealthState on status transitions. Determines alert type and fires."""
        if old_status == new_status:
            return

        # Determine alert type from transition
        if new_status == HealthStatus.HEALTHY:
            alert_type = "recovery"
        elif new_status == HealthStatus.DEGRADED:
            alert_type = "degraded"
        elif new_status == HealthStatus.ERROR:
            alert_type = "error"
        else:
            return

        self._fire(alert_type, message)

    def _fire(self, alert_type: str, message: str) -> bool:
        now = time.monotonic()
        last = self._last_alert.get(alert_type, 0.0)
        if now - last < self._cooldown:
            self._alert_count[alert_type] = self._alert_count.get(alert_type, 0) + 1
            return False

        self._last_alert[alert_type] = now
        count = self._alert_count.pop(alert_type, 0)
        if count > 0:
            message = f"{message} [suppressed {count} duplicates]"

        logger.warning("hermes_health [%s]: %s", alert_type, message)
        for cb in self._callbacks:
            try:
                cb(alert_type, message)
            except Exception:
                logger.exception("hermes_health: notification callback failed")

        return True

    def reset(self, alert_type: str) -> None:
        self._last_alert.pop(alert_type, None)
        self._alert_count.pop(alert_type, None)

    def should_alert(self, alert_type: str) -> bool:
        now = time.monotonic()
        last = self._last_alert.get(alert_type, 0.0)
        return (now - last) >= self._cooldown

    def get_suppressed_counts(self) -> Dict[str, int]:
        return dict(self._alert_count)
