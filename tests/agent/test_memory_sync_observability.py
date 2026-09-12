"""Tests for memory sync observability: on_memory_sync hook emission.

Covers:
    1. Successful memory sync fires on_memory_sync with success=True
    2. Failed memory sync fires on_memory_sync with success=False + error
    3. Hook payload contains provider_name, duration_ms, session_id, error
    4. Provider exception isolation (one provider failing doesn't affect others)
    5. No regression to existing memory sync behavior (background, immediate return)
    6. Hook dispatch failure is caught and logged (doesn't crash sync)
"""
import logging
import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from agent.memory_provider import MemoryProvider
from agent.memory_manager import MemoryManager


# ---------------------------------------------------------------------------
# Test providers
# ---------------------------------------------------------------------------

class _SuccessfulProvider(MemoryProvider):
    """Provider that syncs successfully."""

    _name = "test_ok"

    def __init__(self):
        self.sync_calls: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    def initialize(self, session_id: str = "", **kwargs) -> None:
        pass

    def is_available(self) -> bool:
        return True

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query, *, session_id: str = "") -> str:
        return ""

    def queue_prefetch(self, query, *, session_id: str = "") -> None:
        pass

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None) -> None:
        self.sync_calls.append({"user": user_content, "assistant": assistant_content, "session_id": session_id})

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs) -> str:
        return ""


class _FailingProvider(MemoryProvider):
    """Provider whose sync_turn always raises."""

    _name = "test_fail"

    def __init__(self, error_msg: str = "connection refused"):
        self._error_msg = error_msg

    @property
    def name(self) -> str:
        return self._name

    def initialize(self, session_id: str = "", **kwargs) -> None:
        pass

    def is_available(self) -> bool:
        return True

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query, *, session_id: str = "") -> str:
        return ""

    def queue_prefetch(self, query, *, session_id: str = "") -> None:
        pass

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None) -> None:
        raise RuntimeError(self._error_msg)

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs) -> str:
        return ""


class _SlowProvider(MemoryProvider):
    """Provider whose sync blocks for a configurable delay."""

    _name = "test_slow"

    def __init__(self, delay: float = 0.1):
        self._delay = delay
        self.sync_done = False

    @property
    def name(self) -> str:
        return self._name

    def initialize(self, session_id: str = "", **kwargs) -> None:
        pass

    def is_available(self) -> bool:
        return True

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query, *, session_id: str = "") -> str:
        return ""

    def queue_prefetch(self, query, *, session_id: str = "") -> None:
        pass

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None) -> None:
        time.sleep(self._delay)
        self.sync_done = True

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs) -> str:
        return ""


# ---------------------------------------------------------------------------
# Tests: on_memory_sync hook
# ---------------------------------------------------------------------------

class TestOnMemorySyncHook:
    """Verify the on_memory_sync hook fires with correct data."""

    def test_successful_sync_fires_hook(self, monkeypatch):
        """Successful sync_turn fires on_memory_sync with success=True."""
        hook_events = []

        def mock_invoke_hook(name, **kwargs):
            if name == "on_memory_sync":
                hook_events.append(kwargs)
            return []

        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", mock_invoke_hook)

        mgr = MemoryManager()
        p = _SuccessfulProvider()
        mgr.add_provider(p)

        mgr.sync_all("hello", "hi there", session_id="test-session")
        assert mgr.flush_pending(timeout=5) is True

        assert len(hook_events) == 1
        evt = hook_events[0]
        assert evt["provider_name"] == "test_ok"
        assert evt["success"] is True
        assert evt["session_id"] == "test-session"
        assert isinstance(evt["duration_ms"], int)
        assert evt["duration_ms"] >= 0
        assert evt["error"] is None

    def test_failed_sync_fires_hook(self, monkeypatch):
        """Failed sync_turn fires on_memory_sync with success=False and error."""
        hook_events = []

        def mock_invoke_hook(name, **kwargs):
            if name == "on_memory_sync":
                hook_events.append(kwargs)
            return []

        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", mock_invoke_hook)

        mgr = MemoryManager()
        p = _FailingProvider(error_msg="gateway timeout")
        mgr.add_provider(p)

        mgr.sync_all("hello", "hi there", session_id="fail-session")
        assert mgr.flush_pending(timeout=5) is True

        assert len(hook_events) == 1
        evt = hook_events[0]
        assert evt["provider_name"] == "test_fail"
        assert evt["success"] is False
        assert evt["session_id"] == "fail-session"
        assert "gateway timeout" in evt["error"]
        assert isinstance(evt["duration_ms"], int)

    def test_hook_exception_does_not_break_sync(self, monkeypatch):
        """If the hook dispatch itself raises, sync still completes."""
        call_count = [0]

        def mock_invoke_hook(name, **kwargs):
            if name == "on_memory_sync":
                call_count[0] += 1
                raise RuntimeError("hook crashed")
            return []

        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", mock_invoke_hook)

        mgr = MemoryManager()
        p = _SuccessfulProvider()
        mgr.add_provider(p)

        mgr.sync_all("hello", "hi", session_id="s1")
        assert mgr.flush_pending(timeout=5) is True
        assert call_count[0] == 1
        assert p.sync_calls[0]["user"] == "hello"

    def test_hook_fires_inside_background_thread(self, monkeypatch):
        """Hook fires in the background worker, not in the calling thread."""
        hook_thread_ids = []

        def mock_invoke_hook(name, **kwargs):
            if name == "on_memory_sync":
                hook_thread_ids.append(threading.current_thread().ident)
            return []

        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", mock_invoke_hook)

        mgr = MemoryManager()
        p = _SuccessfulProvider()
        mgr.add_provider(p)

        calling_thread = threading.current_thread().ident
        mgr.sync_all("hello", "hi", session_id="s1")
        mgr.flush_pending(timeout=5)

        assert len(hook_thread_ids) == 1
        assert hook_thread_ids[0] != calling_thread


# ---------------------------------------------------------------------------
# Tests: provider exception isolation
# ---------------------------------------------------------------------------

class TestProviderExceptionIsolation:
    """One provider failing must not affect the other."""

    def test_one_failing_one_ok(self, monkeypatch):
        """sync_all succeeds for the OK provider even when the failing one raises."""
        hook_events = []

        def mock_invoke_hook(name, **kwargs):
            if name == "on_memory_sync":
                hook_events.append(kwargs)
            return []

        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", mock_invoke_hook)

        mgr = MemoryManager()
        # Builtin provider is always accepted; external provider is the failing one.
        ok = _SuccessfulProvider()
        ok._name = "builtin"
        fail = _FailingProvider(error_msg="boom")
        mgr.add_provider(ok)
        mgr.add_provider(fail)

        mgr.sync_all("hello", "hi", session_id="s1")
        assert mgr.flush_pending(timeout=5) is True

        # Both providers got sync_turn called
        assert len(ok.sync_calls) == 1

        # Hook fired for both: one success, one failure
        assert len(hook_events) == 2
        successes = [e for e in hook_events if e["success"]]
        failures = [e for e in hook_events if not e["success"]]
        assert len(successes) == 1
        assert len(failures) == 1
        assert successes[0]["provider_name"] == "builtin"
        assert failures[0]["provider_name"] == "test_fail"

    def test_no_regression_background_sync_behavior(self):
        """sync_all still returns immediately; work completes in background."""
        mgr = MemoryManager()
        p = _SlowProvider(delay=0.05)
        mgr.add_provider(p)

        t0 = time.monotonic()
        mgr.sync_all("hello", "hi", session_id="s1")
        elapsed = time.monotonic() - t0

        # Should return in well under the delay
        assert elapsed < 0.05
        # Work completes in background
        assert mgr.flush_pending(timeout=5) is True
        assert p.sync_done is True
