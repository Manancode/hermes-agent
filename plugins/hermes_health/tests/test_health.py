"""Unit tests for hermes_health plugin — hardened version.

Tests state-based alerting, severity classification, plugin discovery,
and integration with the real Hermes plugin loader.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch

import pytest

# Ensure the plugin is importable
_plugin_dir = Path(__file__).resolve().parent.parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from hermes_health.state import HealthState, HealthStatus
from hermes_health.notifiers import AlertNotifier
from hermes_health import observer


# -- Fixtures -----------------------------------------------------------------

@pytest.fixture
def state() -> HealthState:
    return HealthState()


@pytest.fixture
def notifier() -> AlertNotifier:
    return AlertNotifier(cooldown_secs=0.05)


@pytest.fixture
def initialized(state: HealthState, notifier: AlertNotifier):
    state.on_transition(notifier.on_transition)
    observer.init(state, notifier)
    yield state, notifier
    observer.init(HealthState(), AlertNotifier())


# -- HealthState: status computation ------------------------------------------

class TestHealthState:
    def test_initial_status_is_healthy(self, state: HealthState):
        assert state.get_status() == HealthStatus.HEALTHY

    def test_record_success(self, state: HealthState):
        state.record_event("llm_response", success=True, provider="openai", model="gpt-4", duration_ms=150.0)
        assert state.get_status() == HealthStatus.HEALTHY
        stats = state.get_provider_stats()
        assert "openai" in stats
        assert stats["openai"]["total_calls"] == 1
        assert stats["openai"]["errors"] == 0

    def test_record_error(self, state: HealthState):
        state.record_event(
            "llm_error", success=False, provider="hcnsec", model="auto",
            error_type="timeout", error_message="Request timed out",
        )
        assert state.last_error == "Request timed out"
        assert state.get_error_count_5min() == 1
        assert state.get_status() == HealthStatus.HEALTHY

    def test_degraded_threshold(self, state: HealthState):
        for i in range(5):
            state.record_event("llm_error", success=False, error_type="error")
        assert state.get_status() == HealthStatus.DEGRADED

    def test_error_threshold(self, state: HealthState):
        for i in range(15):
            state.record_event("llm_error", success=False, error_type="error")
        assert state.get_status() == HealthStatus.ERROR

    def test_error_window_expiry(self, state: HealthState):
        state.ERROR_WINDOW_SECS = 0.05
        for i in range(5):
            state.record_event("llm_error", success=False, error_type="error")
        assert state.get_error_count_5min() == 5
        time.sleep(0.1)
        assert state.get_error_count_5min() == 0
        # get_status() re-evaluates from the window
        assert state.get_status() == HealthStatus.HEALTHY

    def test_tool_tracking(self, state: HealthState):
        state.record_event("tool_call", success=True, tool_name="terminal", duration_ms=50.0)
        state.record_event("tool_call", success=True, tool_name="terminal", duration_ms=100.0)
        stats = state.get_tool_stats()
        assert stats["terminal"]["total_calls"] == 2
        assert stats["terminal"]["avg_duration_ms"] == 75.0

    def test_consecutive_errors_tracked(self, state: HealthState):
        state.record_event("tool_call", success=False, tool_name="terminal", error_type="Error")
        state.record_event("tool_call", success=False, tool_name="terminal", error_type="Error")
        state.record_event("tool_call", success=True, tool_name="terminal", duration_ms=10.0)
        stats = state.get_tool_stats()
        assert stats["terminal"]["consecutive_errors"] == 0

    def test_to_dict(self, state: HealthState):
        state.record_event("llm_response", success=True, provider="openai", duration_ms=100.0)
        data = state.to_dict()
        assert data["status"] == "healthy"
        assert "providers" in data
        assert "tools" in data
        assert "recent_events" in data
        assert "uptime_s" in data
        assert "consecutive_memory_sync_failures" in data
        assert "consecutive_computer_use_failures" in data


# -- State transitions --------------------------------------------------------

class TestStateTransitions:
    def test_healthy_to_degraded(self, state: HealthState):
        transitions = []
        state.on_transition(lambda old, new, msg: transitions.append((old, new, msg)))

        for i in range(5):
            state.record_event("llm_error", success=False, error_type="error")

        assert len(transitions) == 1
        assert transitions[0][0] == HealthStatus.HEALTHY
        assert transitions[0][1] == HealthStatus.DEGRADED
        assert "degraded" in transitions[0][2].lower()

    def test_degraded_to_error(self, state: HealthState):
        transitions = []
        state.on_transition(lambda old, new, msg: transitions.append((old, new, msg)))

        for i in range(5):
            state.record_event("llm_error", success=False, error_type="error")
        for i in range(10):
            state.record_event("llm_error", success=False, error_type="error")

        assert len(transitions) == 2
        assert transitions[0][:2] == (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
        assert transitions[1][:2] == (HealthStatus.DEGRADED, HealthStatus.ERROR)

    def test_error_to_healthy_recovery(self, state: HealthState):
        transitions = []
        state.on_transition(lambda old, new, msg: transitions.append((old, new, msg)))

        # Go to error — triggers HEALTHY→DEGRADED→ERROR = 2 transitions
        for i in range(15):
            state.record_event("llm_error", success=False, error_type="error")
        assert state.get_status() == HealthStatus.ERROR
        initial_count = len(transitions)

        # Wait for window to expire, then succeed
        state.ERROR_WINDOW_SECS = 0.05
        time.sleep(0.1)
        state.record_event("llm_response", success=True, duration_ms=10.0)

        assert state.get_status() == HealthStatus.HEALTHY
        # Should have one more transition (ERROR→HEALTHY)
        assert len(transitions) == initial_count + 1
        assert transitions[-1][:2] == (HealthStatus.ERROR, HealthStatus.HEALTHY)
        assert "recovered" in transitions[-1][2].lower()

    def test_no_transition_without_threshold(self, state: HealthState):
        transitions = []
        state.on_transition(lambda old, new, msg: transitions.append((old, new, msg)))

        for i in range(4):
            state.record_event("llm_error", success=False, error_type="error")

        assert len(transitions) == 0
        assert state.get_status() == HealthStatus.HEALTHY

    def test_memory_sync_consecutive_failures(self, state: HealthState):
        for i in range(3):
            state.record_memory_sync(success=False, error="connection refused")
        assert state.consecutive_memory_sync_failures == 3
        assert state.get_error_count_5min() == 3

    def test_memory_sync_success_resets_consecutive(self, state: HealthState):
        state.record_memory_sync(success=False, error="fail")
        state.record_memory_sync(success=False, error="fail")
        state.record_memory_sync(success=True)
        assert state.consecutive_memory_sync_failures == 0


# -- AlertNotifier: state-based alerting --------------------------------------

class TestAlertNotifier:
    def test_transition_healthy_to_degraded_fires(self, notifier: AlertNotifier):
        fired = []
        notifier.add_callback(lambda t, m: fired.append((t, m)))
        notifier.on_transition(HealthStatus.HEALTHY, HealthStatus.DEGRADED, "degraded msg")
        assert len(fired) == 1
        assert fired[0][0] == "degraded"

    def test_transition_error_to_healthy_fires(self, notifier: AlertNotifier):
        fired = []
        notifier.add_callback(lambda t, m: fired.append((t, m)))
        notifier.on_transition(HealthStatus.ERROR, HealthStatus.HEALTHY, "recovered msg")
        assert len(fired) == 1
        assert fired[0][0] == "recovery"

    def test_same_status_no_alert(self, notifier: AlertNotifier):
        fired = []
        notifier.add_callback(lambda t, m: fired.append((t, m)))
        notifier.on_transition(HealthStatus.HEALTHY, HealthStatus.HEALTHY, "no change")
        assert len(fired) == 0

    def test_cooldown_suppresses(self, notifier: AlertNotifier):
        fired = []
        notifier.add_callback(lambda t, m: fired.append((t, m)))
        notifier.on_transition(HealthStatus.HEALTHY, HealthStatus.DEGRADED, "first")
        notifier.on_transition(HealthStatus.HEALTHY, HealthStatus.DEGRADED, "second")
        assert len(fired) == 1

    def test_cooldown_expiry_allows(self, notifier: AlertNotifier):
        fired = []
        notifier.add_callback(lambda t, m: fired.append((t, m)))
        notifier.on_transition(HealthStatus.HEALTHY, HealthStatus.DEGRADED, "first")
        time.sleep(0.1)
        notifier.on_transition(HealthStatus.HEALTHY, HealthStatus.DEGRADED, "second")
        assert len(fired) == 2

    def test_callback_exception_does_not_crash(self, notifier: AlertNotifier):
        def bad_cb(t: str, m: str) -> None:
            raise RuntimeError("boom")
        notifier.add_callback(bad_cb)
        notifier.on_transition(HealthStatus.HEALTHY, HealthStatus.DEGRADED, "test")


# -- Observer hook tests: no per-failure alerts -------------------------------

class TestObserverNoSpam:
    def test_single_llm_failure_no_alert(self, initialized):
        state, notifier = initialized
        fired = []
        notifier.add_callback(lambda t, m: fired.append(t))

        observer.on_api_request_error(
            provider="hcnsec", model="auto", session_id="s1",
            status_code=500, reason="error",
            error={"type": "Error", "message": "fail"},
        )
        assert state.get_status() == HealthStatus.HEALTHY
        assert len(fired) == 0

    def test_single_tool_failure_no_alert(self, initialized):
        state, notifier = initialized
        fired = []
        notifier.add_callback(lambda t, m: fired.append(t))

        observer.on_post_tool_call(
            tool_name="terminal", args={"command": "ls"}, result=None,
            duration_ms=50, status="error",
            error_type="FileNotFoundError", error_message="not found",
        )
        assert state.get_status() == HealthStatus.HEALTHY
        assert len(fired) == 0

    def test_single_computer_use_failure_no_alert(self, initialized):
        state, notifier = initialized
        fired = []
        notifier.add_callback(lambda t, m: fired.append(t))

        observer.on_post_tool_call(
            tool_name="computer_use",
            args={"action": "click"},
            result=None, duration_ms=200, status="error",
            error_type="TimeoutError", error_message="Click timed out",
        )
        assert state.get_status() == HealthStatus.HEALTHY
        assert len(fired) == 0
        with state._lock:
            assert state.consecutive_computer_use_failures == 1

    def test_repeated_llm_failures_trigger_degraded_alert(self, initialized):
        state, notifier = initialized
        fired = []
        notifier.add_callback(lambda t, m: fired.append(t))

        for i in range(5):
            observer.on_api_request_error(
                provider="hcnsec", model="auto", session_id="s1",
                status_code=500, reason="error",
                error={"type": "Error", "message": "fail"},
            )
        assert state.get_status() == HealthStatus.DEGRADED
        assert len(fired) == 1
        assert fired[0] == "degraded"

    def test_recovery_notification(self, initialized):
        state, notifier = initialized
        fired = []
        notifier.add_callback(lambda t, m: fired.append(t))

        # Go to degraded
        for i in range(5):
            observer.on_api_request_error(
                provider="hcnsec", model="auto", session_id="s1",
                status_code=500, reason="error",
                error={"type": "Error", "message": "fail"},
            )
        assert len(fired) == 1

        # Wait for window to expire, then succeed
        state.ERROR_WINDOW_SECS = 0.05
        time.sleep(0.1)
        observer.on_post_api_request(
            provider="hcnsec", model="auto", session_id="s1", api_duration=1.0,
        )
        assert state.get_status() == HealthStatus.HEALTHY
        assert len(fired) == 2
        assert fired[1] == "recovery"


# -- Computer-use tracking ----------------------------------------------------

class TestComputerUse:
    def test_consecutive_failures_tracked(self, initialized):
        state, _ = initialized
        for i in range(3):
            observer.on_post_tool_call(
                tool_name="computer_use", args={"action": "click"},
                result=None, duration_ms=100, status="error",
                error_type="Error", error_message="fail",
            )
        with state._lock:
            assert state.consecutive_computer_use_failures == 3

    def test_success_resets_consecutive(self, initialized):
        state, _ = initialized
        observer.on_post_tool_call(
            tool_name="computer_use", args={"action": "click"},
            result=None, duration_ms=100, status="error",
            error_type="Error", error_message="fail",
        )
        observer.on_post_tool_call(
            tool_name="computer_use", args={"action": "click"},
            result={"screenshot": "..."}, duration_ms=100, status="ok",
        )
        with state._lock:
            assert state.consecutive_computer_use_failures == 0

    def test_failure_recorded_in_state(self, initialized):
        state, _ = initialized
        observer.on_post_tool_call(
            tool_name="computer_use", args={"action": "type", "text": "hello"},
            result=None, duration_ms=50, status="error",
            error_type="ValueError", error_message="invalid",
        )
        with state._lock:
            assert len(state.recent_computer_use_failures) == 1
            assert state.recent_computer_use_failures[0]["action"] == "type"


# -- Memory sync hook readiness -----------------------------------------------

class TestMemorySync:
    def test_on_memory_sync_records(self, initialized):
        state, _ = initialized
        observer.on_memory_sync(
            provider_name="tencentdb", success=True, duration_ms=50.0, session_id="s1",
        )
        assert state.last_memory_sync_time is not None
        assert state.consecutive_memory_sync_failures == 0

    def test_on_memory_sync_failure(self, initialized):
        state, _ = initialized
        observer.on_memory_sync(
            provider_name="tencentdb", success=False, error="connection refused",
        )
        assert state.consecutive_memory_sync_failures == 1
        assert state.last_memory_sync_error == "connection refused"

    def test_on_memory_sync_consecutive_failures(self, initialized):
        state, _ = initialized
        for i in range(3):
            observer.on_memory_sync(success=False, error="fail")
        assert state.consecutive_memory_sync_failures == 3


# -- Hook failure isolation ---------------------------------------------------

class TestHookFailureIsolation:
    def test_hook_exception_does_not_break(self, initialized):
        """If a hook throws, the agent must continue."""
        state, _ = initialized
        # Corrupt the state to cause an exception
        state._events = None  # Will cause AttributeError on append
        # Should not raise — record_event catches exceptions
        observer.on_post_api_request(
            provider="test", model="test", session_id="s1", api_duration=1.0,
        )


# -- Zero synthetic LLM calls ------------------------------------------------

class TestNoSyntheticLLM:
    def test_no_llm_calls_in_plugin(self):
        """The plugin must never make an LLM call."""
        import hermes_health
        import ast
        init_file = Path(hermes_health.__file__).resolve()
        source = init_file.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("complete", "chat", "generate", "invoke"):
                        pytest.fail(f"Plugin makes LLM call: {node.func.attr}")
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("complete", "chat", "generate"):
                        pytest.fail(f"Plugin makes LLM call: {node.func.id}")


# -- Plugin registration (real context) ---------------------------------------

class TestPluginRegistration:
    def test_register_loads(self):
        import hermes_health
        assert hasattr(hermes_health, "register")
        assert callable(hermes_health.register)

    def test_register_sets_up_hooks(self):
        mock_ctx = MagicMock()
        import hermes_health
        hermes_health.register(mock_ctx)

        hook_calls = [call[0][0] for call in mock_ctx.register_hook.call_args_list]
        expected_hooks = [
            "pre_api_request", "post_api_request", "api_request_error",
            "pre_llm_call", "post_llm_call", "pre_tool_call", "post_tool_call",
            "on_session_start", "on_session_end", "on_session_finalize",
            "subagent_start", "subagent_stop",
        ]
        for h in expected_hooks:
            assert h in hook_calls

        mock_ctx.register_cli_command.assert_called_once()
        cli_args = mock_ctx.register_cli_command.call_args
        assert cli_args[1]["name"] == "health"

    def test_register_wires_transition_callback(self):
        mock_ctx = MagicMock()
        import hermes_health
        hermes_health.register(mock_ctx)
        state = hermes_health._get_state()
        assert len(state._transition_callbacks) > 0


# -- CLI formatting -----------------------------------------------------------

class TestCLIFormatting:
    def test_format_healthy(self):
        state = HealthState()
        data = state.to_dict()
        from hermes_health import _format_health_text
        text = _format_health_text(data)
        assert "HEALTHY" in text
        assert "Uptime:" in text

    def test_format_with_consecutive_failures(self):
        state = HealthState()
        state.consecutive_memory_sync_failures = 3
        state.last_memory_sync_time = time.monotonic()
        data = state.to_dict()
        from hermes_health import _format_health_text
        text = _format_health_text(data)
        assert "consecutive failures" in text

    def test_format_json(self):
        state = HealthState()
        data = state.to_dict()
        json_str = json.dumps(data, indent=2)
        parsed = json.loads(json_str)
        assert parsed["status"] == "healthy"


# -- REST endpoint security ---------------------------------------------------

class TestRESTSecurity:
    def test_to_dict_no_secrets(self, state: HealthState):
        data = state.to_dict()
        json_str = json.dumps(data)
        for secret_pattern in ("api_key", "token", "secret", "password", "bearer", "sk-"):
            assert secret_pattern not in json_str.lower()


# -- Integration: real plugin loader (requires hermes_cli) --------------------

class TestRealPluginLoader:
    """Test plugin discovery using the real Hermes PluginManager.

    These tests require hermes_cli to be importable (i.e., running inside
    the hermes-agent virtualenv). They are skipped in standalone test runs.
    """

    def test_plugin_discovered_by_loader(self, tmp_path, monkeypatch):
        try:
            from hermes_cli.plugins import PluginManager
        except ImportError:
            pytest.skip("hermes_cli not importable — run inside hermes-agent venv")

        import shutil
        home = tmp_path / "hermes_home"
        home.mkdir()
        plugins_dir = home / "plugins"
        plugins_dir.mkdir()
        plugin_dest = plugins_dir / "hermes_health"
        shutil.copytree(str(_plugin_dir), str(plugin_dest))
        monkeypatch.setenv("HERMES_HOME", str(home))

        mgr = PluginManager()
        mgr.discover_and_load()

        assert "hermes_health" in mgr._plugins or any(
            "hermes_health" in str(v) for v in mgr._plugins.keys()
        )

    def test_register_ctx_calls(self, tmp_path, monkeypatch):
        try:
            from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
        except ImportError:
            pytest.skip("hermes_cli not importable — run inside hermes-agent venv")

        mgr = PluginManager()
        manifest = PluginManifest(name="hermes_health")
        ctx = PluginContext(manifest, mgr)

        import hermes_health
        hermes_health.register(ctx)

        hook_names = [call[0][0] for call in ctx.register_hook.call_args_list]
        assert len(hook_names) == 12
        ctx.register_cli_command.assert_called_once()
