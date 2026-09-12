"""hermes-health dashboard API — mounted at /api/plugins/hermes-health/."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


def _get_state() -> Any:
    from hermes_health import _get_state
    return _get_state()


@router.get("/health")
async def health_status() -> Any:
    state = _get_state()
    return JSONResponse(content=state.to_dict())


@router.get("/health/providers")
async def provider_stats() -> Any:
    state = _get_state()
    return JSONResponse(content=state.get_provider_stats())


@router.get("/health/tools")
async def tool_stats() -> Any:
    state = _get_state()
    return JSONResponse(content=state.get_tool_stats())


@router.get("/health/events")
async def recent_events(n: int = 50) -> Any:
    state = _get_state()
    return JSONResponse(content=state.get_recent_events(n))


@router.get("/health/computer-use")
async def computer_use_failures() -> Any:
    state = _get_state()
    with state._lock:
        failures = list(state.recent_computer_use_failures)
    return JSONResponse(content=failures)
