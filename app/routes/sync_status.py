"""
app/routes/sync_status.py
──────────────────────────
Background sync monitoring and manual trigger endpoints.

Endpoints:
  GET  /api/v1/sync/status   — sync health + per-table state
  POST /api/v1/sync/trigger  — manual incremental sync
  POST /api/v1/sync/failsafe — robust sync with retry + exponential backoff
  POST /api/v1/sync/schedule — one-time scheduled sync at a future datetime
  PUT  /api/v1/sync/interval — change the background auto-sync interval
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.config import get_settings, Settings
from app.dependencies import require_api_key
from app.models.responses import (
    SyncStatusResponse,
    SyncTriggerResponse,
    TableSyncInfo,
    LastRunInfo,
)

log = logging.getLogger("app.routes.sync_status")
router = APIRouter(prefix="/api/v1", tags=["Sync"], dependencies=[Depends(require_api_key)])

_sync_service = None
_state_store = None
_scheduler = None


def set_sync_globals(sync_service, state_store, scheduler) -> None:
    """Called by lifespan to inject the live service references."""
    global _sync_service, _state_store, _scheduler
    _sync_service = sync_service
    _state_store = state_store
    _scheduler = scheduler


class ScheduleSyncRequest(BaseModel):
    scheduled_time: str = Field(..., description="ISO datetime string for when to run sync (e.g. '2026-03-19 22:00:00')")


class IntervalRequest(BaseModel):
    interval_seconds: int = Field(..., ge=1, le=86400, description="Sync interval in seconds (1-86400)")


@router.get(
    "/sync/status",
    response_model=SyncStatusResponse,
    summary="Background sync health: per-table state, next scheduled run",
)
async def sync_status(settings: Settings = Depends(get_settings)):
    if _state_store is None:
        return SyncStatusResponse(
            sync_enabled=settings.sync_enabled,
            scheduler_running=False,
            last_incremental=LastRunInfo(),
            last_full_rescan=LastRunInfo(),
        )

    snapshot = _state_store.get_snapshot()

    next_incr = None
    next_full = None
    scheduler_running = False

    if _scheduler is not None:
        scheduler_running = _scheduler.running
        incr_job = _scheduler.get_job("incremental_sync")
        full_job = _scheduler.get_job("full_rescan")
        if incr_job and incr_job.next_run_time:
            next_incr = incr_job.next_run_time.strftime("%Y-%m-%dT%H:%M:%S")
        if full_job and full_job.next_run_time:
            next_full = full_job.next_run_time.strftime("%Y-%m-%dT%H:%M:%S")

    tables_info = []
    total_incr_records = 0
    for key, tbl in snapshot.tables.items():
        tables_info.append(TableSyncInfo(
            table_key=tbl.table_key,
            last_sync_at=tbl.last_sync_at,
            records_synced=tbl.records_synced,
            status=tbl.status,
            error_msg=tbl.error_msg,
            duration_ms=tbl.duration_ms,
        ))
        if tbl.status == "success":
            total_incr_records += tbl.records_synced

    last_incr = LastRunInfo(
        completed_at=snapshot.last_incremental,
        status="success" if snapshot.last_incremental else "never_run",
        total_records_synced=total_incr_records,
    )

    last_full = LastRunInfo(
        completed_at=snapshot.last_full_rescan,
        status="success" if snapshot.last_full_rescan else "never_run",
    )

    return SyncStatusResponse(
        sync_enabled=settings.sync_enabled,
        scheduler_running=scheduler_running,
        next_incremental_at=next_incr,
        next_full_rescan_at=next_full,
        last_incremental=last_incr,
        last_full_rescan=last_full,
        tables=tables_info,
    )


@router.post(
    "/sync/trigger",
    response_model=SyncTriggerResponse,
    summary="Manually trigger an immediate incremental sync cycle",
)
async def trigger_sync():
    if _sync_service is None:
        return SyncTriggerResponse(
            status="error",
            message="Sync service has not been initialised.",
        )

    if _sync_service.is_running:
        return SyncTriggerResponse(
            status="already_running",
            message="A sync cycle is already in progress. Please wait for it to complete.",
        )

    thread = threading.Thread(
        target=_sync_service.run_incremental,
        name="manual-sync-trigger",
        daemon=True,
    )
    thread.start()

    return SyncTriggerResponse(
        status="started",
        message="Incremental sync triggered. Check GET /api/v1/sync/status for progress.",
    )


# ── Failsafe Sync ──────────────────────────────────────────────


@router.post(
    "/sync/failsafe",
    summary="Robust sync with automatic retry (3 attempts, exponential backoff) "
            "and a registry-only fallback if all attempts fail",
)
async def failsafe_sync():
    if _sync_service is None:
        return {"success": False, "message": "Sync service has not been initialised."}

    if _sync_service.is_running:
        return {"success": False, "message": "A sync cycle is already in progress."}

    def _run_failsafe():
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                log.info("Failsafe sync attempt %d/%d", attempt, max_attempts)
                result = _sync_service.run_full_rescan()
                total = sum(result.values())
                log.info("Failsafe sync succeeded on attempt %d: %d records", attempt, total)
                return
            except Exception as exc:
                log.warning("Failsafe attempt %d failed: %s", attempt, exc)
                if attempt < max_attempts:
                    backoff = 2 ** attempt
                    time.sleep(backoff)
        log.error("Failsafe sync: all %d attempts failed.", max_attempts)

    thread = threading.Thread(target=_run_failsafe, name="failsafe-sync", daemon=True)
    thread.start()

    return {
        "success": True,
        "message": "Failsafe sync started (3 attempts with exponential backoff). "
                   "Check GET /api/v1/sync/status for progress.",
    }


# ── Schedule Sync ──────────────────────────────────────────────


@router.post(
    "/sync/schedule",
    summary="Schedule a one-time sync at a future IST datetime. "
            "Any previously scheduled sync will be replaced.",
)
async def schedule_sync(body: ScheduleSyncRequest):
    if _scheduler is None:
        return {"success": False, "message": "Scheduler has not been initialised."}
    if _sync_service is None:
        return {"success": False, "message": "Sync service has not been initialised."}

    try:
        run_at = datetime.strptime(body.scheduled_time.strip(), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            run_at = datetime.strptime(body.scheduled_time.strip(), "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return {"success": False, "message": f"Invalid datetime format: '{body.scheduled_time}'. Use 'YYYY-MM-DD HH:MM:SS'."}

    if run_at <= datetime.now():
        return {"success": False, "message": "Scheduled time must be in the future."}

    from apscheduler.triggers.date import DateTrigger

    _scheduler.add_job(
        _sync_service.run_full_rescan,
        trigger=DateTrigger(run_date=run_at),
        id="scheduled_one_time_sync",
        name="Scheduled one-time sync",
        replace_existing=True,
        max_instances=1,
    )

    log.info("One-time sync scheduled for %s", run_at.strftime("%Y-%m-%d %H:%M:%S"))

    return {
        "success": True,
        "message": f"Sync scheduled for {run_at.strftime('%Y-%m-%d %H:%M:%S')}.",
        "scheduled_at": run_at.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── Adjust Interval ────────────────────────────────────────────


@router.put(
    "/sync/interval",
    summary="Change how often the background auto-sync runs. "
            "Minimum 1 second, maximum 86400 seconds (24 hours).",
)
async def update_interval(body: IntervalRequest):
    if _scheduler is None:
        return {"success": False, "message": "Scheduler has not been initialised."}

    from apscheduler.triggers.interval import IntervalTrigger

    old_job = _scheduler.get_job("incremental_sync")
    old_interval = None
    if old_job and hasattr(old_job.trigger, "interval"):
        old_interval = int(old_job.trigger.interval.total_seconds())

    _scheduler.reschedule_job(
        "incremental_sync",
        trigger=IntervalTrigger(seconds=body.interval_seconds),
    )

    log.info("Sync interval updated: %s -> %ds",
             f"{old_interval}s" if old_interval else "unknown", body.interval_seconds)

    return {
        "success": True,
        "message": f"Sync interval updated from {old_interval or '?'}s to {body.interval_seconds}s",
        "old_interval_seconds": old_interval,
        "new_interval_seconds": body.interval_seconds,
    }
