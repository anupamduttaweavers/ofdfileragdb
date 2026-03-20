"""
app.services.scheduler
───────────────────────
APScheduler setup for background sync.

Single Responsibility: owns all scheduler configuration and lifecycle.
"""

from __future__ import annotations

import logging

from app.config import Settings

log = logging.getLogger("app.services.scheduler")


def start_sync_scheduler(settings: Settings, vec_config, search_engine):
    """Configure and start APScheduler with incremental + full rescan jobs."""
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.cron import CronTrigger

    from app.services.sync_state import SyncStateStore
    from app.services.sync_service import SyncService
    from app.routes.sync_status import set_sync_globals
    from app.core.vector_store import FaissStore

    state_store = SyncStateStore(settings.sync_state_file)

    if search_engine is not None:
        faiss_store = search_engine.store
    else:
        faiss_store = FaissStore(settings.faiss_persist_dir, settings.faiss_collection_name)

    sync_service = SyncService(vec_config, state_store, faiss_store)

    scheduler = BackgroundScheduler(daemon=True)

    scheduler.add_job(
        sync_service.run_incremental,
        trigger=IntervalTrigger(seconds=settings.sync_interval_seconds),
        id="incremental_sync",
        name="Incremental vectorization sync",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.add_job(
        sync_service.run_full_rescan,
        trigger=CronTrigger(hour=settings.sync_full_rescan_hour, minute=0),
        id="full_rescan",
        name="Daily full rescan",
        replace_existing=True,
        max_instances=1,
    )

    set_sync_globals(sync_service, state_store, scheduler)
    scheduler.start()

    log.info(
        "Background sync scheduler started: incremental every %ds, full rescan daily at %02d:00.",
        settings.sync_interval_seconds,
        settings.sync_full_rescan_hour,
    )
    return scheduler
