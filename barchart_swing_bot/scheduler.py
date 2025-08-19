from __future__ import annotations

import logging
from datetime import datetime

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from .config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


def _jobstore() -> dict[str, SQLAlchemyJobStore]:
    return {"default": SQLAlchemyJobStore(url=settings.mysql_url)}


def _scheduler() -> BackgroundScheduler:
    tz = pytz.timezone(settings.tz)
    scheduler = BackgroundScheduler(jobstores=_jobstore(), timezone=tz)
    return scheduler


SCHEDULE = {
    "ingest": (9, 0),
    "verify": (9, 5),
    "enrich": (9, 10),
    "analyze": (9, 15),
    "publish": (9, 20),
}


# Job stubs ---------------------------------------------------------------


def ingest():
    logger.info("ingest job executed")


def verify():
    logger.info("verify job executed")


def enrich():
    logger.info("enrich job executed")


def analyze():
    logger.info("analyze job executed")


def publish():
    logger.info("publish job executed")


JOBS = {
    "ingest": ingest,
    "verify": verify,
    "enrich": enrich,
    "analyze": analyze,
    "publish": publish,
}


scheduler = _scheduler()


def init_schedule() -> None:
    tz = pytz.timezone(settings.tz)
    for job_id, (hour, minute) in SCHEDULE.items():
        if scheduler.get_job(job_id):
            continue
        trigger = CronTrigger(hour=hour, minute=minute, timezone=tz)
        scheduler.add_job(
            JOBS[job_id], trigger=trigger, id=job_id, replace_existing=True
        )


def start() -> None:
    init_schedule()
    scheduler.start()


def shutdown() -> None:
    scheduler.shutdown()


def run_job_now(job_id: str) -> bool:
    job = scheduler.get_job(job_id)
    if not job:
        return False
    scheduler.add_job(job.func, id=f"run_now_{job.id}_{datetime.utcnow().timestamp()}")
    return True
