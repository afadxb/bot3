from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Callable, Optional

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
from sqlalchemy.orm import Session

from .config import get_settings
from .db import SessionLocal
from . import ingest as ingest_mod, confidence, notifier
from .models import Top100Norm, Signal
from .paper import PaperBroker

logger = logging.getLogger(__name__)

settings = get_settings()
broker = PaperBroker(settings.paper_start_balance)


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


# helpers -----------------------------------------------------------------


def _run_stage(
    name: str, func: Callable[[Optional[Session]], None], db: Optional[Session] = None
) -> None:
    try:
        func(db)
        notifier.send("stage completion", name)
    except Exception as exc:  # pragma: no cover - best effort
        logger.exception("%s failed", name)
        notifier.send("stage failure", f"{name}: {exc}")


def premarket_gate() -> bool:
    return broker.balance >= settings.premarket_min_available


# job implementations ------------------------------------------------------


def _ingest(db: Session | None) -> None:
    if not settings.scrape_allowed or db is None:
        logger.info("scrape not allowed; skipping ingest")
        return
    ingest_mod.ingest(db, date.today())


def ingest() -> None:
    passed = premarket_gate()
    notifier.send("pre-market gate", "passed" if passed else "failed")
    if not passed:
        return
    db = SessionLocal()
    try:
        _run_stage("ingest", _ingest, db)
    finally:
        db.close()


def verify() -> None:
    _run_stage("verify", lambda db: None)


def enrich() -> None:
    _run_stage("enrich", lambda db: None)


def _analyze(db: Session | None) -> None:
    if db is None:
        return
    rows = db.query(Top100Norm).filter(Top100Norm.run_date == date.today()).all()
    for row in rows:
        features = {"wtd_alpha": row.wtd_alpha}
        conf = confidence.compute_confidence(features)
        if conf >= 80:
            sig = Signal(symbol=row.symbol, confidence=conf, entry="MARKET")
            db.add(sig)
            notifier.send("signal armed", f"{row.symbol} {conf:.1f}%")
    db.commit()


def analyze() -> None:
    db = SessionLocal()
    try:
        _run_stage("analyze", _analyze, db)
    finally:
        db.close()


def _publish(db: Session | None) -> None:
    broker.check_risk()


def publish() -> None:
    _run_stage("publish", _publish)


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
        scheduler.add_job(JOBS[job_id], trigger=trigger, id=job_id, replace_existing=True)


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

