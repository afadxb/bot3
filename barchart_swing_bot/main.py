from fastapi import Depends, FastAPI, Header, HTTPException
from sqlalchemy.orm import Session

from .config import get_settings
from .db import Base, engine, get_db
from .models import Setting
from . import scheduler

settings = get_settings()
app = FastAPI(title="barchart-swing-bot")


@app.on_event("startup")
def startup_event() -> None:
    Base.metadata.create_all(bind=engine)
    scheduler.start()


@app.on_event("shutdown")
def shutdown_event() -> None:
    scheduler.shutdown()


def auth(token: str = Header(..., alias="X-API-TOKEN")) -> None:
    if token != settings.api_token:
        raise HTTPException(status_code=401, detail="invalid token")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/settings", dependencies=[Depends(auth)])
def get_settings_endpoint(db: Session = Depends(get_db)) -> dict[str, str]:
    rows = db.query(Setting).all()
    return {row.key: row.value for row in rows}


@app.post("/settings/{key}", dependencies=[Depends(auth)])
def set_setting(key: str, value: str, db: Session = Depends(get_db)) -> dict[str, str]:
    row = db.get(Setting, key)
    if row:
        row.value = value
    else:
        row = Setting(key=key, value=value)
        db.add(row)
    db.commit()
    return {"key": key, "value": value}


@app.post("/scheduler/{job_id}/run", dependencies=[Depends(auth)])
def run_job(job_id: str) -> dict[str, str]:
    if not scheduler.run_job_now(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    return {"status": "queued"}
