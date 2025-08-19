from __future__ import annotations

import hashlib
import logging
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from .config import get_settings
from .models import ScrapeTop100Raw, Top100Norm


logger = logging.getLogger(__name__)
settings = get_settings()
SCRAPE_URL = "https://www.barchart.com/stocks/top-100-stocks"


def _hash_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


EXPECTED_COLUMNS = {
    "symbol",
    "name",
    "wtd_alpha",
    "rank",
    "prev_rank",
    "last",
    "chg_pct",
    "high_52w",
    "low_52w",
    "chg_52w_pct",
    "time",
}


def load_csv(path: Path, db: Session, run_date: date) -> None:
    df = pd.read_csv(path)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")
    sha = _hash_file(path)
    raw = ScrapeTop100Raw(run_date=run_date, filename=str(path), sha256=sha)
    db.add(raw)
    for _, row in df.iterrows():
        db.add(
            Top100Norm(
                run_date=run_date,
                symbol=row["symbol"],
                rank=int(row["rank"]),
                wtd_alpha=float(row["wtd_alpha"]),
            )
        )
    db.commit()


def scrape_top100(db: Session, run_date: date) -> bool:
    """Attempt to scrape the Top‑100 page. Return True on success."""
    if not settings.scrape_allowed:
        return False
    try:
        tables = pd.read_html(SCRAPE_URL)
    except Exception as exc:  # pragma: no cover - network best effort
        logger.warning("scrape failed: %s", exc)
        return False
    if not tables:
        return False
    df = tables[0]
    mapping = {
        "Symbol": "symbol",
        "Name": "name",
        "Wtd Alpha": "wtd_alpha",
        "Rank": "rank",
        "Prev Rank": "prev_rank",
        "Last": "last",
        "Chg %": "chg_pct",
        "High 52W": "high_52w",
        "Low 52W": "low_52w",
        "Chg 52W %": "chg_52w_pct",
        "Time": "time",
    }
    df = df.rename(columns=mapping)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        logger.warning("scrape missing columns: %s", missing)
        return False
    path = Path(f"top100_{run_date}.csv")
    df.to_csv(path, index=False)
    load_csv(path, db, run_date)
    return True


def ingest(db: Session, run_date: date, csv_path: Path | None = None) -> None:
    """Ingest Top‑100 data via scrape or CSV fallback."""
    if scrape_top100(db, run_date):
        return
    if not csv_path:
        raise RuntimeError("scrape disabled and no CSV provided")
    load_csv(csv_path, db, run_date)
