from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
import pandas as pd
from sqlalchemy.orm import Session

from .models import ScrapeTop100Raw, Top100Norm


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
