from datetime import date
import pandas as pd

from barchart_swing_bot.ingest import load_csv
from barchart_swing_bot.models import Top100Norm, ScrapeTop100Raw


def test_load_csv(tmp_path, session):
    data = pd.DataFrame(
        [
            {
                "symbol": "ABC",
                "name": "ABC Inc",
                "wtd_alpha": 1.2,
                "rank": 1,
                "prev_rank": 2,
                "last": 10,
                "chg_pct": 1.0,
                "high_52w": 15,
                "low_52w": 5,
                "chg_52w_pct": 10,
                "time": "2023-01-01",
            }
        ]
    )
    csv_path = tmp_path / "top100.csv"
    data.to_csv(csv_path, index=False)
    load_csv(csv_path, session, run_date=date(2023, 1, 2))
    assert session.query(ScrapeTop100Raw).count() == 1
    row = session.query(Top100Norm).first()
    assert row.symbol == "ABC"
    assert row.rank == 1
