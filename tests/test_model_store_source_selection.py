import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.model_store import choose_newest_source


def test_choose_newest_source_prefers_newer_disk_when_dates_present():
    db_meta = {"train_date": "2026-04-20T10:00:00"}
    disk_meta = {"train_date": "2026-04-21T10:00:00"}
    assert choose_newest_source(db_meta=db_meta, disk_meta=disk_meta) == "disk"


def test_choose_newest_source_prefers_db_when_db_is_newer():
    db_meta = {"train_date": "2026-04-21T10:00:00"}
    disk_meta = {"train_date": "2026-04-20T10:00:00"}
    assert choose_newest_source(db_meta=db_meta, disk_meta=disk_meta) == "db"


def test_choose_newest_source_defaults_to_db_when_dates_missing():
    assert choose_newest_source(db_meta={}, disk_meta={}) == "db"
    assert choose_newest_source(db_meta={"foo": "bar"}, disk_meta={"train_date": "bad"}) == "db"
