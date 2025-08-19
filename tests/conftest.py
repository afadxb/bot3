import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from barchart_swing_bot.db import Base


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)
    sess = TestingSession()
    try:
        yield sess
    finally:
        sess.close()
