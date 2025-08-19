from datetime import datetime
from sqlalchemy import Column, Date, DateTime, Float, ForeignKey, Integer, String

from .db import Base


class Symbol(Base):
    __tablename__ = "symbols"
    id = Column(Integer, primary_key=True)
    symbol = Column(String, unique=True, nullable=False)
    name = Column(String)


class ScrapeTop100Raw(Base):
    __tablename__ = "scrapes_top100_raw"
    id = Column(Integer, primary_key=True)
    run_date = Column(Date, nullable=False)
    filename = Column(String, nullable=False)
    sha256 = Column(String, nullable=False)


class Top100Norm(Base):
    __tablename__ = "top100_norm"
    id = Column(Integer, primary_key=True)
    run_date = Column(Date, nullable=False)
    symbol = Column(String, nullable=False)
    rank = Column(Integer)
    wtd_alpha = Column(Float)
    time = Column(DateTime, nullable=False, default=datetime.utcnow)


class Setting(Base):
    __tablename__ = "settings"
    key = Column(String, primary_key=True)
    value = Column(String, nullable=False)


class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    entry = Column(String)
    status = Column(String, default="ARMED")


class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    signal_id = Column(ForeignKey("signals.id"))
    symbol = Column(String, nullable=False)
    qty = Column(Integer)
    price = Column(Float)
    status = Column(String, default="NEW")
    created_at = Column(DateTime, default=datetime.utcnow)


class Execution(Base):
    __tablename__ = "executions"
    id = Column(Integer, primary_key=True)
    order_id = Column(ForeignKey("orders.id"))
    price = Column(Float)
    qty = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Position(Base):
    __tablename__ = "positions"
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    qty = Column(Integer, nullable=False)
    avg_price = Column(Float, nullable=False)
    opened_at = Column(DateTime, default=datetime.utcnow)


class SchedulerJob(Base):
    __tablename__ = "scheduler_jobs"
    id = Column(Integer, primary_key=True)
    job_id = Column(String, unique=True)
    next_run = Column(DateTime)
