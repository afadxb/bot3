import importlib


def test_scheduler_registers_jobs(monkeypatch):
    monkeypatch.setenv("MYSQL_URL", "sqlite:///./test_scheduler.db")
    sched = importlib.import_module("barchart_swing_bot.scheduler")
    importlib.reload(sched)
    sched.init_schedule()
    # start and stop scheduler to ensure jobs persist
    sched.start()
    try:
        for job_id in sched.SCHEDULE.keys():
            assert sched.scheduler.get_job(job_id) is not None
    finally:
        sched.shutdown()
