"""Entry point for the Propulsion Bot Phase 2 system."""

from __future__ import annotations

from propulsion_bot import IntradayOrchestrator


def main() -> None:
    print("=" * 60)
    print("PROPULSION BOT - PHASE 2 (INTRADAY)")
    print("=" * 60)

    orchestrator = IntradayOrchestrator()

    print("Starting continuous operation (auto)...")
    try:
        orchestrator.start()
    except KeyboardInterrupt:
        orchestrator.stop()
    finally:
        try:
            if getattr(orchestrator, "ib", None) is not None:
                orchestrator.ib.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()

