"""Core package for the Propulsion bot phase 2 implementation."""

from __future__ import annotations

import logging

logging.basicConfig(level=logging.WARN)

logger = logging.getLogger("propulsion_bot")

__all__ = ["IntradayOrchestrator", "logger"]


def __getattr__(name: str):
    if name == "IntradayOrchestrator":
        from .orchestrator import IntradayOrchestrator

        return IntradayOrchestrator
    raise AttributeError(f"module 'propulsion_bot' has no attribute {name!r}")

