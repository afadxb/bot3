from __future__ import annotations

import logging
from typing import Optional

import requests

from .config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

PUSHOVER_API = "https://api.pushover.net/1/messages.json"

def send(event: str, message: str) -> None:
    """Send a Pushover notification if credentials are configured."""
    if not settings.pushover_token or not settings.pushover_user:
        logger.debug("Pushover credentials missing; skipping %s", event)
        return
    try:
        requests.post(
            PUSHOVER_API,
            data={
                "token": settings.pushover_token,
                "user": settings.pushover_user,
                "title": event,
                "message": message,
            },
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("failed to send pushover: %s", exc)
