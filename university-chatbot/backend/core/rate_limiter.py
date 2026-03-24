"""
Simple in-memory sliding-window rate limiter.
"""

import asyncio
import time


class RateLimiter:
    """Per-user sliding-window rate limiter."""

    def __init__(
        self,
        requests_per_minute: int = 30,
        min_interval_seconds: float = 1.0,
    ):
        self.requests_per_minute = requests_per_minute
        self.min_interval = min_interval_seconds
        self._windows: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, user_id: str) -> bool:
        """
        Return True if the request is allowed, False if rate-limited.

        Maintains a sliding window of timestamps per user and enforces:
        - A maximum of *requests_per_minute* within any 60-second window.
        - A minimum gap of *min_interval* seconds between consecutive requests.
        """
        async with self._lock:
            now = time.monotonic()
            timestamps = self._windows.get(user_id, [])

            # Prune entries older than 60 seconds
            cutoff = now - 60.0
            timestamps = [ts for ts in timestamps if ts > cutoff]

            # Check per-minute cap
            if len(timestamps) >= self.requests_per_minute:
                self._windows[user_id] = timestamps
                return False

            # Check minimum interval
            if timestamps and (now - timestamps[-1]) < self.min_interval:
                self._windows[user_id] = timestamps
                return False

            timestamps.append(now)
            self._windows[user_id] = timestamps
            return True
