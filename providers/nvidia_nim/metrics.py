"""Throughput metrics collector for NVIDIA NIM proxy.

Thread-safe singleton that tracks TTFT, TPS, RPM, latency,
request interceptions, and per-account breakdowns using a rolling window.
"""

import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


ROLLING_WINDOW_SIZE = 100
RPM_WINDOW_SECONDS = 60


@dataclass(frozen=True)
class RequestMetric:
    """A single recorded request metric."""

    timestamp: float
    account_index: int
    ttft_ms: float
    tps: float
    total_latency_s: float
    output_tokens: int


class MetricsCollector:
    """Thread-safe singleton that collects throughput and latency metrics.

    Uses a rolling deque of the last N requests and a sliding RPM window.
    """

    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsCollector":
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance

    def __init__(self) -> None:
        with self.__class__._lock:
            if self._initialized:
                return
            self._metrics: deque = deque(maxlen=ROLLING_WINDOW_SIZE)
            self._rpm_timestamps: deque = deque()
            self._intercept_count: int = 0
            self._total_requests: int = 0
            self._rpm_peak: float = 0.0
            self._active_streams: Dict[int, int] = {}  # account_index -> count
            self._data_lock = threading.Lock()
            self._initialized = True

    @classmethod
    def get_instance(cls) -> "MetricsCollector":
        return cls()

    @classmethod
    def reset(cls) -> None:
        """Reset singleton for testing."""
        with cls._lock:
            cls._instance = None

    def record_request(
        self,
        account_index: int,
        ttft_ms: float,
        tps: float,
        total_latency_s: float,
        output_tokens: int,
    ) -> None:
        """Record metrics for a completed request."""
        now = time.time()
        metric = RequestMetric(
            timestamp=now,
            account_index=account_index,
            ttft_ms=ttft_ms,
            tps=tps,
            total_latency_s=total_latency_s,
            output_tokens=output_tokens,
        )

        with self._data_lock:
            self._metrics.append(metric)
            self._rpm_timestamps.append(now)
            self._total_requests += 1
            self._cleanup_rpm_window(now)

            current_rpm = len(self._rpm_timestamps)
            if current_rpm > self._rpm_peak:
                self._rpm_peak = current_rpm

    def record_interception(self) -> None:
        """Increment the intercepted request counter."""
        with self._data_lock:
            self._intercept_count += 1

    def increment_active_streams(self, account_index: int) -> None:
        """Track a new active stream for an account."""
        with self._data_lock:
            self._active_streams[account_index] = (
                self._active_streams.get(account_index, 0) + 1
            )

    def decrement_active_streams(self, account_index: int) -> None:
        """Remove an active stream for an account."""
        with self._data_lock:
            current = self._active_streams.get(account_index, 0)
            self._active_streams[account_index] = max(0, current - 1)

    def get_active_streams(self, account_index: int) -> int:
        """Get current active stream count for an account."""
        with self._data_lock:
            return self._active_streams.get(account_index, 0)

    def _cleanup_rpm_window(self, now: float) -> None:
        """Remove RPM timestamps outside the 60s window."""
        cutoff = now - RPM_WINDOW_SECONDS
        while self._rpm_timestamps and self._rpm_timestamps[0] < cutoff:
            self._rpm_timestamps.popleft()

    def _percentile(self, values: List[float], pct: float) -> float:
        """Calculate percentile from a sorted list of values."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * pct / 100.0)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    def get_snapshot(
        self,
        accounts: Optional[list] = None,
        rate_limiter_rpm: int = 40,
    ) -> dict:
        """Build the full metrics snapshot.

        Args:
            accounts: List of NimApiAccount objects for per-account data.
            rate_limiter_rpm: Configured requests-per-minute limit per account.

        Returns:
            Complete metrics dict matching the /metrics endpoint schema.
        """
        with self._data_lock:
            now = time.time()
            self._cleanup_rpm_window(now)
            metrics_list = list(self._metrics)
            rpm_current = float(len(self._rpm_timestamps))
            rpm_peak = self._rpm_peak
            total_requests = self._total_requests
            intercept_count = self._intercept_count
            active_streams_snapshot = dict(self._active_streams)

        total_with_intercepts = total_requests + intercept_count
        interception_rate = (
            intercept_count / total_with_intercepts
            if total_with_intercepts > 0
            else 0.0
        )

        # Compute aggregate latency stats
        ttft_values = [m.ttft_ms for m in metrics_list if m.ttft_ms > 0]
        tps_values = [m.tps for m in metrics_list if m.tps > 0]
        latency_values = [m.total_latency_s for m in metrics_list if m.total_latency_s > 0]

        throughput = {
            "rpm_current": round(rpm_current, 1),
            "rpm_peak": round(rpm_peak, 1),
            "requests_total": total_requests,
            "requests_intercepted": intercept_count,
            "interception_rate": round(interception_rate, 3),
        }

        latency = {
            "ttft_avg_ms": round(statistics.mean(ttft_values), 1) if ttft_values else 0,
            "ttft_p50_ms": round(self._percentile(ttft_values, 50), 1) if ttft_values else 0,
            "ttft_p95_ms": round(self._percentile(ttft_values, 95), 1) if ttft_values else 0,
            "tps_avg": round(statistics.mean(tps_values), 1) if tps_values else 0,
            "total_latency_avg_s": round(statistics.mean(latency_values), 1) if latency_values else 0,
            "total_latency_p95_s": round(self._percentile(latency_values, 95), 1) if latency_values else 0,
        }

        # Per-account breakdown
        per_account: List[dict] = []
        if accounts:
            for acc in accounts:
                acc_metrics = [m for m in metrics_list if m.account_index == acc.index]
                acc_requests = len(acc_metrics)
                acc_ttft = [m.ttft_ms for m in acc_metrics if m.ttft_ms > 0]
                acc_tps = [m.tps for m in acc_metrics if m.tps > 0]

                # RPM for this account from recent metrics
                window_cutoff = now - RPM_WINDOW_SECONDS
                acc_rpm_count = sum(
                    1 for m in acc_metrics if m.timestamp > window_cutoff
                )

                traffic_share = (
                    (acc.request_count / total_requests * 100)
                    if total_requests > 0
                    else 0.0
                )

                # Rate limit headroom (read-only, no mutation of account state)
                cutoff = now - RPM_WINDOW_SECONDS
                requests_in_window = sum(
                    1 for ts in acc.request_timestamps if ts > cutoff
                )
                headroom = max(0, rate_limiter_rpm - requests_in_window)

                # Determine status
                if active_streams_snapshot.get(acc.index, 0) > 0:
                    status = "active"
                elif acc.circuit_breaker_state.value == "open":
                    status = "blocked"
                elif acc.circuit_breaker_state.value == "half_open":
                    status = "recovering"
                else:
                    status = "idle"

                key_hint = acc.api_key[:10] + "..." if len(acc.api_key) > 10 else "***"

                per_account.append({
                    "index": acc.index,
                    "key_hint": key_hint,
                    "status": status,
                    "active_streams": active_streams_snapshot.get(acc.index, 0),
                    "traffic_share_pct": round(traffic_share, 1),
                    "requests": acc.request_count,
                    "ttft_avg_ms": round(statistics.mean(acc_ttft), 1) if acc_ttft else 0,
                    "tps_avg": round(statistics.mean(acc_tps), 1) if acc_tps else 0,
                    "rpm_current": float(acc_rpm_count),
                    "rate_limit_headroom": headroom,
                })

        definitions = {
            "rpm_current": "Requests per minute - current sliding window (last 60s)",
            "rpm_peak": "Requests per minute - highest value observed since server start",
            "requests_total": "Total API requests made since server start",
            "requests_intercepted": "Requests handled locally without calling the API (optimizations)",
            "interception_rate": "Fraction of total requests intercepted (0.0-1.0)",
            "ttft_avg_ms": "Time to First Token - average milliseconds from request to first streaming chunk",
            "ttft_p50_ms": "Time to First Token - 50th percentile (median)",
            "ttft_p95_ms": "Time to First Token - 95th percentile (worst-case typical)",
            "tps_avg": "Tokens per Second - average output generation speed",
            "total_latency_avg_s": "Total request duration - average in seconds (start to finish)",
            "total_latency_p95_s": "Total request duration - 95th percentile in seconds",
            "key_hint": "First few characters of the API key for identification",
            "status": "Current key state: active (serving requests), idle (available but unused), recovering (warming up after failure), blocked (rate-limited or failed)",
            "active_streams": "Number of streaming responses currently in progress on this key",
            "traffic_share_pct": "Percentage of total requests handled by this key",
            "rate_limit_headroom": "Remaining requests available in current rate limit window before hitting the limit",
        }

        return {
            "throughput": throughput,
            "latency": latency,
            "per_account": per_account,
            "definitions": definitions,
        }
