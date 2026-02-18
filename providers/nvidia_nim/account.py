"""NIM API Account dataclass with health tracking."""

import datetime
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from openai import AsyncOpenAI


class CircuitBreakerState(Enum):
    """Circuit breaker states for account health management."""

    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Account blocked due to failures
    HALF_OPEN = "half_open"  # Testing account with limited probe requests


@dataclass
class NimApiAccount:
    """Represents a single NVIDIA NIM API account with health tracking.

    Each account maintains its own AsyncOpenAI client and tracks usage
    statistics for intelligent rotation decisions.
    """

    api_key: str
    base_url: str
    index: int  # Position in the accounts list

    # Health tracking
    request_count: int = field(default=0)
    failure_count: int = field(default=0)
    last_used: float = field(default=0.0)
    blocked_until: float = field(default=0.0)
    last_failure: float = field(default=0.0)

    # Circuit breaker state
    circuit_breaker_state: CircuitBreakerState = field(default=CircuitBreakerState.CLOSED)
    half_open_probe_count: int = field(default=0)  # Count of probes in HALF_OPEN state
    consecutive_successes: int = field(default=0)  # Success count for HALF_OPEN -> CLOSED transition

    # Per-account timeout
    timeout: float = field(default=300.0)

    # Rate limiting state (sliding window)
    request_timestamps: list = field(default_factory=list)

    # Latency tracking
    avg_latency: float = field(default=0.0)
    latency_samples: int = field(default=0)

    # Weighted routing (higher weight = more traffic)
    weight: float = field(default=1.0)

    # Daily quota tracking
    daily_request_count: int = field(default=0)
    daily_quota_limit: int = field(default=0)  # 0 = unlimited
    daily_reset_time: float = field(default=0.0)

    # Warm-up state (gradual ramp after recovery)
    warmup_factor: float = field(default=1.0)  # 0.0-1.0, starts low after recovery
    needs_warmup: bool = field(default=False)  # Internal flag for reset on next success

    # Per-account client
    client: Optional[AsyncOpenAI] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize the AsyncOpenAI client for this account."""
        if self.client is None:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                max_retries=0,
                timeout=self.timeout,
            )
        # Initialize daily quota reset time if quota limit is set
        if self.daily_quota_limit > 0 and self.daily_reset_time == 0:
            self.daily_reset_time = self._get_next_reset_time()

    def transition_to_closed(self) -> None:
        """Transition to CLOSED state - normal operation."""
        self.circuit_breaker_state = CircuitBreakerState.CLOSED
        self.half_open_probe_count = 0
        self.consecutive_successes = 0
        self.failure_count = 0
        self.blocked_until = 0.0

    def transition_to_open(self, duration_seconds: int) -> None:
        """Transition to OPEN state - account blocked due to failures."""
        self.circuit_breaker_state = CircuitBreakerState.OPEN
        self.half_open_probe_count = 0
        self.consecutive_successes = 0
        self.blocked_until = time.time() + duration_seconds

    def transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state - testing account with limited probes."""
        self.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
        self.half_open_probe_count = 0
        self.consecutive_successes = 0
        self.blocked_until = 0.0

    def is_available(self) -> bool:
        """Check if account is currently available for use.

        Returns:
            True if account can be used (not blocked, not in OPEN state, quota available).
        """
        now = time.time()

        # In OPEN state, check if cooldown has expired
        if self.circuit_breaker_state == CircuitBreakerState.OPEN:
            if now <= self.blocked_until:
                return False
            # Cooldown expired, transition to HALF_OPEN for testing
            self.transition_to_half_open()
            return True

        # In HALF_OPEN state, account is available but with limited probing
        if self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            return True

        # In CLOSED state, check blocked_until and daily quota
        if now <= self.blocked_until:
            return False
        return self.check_daily_quota()

    def is_healthy(self, max_failures: int = 3) -> bool:
        """Check if account is healthy based on circuit breaker state and failure count.

        Auto-recovers accounts whose cooldown has expired by transitioning to HALF_OPEN.

        Args:
            max_failures: Maximum failures before considering unhealthy.

        Returns:
            True if account is considered healthy for use.
        """
        # In OPEN state, account is unhealthy until cooldown expires
        if self.circuit_breaker_state == CircuitBreakerState.OPEN:
            now = time.time()
            if now <= self.blocked_until:
                return False
            # Cooldown expired, transition to HALF_OPEN for testing
            self.transition_to_half_open()
            return True

        # In HALF_OPEN state, account is conditionally healthy
        if self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            return True

        # In CLOSED state, check failure count with auto-recovery
        if self.failure_count >= max_failures:
            # Auto-recover if cooldown has expired
            if time.time() > self.blocked_until:
                self.failure_count = 0
                return True
            return False
        return True

    def mark_used(self) -> None:
        """Mark account as just used - update last_used and request count."""
        self.last_used = time.time()
        self.request_count += 1

    def mark_failed(self, max_failures: int = 3, failure_cooldown: int = 60) -> None:
        """Mark account as having a failure and update circuit breaker state.

        Args:
            max_failures: Threshold for transitioning to OPEN state.
            failure_cooldown: Duration in seconds for OPEN state cooldown.
        """
        now = time.time()
        self.failure_count += 1
        self.last_failure = now
        self.needs_warmup = True

        # If we exceed failure threshold, transition to OPEN state
        if self.failure_count >= max_failures and self.circuit_breaker_state != CircuitBreakerState.OPEN:
            self.transition_to_open(failure_cooldown)

    def block(self, duration_seconds: int) -> None:
        """Block account for specified duration.

        Args:
            duration_seconds: How long to block the account.
        """
        self.blocked_until = time.time() + duration_seconds

    def reset_failures(self, half_open_success_threshold: int = 3) -> None:
        """Reset failure count and update circuit breaker state based on success.

        Args:
            half_open_success_threshold: Number of consecutive successes needed to
                                       transition from HALF_OPEN to CLOSED.
        """
        if self.needs_warmup:
            self.warmup_factor = 0.1
            self.needs_warmup = False

        self.failure_count = 0
        self.last_failure = 0.0

        # Handle circuit breaker state transitions on success
        if self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            self.consecutive_successes += 1
            # If we have enough consecutive successes, transition to CLOSED
            if self.consecutive_successes >= half_open_success_threshold:
                self.transition_to_closed()
        elif self.circuit_breaker_state == CircuitBreakerState.OPEN:
            # If somehow we get a success while in OPEN state, transition to HALF_OPEN
            self.transition_to_half_open()

    def cleanup_old_timestamps(self, window_seconds: int = 60) -> None:
        """Remove timestamps older than window for rate limiting.

        Args:
            window_seconds: Size of sliding window in seconds.
        """
        cutoff = time.time() - window_seconds
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > cutoff]

    def get_request_count_in_window(self, window_seconds: int = 60) -> int:
        """Count requests in the sliding window.

        Args:
            window_seconds: Size of sliding window in seconds.

        Returns:
            Number of requests made within the window.
        """
        self.cleanup_old_timestamps(window_seconds)
        return len(self.request_timestamps)

    def add_request_timestamp(self) -> None:
        """Record timestamp of current request for rate limiting."""
        self.request_timestamps.append(time.time())

    def record_latency(self, latency_seconds: float) -> None:
        """Record response latency using exponential moving average.

        Updates the average latency with alpha=0.3 for smoothing.

        Args:
            latency_seconds: Response time in seconds.
        """
        alpha = 0.3
        if self.latency_samples == 0:
            self.avg_latency = latency_seconds
        else:
            self.avg_latency = alpha * latency_seconds + (1 - alpha) * self.avg_latency
        self.latency_samples += 1

    def get_effective_weight(self) -> float:
        """Get effective weight considering warmup factor (weight * warmup_factor).

        Returns:
            Effective weight for routing decisions.
        """
        return self.weight * max(0.0, min(1.0, self.warmup_factor))

    def check_daily_quota(self) -> bool:
        """Check if account has remaining daily quota.

        Returns:
            True if quota is unlimited (0) or under limit.
        """
        if self.daily_quota_limit == 0:
            return True
        if self.daily_reset_time > time.time():
            # Still within current day, check count
            return self.daily_request_count < self.daily_quota_limit
        # New day, reset count automatically
        self.daily_request_count = 0
        self.daily_reset_time = self._get_next_reset_time()
        return True

    def increment_daily_usage(self) -> None:
        """Increment daily request count. Should be called when request is made."""
        self.daily_request_count += 1

    def _get_next_reset_time(self) -> float:
        """Get timestamp for next daily reset (midnight)."""
        import datetime
        now = datetime.datetime.now()
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
        return tomorrow.timestamp()

    def reset_warmup(self) -> None:
        """Reset warmup to 0.1 for gradual ramp-up after a cooldown."""
        self.warmup_factor = 0.1

    def increase_warmup(self, increment: float = 0.1) -> None:
        """Increase warmup factor gradually up to 1.0."""
        self.warmup_factor = min(1.0, self.warmup_factor + increment)

    def can_probe_in_half_open(self, max_probes: int = 5) -> bool:
        """Check if account can be used for a probe request in HALF_OPEN state.

        Args:
            max_probes: Maximum number of probe requests allowed in HALF_OPEN state.

        Returns:
            True if account can be used for probing.
        """
        if self.circuit_breaker_state != CircuitBreakerState.HALF_OPEN:
            return True  # Not in HALF_OPEN state, so can be used normally

        # In HALF_OPEN state, limit probe requests
        if self.half_open_probe_count < max_probes:
            self.half_open_probe_count += 1
            return True
        return False

    def __str__(self) -> str:
        """String representation masking API key."""
        key_preview = self.api_key[:8] + "..." if len(self.api_key) > 8 else "***"
        return f"NimApiAccount(index={self.index}, key={key_preview}, requests={self.request_count}, failures={self.failure_count})"
