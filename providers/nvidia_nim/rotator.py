"""Account rotator with rate limiting and rotation strategies for NVIDIA NIM."""

import asyncio
import logging
import time
from enum import Enum
from typing import Callable, List, Optional, TypeVar

from openai import APIError, RateLimitError

from .account import CircuitBreakerState, NimApiAccount
from .pool import AccountPool

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AllAccountsExhaustedError(Exception):
    """Raised when all accounts are unhealthy or rate-limited."""

    def __init__(self, total: int = 0, healthy: int = 0):
        self.total = total
        self.healthy = healthy
        super().__init__(
            f"All {total} accounts exhausted ({healthy} healthy). "
            "Try again later or add more API keys."
        )


class RotationStrategy(Enum):
    """Strategy for selecting next account."""

    ROUND_ROBIN = "round_robin"
    LEAST_USED = "least_used"
    ON_FAILURE = "on_failure"


class PerAccountRateLimiter:
    """Sliding window rate limiter for individual accounts.

    Tracks requests per minute and blocks accounts that exceed limits.
    """

    def __init__(self, requests_per_minute: int = 40):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute.
        """
        self.requests_per_minute = requests_per_minute
        self._window_seconds = 60

    def is_allowed(self, account: NimApiAccount) -> bool:
        """Check if account can make a request under rate limit.

        Args:
            account: Account to check.

        Returns:
            True if request is allowed.
        """
        effective_limit = self.requests_per_minute * account.get_effective_weight()
        account.cleanup_old_timestamps(self._window_seconds)
        return len(account.request_timestamps) < effective_limit

    def get_wait_time(self, account: NimApiAccount) -> float:
        """Get seconds to wait before account is available.

        Args:
            account: Account to check.

        Returns:
            Seconds to wait (0 if available now).
        """
        if self.is_allowed(account):
            return 0.0

        account.cleanup_old_timestamps(self._window_seconds)
        if not account.request_timestamps:
            return 0.0

        # Time until oldest timestamp falls out of window
        oldest = min(account.request_timestamps)
        return max(0.0, (oldest + self._window_seconds) - time.time())


class AccountRotator:
    """Rotates through accounts with configurable strategies and automatic failover.

    Supports multiple rotation strategies:
    - round_robin: Cycles through accounts in order
    - least_used: Picks account with oldest last_used timestamp
    - on_failure: Stays on one account until it fails, then switches
    """

    def __init__(
        self,
        pool: AccountPool,
        strategy: RotationStrategy = RotationStrategy.ON_FAILURE,
        requests_per_minute: int = 40,
        max_failures: int = 3,
        failure_cooldown: int = 60,
        max_retries: int = 3,
        warmup_enabled: bool = True,
        warmup_increment: float = 0.1,
        half_open_success_threshold: int = 3,
        half_open_max_probes: int = 5,
    ):
        """Initialize account rotator.

        Args:
            pool: Account pool to rotate through.
            strategy: Rotation strategy to use.
            requests_per_minute: Rate limit per account.
            max_failures: Max failures before marking account unhealthy.
            failure_cooldown: Seconds to block account after failure.
            max_retries: Max retry attempts for execute_with_rotation.
            half_open_success_threshold: Successes needed to transition from HALF_OPEN to CLOSED.
            half_open_max_probes: Max probe requests allowed in HALF_OPEN state.
        """
        self._pool = pool
        self._strategy = strategy
        self._rate_limiter = PerAccountRateLimiter(requests_per_minute)
        self._max_failures = max_failures
        self._failure_cooldown = failure_cooldown
        self._max_retries = max_retries
        self._warmup_enabled = warmup_enabled
        self._warmup_increment = warmup_increment
        self._half_open_success_threshold = half_open_success_threshold
        self._half_open_max_probes = half_open_max_probes

        # Round-robin state
        self._rr_index = 0

        # Current account for sticky strategies
        self._current_account: Optional[NimApiAccount] = None

        # Lock for thread safety
        self._lock = asyncio.Lock()

    def _extract_cooldown(self, error: RateLimitError) -> int:
        """Extract cooldown duration from RateLimitError headers.

        Tries to get retry-after or x-ratelimit-reset from response headers,
        falling back to configured failure_cooldown.

        Args:
            error: The RateLimitError with response headers.

        Returns:
            Cooldown duration in seconds (clamped between 1 and 300).
        """
        try:
            if hasattr(error, 'response') and error.response is not None:
                headers = error.response.headers

                # Try retry-after header (seconds to wait)
                if 'retry-after' in headers:
                    retry_after = int(headers['retry-after'])
                    return max(1, min(retry_after, 300))

                # Try x-ratelimit-reset (Unix timestamp)
                if 'x-ratelimit-reset' in headers:
                    reset_time = int(headers['x-ratelimit-reset'])
                    cooldown = max(1, reset_time - int(time.time()))
                    return min(cooldown, 300)

        except (ValueError, AttributeError, KeyError):
            pass

        # Fallback to configured cooldown
        return self._failure_cooldown

    async def _get_next_round_robin(self) -> Optional[NimApiAccount]:
        """Get next account using round-robin strategy."""
        accounts = await self._pool.get_available_accounts(self._max_failures)
        if not accounts:
            return None

        async with self._lock:
            # Find next available account starting from current index
            for _ in range(len(accounts)):
                idx = self._rr_index % len(accounts)
                self._rr_index = (self._rr_index + 1) % len(accounts)

                account = accounts[idx]
                if self._rate_limiter.is_allowed(account):
                    # For HALF_OPEN accounts, check if probing is allowed
                    if account.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
                        if account.can_probe_in_half_open():
                            return account
                    else:
                        return account

        # No account available under rate limit
        return None

    async def _get_least_used(self) -> Optional[NimApiAccount]:
        """Get account with oldest last_used timestamp.

        When multiple accounts have similar last_used times (within 5 seconds),
        prefers the one with lower average latency.
        For HALF_OPEN accounts, respects probe limits.
        """
        accounts = await self._pool.get_available_accounts(self._max_failures)
        if not accounts:
            return None

        # Sort by last_used (oldest first) and rate limit status
        async with self._lock:
            # Filter to only rate-limited-available accounts
            available = [a for a in accounts if self._rate_limiter.is_allowed(a)]

        if not available:
            return None

        # Filter HALF_OPEN accounts by probe eligibility
        probe_eligible = []
        for account in available:
            if account.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
                # For HALF_OPEN accounts, check if probing is allowed
                if account.can_probe_in_half_open():
                    probe_eligible.append(account)
            else:
                probe_eligible.append(account)

        if not probe_eligible:
            return None

        # Find oldest used among eligible accounts
        oldest = min(probe_eligible, key=lambda a: a.last_used)

        # Check for accounts with similar last_used time (within 5 seconds)
        similar_age = [
            a for a in probe_eligible
            if abs(a.last_used - oldest.last_used) <= 5.0
        ]

        # If multiple accounts have similar age, prefer lower latency
        if len(similar_age) > 1:
            return min(similar_age, key=lambda a: (a.avg_latency if a.latency_samples > 0 else float('inf')))

        return oldest

    async def _get_on_failure(self) -> Optional[NimApiAccount]:
        """Get account using sticky-with-failover strategy.

        Holds lock for the entire check-and-assign to prevent race conditions
        where two concurrent requests both switch to different accounts.
        Respects HALF_OPEN probe limits.
        """
        async with self._lock:
            # Check if current account is still good
            if self._current_account:
                account = await self._pool.get_account_by_index(self._current_account.index)
                if account and account.is_available() and account.is_healthy(self._max_failures):
                    if self._rate_limiter.is_allowed(account):
                        # For HALF_OPEN accounts, check if probing is allowed
                        if account.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
                            if account.can_probe_in_half_open():
                                return account
                        else:
                            return account

            # Current account not available, pick new one
            accounts = await self._pool.get_available_accounts(self._max_failures)
            if not accounts:
                return None

            # Find first account under rate limit that can be probed if HALF_OPEN
            for account in accounts:
                if self._rate_limiter.is_allowed(account):
                    # For HALF_OPEN accounts, check if probing is allowed
                    if account.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
                        if not account.can_probe_in_half_open():
                            continue  # Skip this account, reached probe limit

                    old_index = self._current_account.index if self._current_account else None
                    self._current_account = account
                    if old_index is not None:
                        logger.info(
                            f"ROTATOR: Failover from account {old_index} to {account.index}"
                        )
                    return account

            return None

    async def get_next_account(self) -> Optional[NimApiAccount]:
        """Get next available account based on rotation strategy.

        Returns:
            Available account or None if all blocked/rate-limited.
        """
        if self._strategy == RotationStrategy.ROUND_ROBIN:
            return await self._get_next_round_robin()
        elif self._strategy == RotationStrategy.LEAST_USED:
            return await self._get_least_used()
        else:  # ON_FAILURE
            return await self._get_on_failure()

    async def get_any_available(self, wait: bool = True, max_wait: float = 30.0) -> NimApiAccount:
        """Get any available account, optionally waiting for rate limit reset.

        Args:
            wait: Whether to wait if all accounts are rate limited.
            max_wait: Maximum seconds to wait.

        Returns:
            Available account.

        Raises:
            AllAccountsExhaustedError: If no account available after waiting.
        """
        start_time = time.time()

        while True:
            account = await self.get_next_account()
            if account:
                return account

            all_accounts = self._pool.get_all_accounts()
            health = await self._pool.get_health_status()

            if not wait:
                raise AllAccountsExhaustedError(
                    total=health["total"], healthy=health["healthy"]
                )

            # Check if we should continue waiting
            elapsed = time.time() - start_time
            if elapsed >= max_wait:
                logger.warning(
                    f"ROTATOR: All accounts exhausted after {elapsed:.1f}s wait "
                    f"(total={health['total']}, healthy={health['healthy']})"
                )
                raise AllAccountsExhaustedError(
                    total=health["total"], healthy=health["healthy"]
                )

            # Calculate minimum wait time across all accounts
            wait_times = [self._rate_limiter.get_wait_time(a) for a in all_accounts]
            min_wait = min((w for w in wait_times if w > 0), default=1.0)

            # Don't exceed max_wait
            sleep_time = min(min_wait, max_wait - elapsed, 5.0)
            logger.debug(
                f"ROTATOR: All accounts busy, sleeping {sleep_time:.1f}s"
            )

            await asyncio.sleep(sleep_time)

    async def execute_with_rotation(
        self,
        operation: Callable[[NimApiAccount], T],
        retry_delay: float = 1.0,
    ) -> T:
        """Execute operation with automatic retry and account failover.

        Automatically rotates to next account on rate limit (429) errors
        and retries on transient errors. Records latency on success.

        Args:
            operation: Async function that takes an account and returns result.
            retry_delay: Seconds to wait between retries.

        Returns:
            Result from operation.

        Raises:
            AllAccountsExhaustedError: If no accounts available.
            Exception: If all retries exhausted.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            account = await self.get_any_available(wait=True)

            logger.info(
                f"ROTATOR: Attempt {attempt + 1}/{self._max_retries} "
                f"using account {account.index}"
            )

            try:
                await self._pool.update_account_usage(account)

                start_time = time.time()
                result = await operation(account)
                elapsed = time.time() - start_time

                await self._pool.record_account_latency(account, elapsed)
                await self._pool.mark_account_success(
                    account,
                    half_open_success_threshold=getattr(self, '_half_open_success_threshold', 3),
                    latency=elapsed
                )
                # Note: warmup factor is now managed by the warmup manager
                return result

            except RateLimitError as e:
                cooldown = self._extract_cooldown(e)
                logger.warning(
                    f"ROTATOR: Account {account.index} rate-limited, "
                    f"cooldown={cooldown}s, failing over"
                )
                await self._pool.mark_account_failed(account, cooldown)
                last_error = e
                await asyncio.sleep(0.1)

            except APIError as e:
                logger.warning(
                    f"ROTATOR: Account {account.index} API error: {e}, failing over"
                )
                await self._pool.mark_account_failed(account, self._failure_cooldown // 2)
                last_error = e
                await asyncio.sleep(retry_delay * (2 ** attempt))

            except Exception as e:
                logger.error(
                    f"ROTATOR: Account {account.index} unexpected error: {type(e).__name__}: {e}"
                )
                await self._pool.mark_account_failed(account, self._failure_cooldown)
                last_error = e
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))

        if last_error:
            raise last_error
        raise RuntimeError("All retry attempts failed")

    async def execute_streaming_with_rotation(
        self,
        operation: Callable[[NimApiAccount], T],
    ):
        """Execute streaming operation with automatic retry and failover.

        This is a generator that yields from the streaming operation,
        with automatic failover on errors. Records latency on success.

        Args:
            operation: Async generator function that takes an account.

        Yields:
            Items from the streaming operation.

        Raises:
            AllAccountsExhaustedError: If no accounts available.
            Exception: If all retries exhausted.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            account = await self.get_any_available(wait=True)

            logger.info(
                f"ROTATOR: Stream attempt {attempt + 1}/{self._max_retries} "
                f"using account {account.index}"
            )

            try:
                await self._pool.update_account_usage(account)

                start_time = time.time()
                async for item in operation(account):
                    yield item
                elapsed = time.time() - start_time

                await self._pool.record_account_latency(account, elapsed)
                await self._pool.mark_account_success(
                    account,
                    half_open_success_threshold=getattr(self, '_half_open_success_threshold', 3),
                    latency=elapsed
                )
                # Note: warmup factor is now managed by the warmup manager
                return

            except RateLimitError as e:
                cooldown = self._extract_cooldown(e)
                logger.warning(
                    f"ROTATOR: Account {account.index} rate-limited during stream, "
                    f"cooldown={cooldown}s, failing over"
                )
                await self._pool.mark_account_failed(account, cooldown)
                last_error = e
                await asyncio.sleep(0.1)

            except APIError as e:
                logger.warning(
                    f"ROTATOR: Account {account.index} API error during stream: {e}"
                )
                await self._pool.mark_account_failed(account, self._failure_cooldown // 2)
                last_error = e
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(1.0 * (2 ** attempt))

            except Exception as e:
                logger.error(
                    f"ROTATOR: Account {account.index} unexpected error during stream: "
                    f"{type(e).__name__}: {e}"
                )
                await self._pool.mark_account_failed(account, self._failure_cooldown)
                last_error = e
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(1.0 * (2 ** attempt))

        if last_error:
            raise last_error
        raise RuntimeError("All retry attempts failed for streaming operation")

    def set_strategy(self, strategy: RotationStrategy) -> None:
        """Change rotation strategy.

        Args:
            strategy: New strategy to use.
        """
        self._strategy = strategy

    async def get_status(self) -> dict:
        """Get current rotator status.

        Returns:
            Dict with rotator state and account status.
        """
        pool_health = await self._pool.get_health_status()

        async with self._lock:
            current_key = "***"
            if self._current_account:
                current_key = self._current_account.api_key[:8] + "..."

        return {
            "strategy": self._strategy.value,
            "current_account": current_key if self._strategy == RotationStrategy.ON_FAILURE else None,
            "round_robin_index": self._rr_index if self._strategy == RotationStrategy.ROUND_ROBIN else None,
            "rate_limit": self._rate_limiter.requests_per_minute,
            "pool_health": pool_health,
        }
