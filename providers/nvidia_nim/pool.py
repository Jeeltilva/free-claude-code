"""Account pool management for NVIDIA NIM multi-account rotation."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from .account import CircuitBreakerState, NimApiAccount
from .monitoring import HealthMonitor
from .warmup import WarmupManager, WarmupConfig
from .distributed import DistributedPersistence
from .dynamic_weight import DynamicWeightAdjuster, PerformanceMetrics

logger = logging.getLogger(__name__)


class AccountPool:
    """Manages a pool of NVIDIA NIM API accounts with thread-safe access.

    Provides health tracking and account selection with asyncio.Lock
    for safe concurrent access across multiple Claude Code instances.
    """

    def __init__(
        self,
        api_keys: List[str],
        base_url: str,
        per_account_timeout: float = 300.0,
        health_state_file: str = "",
        default_weight: float = 1.0,
        weights_override: Optional[List[float]] = None,
        daily_quota_limit: int = 0,
        node_id: str = "default",
    ):
        """Initialize account pool with multiple API keys.

        Args:
            api_keys: List of NVIDIA NIM API keys.
            base_url: Base URL for NIM API endpoint.
            per_account_timeout: Timeout per account in seconds.
            health_state_file: Path to persist health state (empty = disabled).
            node_id: Node identifier for distributed deployments.
        """
        self._accounts: List[NimApiAccount] = []
        self._lock = asyncio.Lock()
        self._base_url = base_url
        self._health_state_file = health_state_file
        self._health_monitor = HealthMonitor()
        self._warmup_manager = WarmupManager()
        self._distributed = DistributedPersistence(
            node_id=node_id,
            state_file=health_state_file.replace(".json", ".distributed.json") if health_state_file else ""
        )
        self._weight_adjuster = DynamicWeightAdjuster()

        for i, key in enumerate(api_keys):
            weight = default_weight
            if weights_override and i < len(weights_override):
                weight = weights_override[i]
            account = NimApiAccount(
                api_key=key,
                base_url=base_url,
                index=i,
                timeout=per_account_timeout,
                weight=weight,
                daily_quota_limit=daily_quota_limit,
            )
            self._accounts.append(account)

        # Restore persisted health state if available
        if self._health_state_file:
            self._load_state()

    @property
    def size(self) -> int:
        """Total number of accounts in pool."""
        return len(self._accounts)

    @property
    def healthy_account_count(self) -> int:
        """Number of accounts that are healthy (not blocked, failure count low)."""
        return sum(1 for acc in self._accounts if acc.is_healthy())

    async def get_available_accounts(self, max_failures: int = 3) -> List[NimApiAccount]:
        """Get all accounts currently available for use.

        Returns accounts that are:
        - Not blocked by cooldown
        - Below maximum failure threshold

        Args:
            max_failures: Maximum failures before account is excluded.

        Returns:
            List of available accounts, sorted by availability (unblocked first).
        """
        async with self._lock:
            available = []
            for account in self._accounts:
                if account.is_healthy(max_failures) and account.is_available():
                    available.append(account)

            # Sort: unblocked accounts first, then by failure count
            available.sort(key=lambda a: (not a.is_available(), a.failure_count))
            return available

    async def get_account_by_index(self, index: int) -> Optional[NimApiAccount]:
        """Get account by its index - thread-safe.

        Args:
            index: Account index.

        Returns:
            Account if exists, None otherwise.
        """
        async with self._lock:
            if 0 <= index < len(self._accounts):
                return self._accounts[index]
            return None

    async def mark_account_failed(self, account: NimApiAccount, cooldown_seconds: int = 60, max_failures: int = 3, error_message: str = "unknown") -> None:
        """Mark an account as failed and update circuit breaker state.

        Args:
            account: Account that failed.
            cooldown_seconds: Duration for OPEN state cooldown.
            max_failures: Failure threshold for OPEN state transition.
            error_message: Description of the error that occurred.
        """
        async with self._lock:
            # Find and update the account
            for acc in self._accounts:
                if acc.index == account.index:
                    acc.mark_failed(max_failures=max_failures, failure_cooldown=cooldown_seconds)
                    # Always block for the cooldown period (backward compatible)
                    acc.block(cooldown_seconds)
                    # Record failure in health monitor
                    self._health_monitor.record_failure(acc, error_message, 0.0)

                    # Update warmup manager
                    if hasattr(self, '_warmup_manager'):
                        factor = self._warmup_manager.update_warmup_on_failure(acc.index, error_message)
                        acc.warmup_factor = factor
                    break

    async def mark_account_success(self, account: NimApiAccount, half_open_success_threshold: int = 3, latency: float = 0.0) -> None:
        """Mark account as successful - updates failure count, circuit breaker state, and warmup.

        Args:
            account: Account that succeeded.
            half_open_success_threshold: Successes needed to transition from HALF_OPEN to CLOSED.
            latency: Request latency in seconds.
        """
        async with self._lock:
            for acc in self._accounts:
                if acc.index == account.index:
                    acc.reset_failures(half_open_success_threshold=half_open_success_threshold)

                    # Update warmup manager
                    if hasattr(self, '_warmup_manager'):
                        factor = self._warmup_manager.update_warmup_on_success(acc.index, latency)
                        acc.warmup_factor = factor
                    break

    async def update_account_usage(self, account: NimApiAccount) -> None:
        """Update account last_used timestamp, request count, and daily quota.

        Args:
            account: Account that was used.
        """
        async with self._lock:
            for acc in self._accounts:
                if acc.index == account.index:
                    acc.mark_used()
                    acc.add_request_timestamp()
                    acc.increment_daily_usage()
                    # Record successful request in health monitor
                    self._health_monitor.record_request(acc, success=True, latency=0.0)
                    break

    async def record_account_latency(self, account: NimApiAccount, latency_seconds: float) -> None:
        """Record response latency for an account.

        Args:
            account: Account that completed the request.
            latency_seconds: Response time in seconds.
        """
        async with self._lock:
            for acc in self._accounts:
                if acc.index == account.index:
                    acc.record_latency(latency_seconds)
                    # Record latency in health monitor
                    self._health_monitor.record_request(acc, success=True, latency=latency_seconds)
                    break

    def get_all_accounts(self) -> List[NimApiAccount]:
        """Get all accounts (read-only, no lock needed for iteration).

        Returns:
            List of all account objects.
        """
        return self._accounts.copy()

    async def get_health_status(self) -> dict:
        """Get health status of all accounts with monitoring data.

        Returns:
            Dict with health statistics and monitoring information.
        """
        async with self._lock:
            total = len(self._accounts)
            available = sum(1 for a in self._accounts if a.is_available())
            healthy = sum(1 for a in self._accounts if a.is_healthy())
            blocked = total - available

            account_details = []
            for acc in self._accounts:
                # Get health alerts for this account
                alerts = self._health_monitor.get_health_alerts(acc)

                # Get performance metrics for dynamic weight calculation
                failure_rate_5min = self._health_monitor.get_failure_rate(acc, 300)
                success_rate = 1.0 - failure_rate_5min
                avg_latency_5min = self._health_monitor.get_avg_latency(acc, 300)

                metrics = PerformanceMetrics(
                    avg_latency=avg_latency_5min,
                    success_rate=success_rate,
                    request_count=acc.request_count,
                    failure_count=acc.failure_count,
                    latency_samples=acc.latency_samples,
                )

                # Calculate dynamic weight
                dynamic_weight = self._weight_adjuster.calculate_dynamic_weight(
                    acc.index, acc.weight, metrics
                )

                account_details.append({
                    "index": acc.index,
                    "available": acc.is_available(),
                    "healthy": acc.is_healthy(),
                    "request_count": acc.request_count,
                    "failure_count": acc.failure_count,
                    "avg_latency": acc.avg_latency,
                    "circuit_breaker_state": acc.circuit_breaker_state.value,
                    "half_open_probe_count": acc.half_open_probe_count,
                    "consecutive_successes": acc.consecutive_successes,
                    "blocked_until": acc.blocked_until,
                    "alerts": alerts,
                    "failure_rate_5min": failure_rate_5min,
                    "avg_latency_5min": avg_latency_5min,
                    "static_weight": acc.weight,
                    "dynamic_weight": dynamic_weight,
                })

            # Get pool health summary
            pool_summary = self._health_monitor.get_pool_health_summary(self._accounts)

            return {
                "total": total,
                "available": available,
                "healthy": healthy,
                "blocked": blocked,
                "accounts": account_details,
                "pool_health_summary": pool_summary,
            }

    async def update_dynamic_weights(self) -> None:
        """Update dynamic weights for all accounts based on current performance."""
        async with self._lock:
            for acc in self._accounts:
                # Calculate performance metrics
                failure_rate_5min = self._health_monitor.get_failure_rate(acc, 300)
                success_rate = 1.0 - failure_rate_5min
                avg_latency_5min = self._health_monitor.get_avg_latency(acc, 300)

                metrics = PerformanceMetrics(
                    avg_latency=avg_latency_5min,
                    success_rate=success_rate,
                    request_count=acc.request_count,
                    failure_count=acc.failure_count,
                    latency_samples=acc.latency_samples,
                )

                # Update dynamic weight
                dynamic_weight = self._weight_adjuster.calculate_dynamic_weight(
                    acc.index, acc.weight, metrics
                )

                # For now, we'll update the static weight to the dynamic weight
                # In a more sophisticated implementation, we might keep them separate
                acc.weight = dynamic_weight

    def _load_state(self) -> None:
        """Load persisted health state from disk."""
        path = Path(self._health_state_file)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for entry in data.get("accounts", []):
                idx = entry.get("index", -1)
                if 0 <= idx < len(self._accounts):
                    acc = self._accounts[idx]
                    acc.request_count = entry.get("request_count", 0)
                    acc.failure_count = entry.get("failure_count", 0)
                    acc.blocked_until = entry.get("blocked_until", 0.0)
                    acc.last_failure = entry.get("last_failure", 0.0)
                    # Load additional fields
                    acc.daily_request_count = entry.get("daily_request_count", 0)
                    acc.daily_reset_time = entry.get("daily_reset_time", 0.0)
                    acc.warmup_factor = entry.get("warmup_factor", 1.0)
                    acc.avg_latency = entry.get("avg_latency", 0.0)
                    # Load circuit breaker state
                    state_str = entry.get("circuit_breaker_state", "closed")
                    if state_str == "open":
                        acc.circuit_breaker_state = CircuitBreakerState.OPEN
                    elif state_str == "half_open":
                        acc.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                    else:
                        acc.circuit_breaker_state = CircuitBreakerState.CLOSED
            logger.info(f"Restored health state from {self._health_state_file}")
        except Exception as e:
            logger.warning(f"Failed to load health state: {e}")

    def save_state(self) -> None:
        """Persist current health state to disk and update distributed state."""
        if not self._health_state_file:
            return

        try:
            # Prepare account data for both local and distributed storage
            accounts_data = []
            for acc in self._accounts:
                account_entry = {
                    "index": acc.index,
                    "request_count": acc.request_count,
                    "failure_count": acc.failure_count,
                    "blocked_until": acc.blocked_until,
                    "last_failure": acc.last_failure,
                    "daily_request_count": acc.daily_request_count,
                    "daily_reset_time": acc.daily_reset_time,
                    "warmup_factor": acc.warmup_factor,
                    "avg_latency": acc.avg_latency,
                    "circuit_breaker_state": acc.circuit_breaker_state.value,
                    "half_open_probe_count": acc.half_open_probe_count,
                    "consecutive_successes": acc.consecutive_successes,
                }
                accounts_data.append(account_entry)

                # Update distributed state
                self._distributed.update_account_state(acc.index, account_entry)

            # Save local state
            data = {"accounts": accounts_data}
            path = Path(self._health_state_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            # Save distributed state
            self._distributed.save_local_state()

        except Exception as e:
            logger.warning(f"Failed to save health state: {e}")

    async def add_account(self, api_key: str) -> int:
        """Add a new account to the pool at runtime.

        Args:
            api_key: NVIDIA NIM API key for the new account.

        Returns:
            Index of the newly added account.
        """
        async with self._lock:
            # Get timeout from existing accounts or use default
            per_account_timeout = 300.0
            if self._accounts:
                per_account_timeout = self._accounts[0].timeout

            # Create new account with next index
            new_index = len(self._accounts)
            account = NimApiAccount(
                api_key=api_key,
                base_url=self._base_url,
                index=new_index,
                timeout=per_account_timeout,
            )
            self._accounts.append(account)

            logger.info(f"POOL: Added account {new_index} (total: {len(self._accounts)})")
            return new_index

    async def remove_account(self, index: int) -> bool:
        """Remove an account from the pool by index.

        Args:
            index: Index of account to remove.

        Returns:
            True if account was found and removed, False otherwise.
        """
        async with self._lock:
            # Find account by index
            account = None
            for i, acc in enumerate(self._accounts):
                if acc.index == index:
                    account = acc
                    break

            if account is None:
                return False

            # Close the client
            if account.client:
                await account.client.close()
                account.client = None

            # Remove from list
            self._accounts.remove(account)

            logger.info(f"POOL: Removed account {index} (remaining: {len(self._accounts)})")
            return True

    async def validate_accounts(self) -> list:
        """Validate all accounts by making lightweight test requests.

        Returns:
            List of dicts with validation results:
            [{"index": 0, "valid": True, "error": None}, ...]
        """
        results = []

        for account in self._accounts:
            try:
                # Make a lightweight completion request (1 token)
                response = await asyncio.wait_for(
                    account.client.chat.completions.create(
                        model="meta/llama-3.1-8b-instruct",
                        messages=[{"role": "user", "content": "Hi"}],
                        max_tokens=1,
                    ),
                    timeout=10.0,
                )
                results.append({
                    "index": account.index,
                    "valid": True,
                    "error": None,
                })
                logger.info(f"POOL: Account {account.index} validation passed")
            except asyncio.TimeoutError:
                results.append({
                    "index": account.index,
                    "valid": False,
                    "error": "Request timed out after 10s",
                })
                logger.warning(f"POOL: Account {account.index} validation timeout")
            except Exception as e:
                results.append({
                    "index": account.index,
                    "valid": False,
                    "error": str(e),
                })
                logger.warning(f"POOL: Account {account.index} validation failed: {e}")

        return results

    async def close_all(self) -> None:
        """Close all account clients and persist state on shutdown."""
        self.save_state()
        async with self._lock:
            for account in self._accounts:
                if account.client:
                    await account.client.close()
                    account.client = None

    def __len__(self) -> int:
        """Return number of accounts in pool."""
        return len(self._accounts)

    def __iter__(self):
        """Iterate over accounts."""
        return iter(self._accounts)
