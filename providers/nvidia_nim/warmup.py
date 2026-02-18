"""Intelligent warm-up management for NVIDIA NIM accounts."""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any
from enum import Enum


class WarmupProfile(Enum):
    """Predefined warm-up profiles."""

    CONSERVATIVE = "conservative"  # Slow, cautious ramp-up
    AGGRESSIVE = "aggressive"      # Fast ramp-up
    ADAPTIVE = "adaptive"          # Performance-based pacing
    MANUAL = "manual"              # Manual control only


class WarmupState(Enum):
    """Warm-up state progression."""

    INACTIVE = "inactive"      # Warm-up not needed
    INITIATED = "initiated"    # Warm-up started after recovery
    WARMING_UP = "warming_up"  # Actively warming up
    COMPLETE = "complete"      # Warm-up finished


@dataclass
class WarmupConfig:
    """Configuration for warm-up behavior."""

    profile: WarmupProfile = WarmupProfile.ADAPTIVE
    initial_factor: float = 0.1    # Starting warm-up factor (0.0-1.0)
    increment: float = 0.1         # Factor increase per success
    max_factor: float = 1.0        # Maximum factor
    min_successes: int = 5         # Successes needed for adaptive profiles
    max_duration: int = 300        # Max warm-up duration in seconds


@dataclass
class WarmupMetrics:
    """Metrics tracked during warm-up."""

    start_time: float = 0.0
    successes: int = 0
    failures: int = 0
    current_factor: float = 1.0
    state: WarmupState = WarmupState.INACTIVE
    latency_history: list = field(default_factory=list)  # [(timestamp, latency), ...]
    error_history: list = field(default_factory=list)    # [(timestamp, error), ...]


class WarmupManager:
    """Manages intelligent warm-up for recovered accounts."""

    def __init__(self):
        """Initialize warm-up manager."""
        self._configs: Dict[int, WarmupConfig] = {}
        self._metrics: Dict[int, WarmupMetrics] = {}
        self._performance_baselines: Dict[int, float] = {}  # Baseline latencies

    def set_account_config(self, account_index: int, config: WarmupConfig) -> None:
        """Set warm-up configuration for an account.

        Args:
            account_index: Account index
            config: Warm-up configuration
        """
        self._configs[account_index] = config
        if account_index not in self._metrics:
            self._metrics[account_index] = WarmupMetrics()

    def get_account_config(self, account_index: int) -> Optional[WarmupConfig]:
        """Get warm-up configuration for an account.

        Args:
            account_index: Account index

        Returns:
            Warm-up configuration or None if not set
        """
        return self._configs.get(account_index)

    def initiate_warmup(self, account_index: int) -> None:
        """Initiate warm-up process for an account.

        Args:
            account_index: Account index
        """
        config = self._configs.get(account_index, WarmupConfig())
        metrics = self._metrics.setdefault(account_index, WarmupMetrics())

        metrics.state = WarmupState.INITIATED
        metrics.start_time = time.time()
        metrics.successes = 0
        metrics.failures = 0
        metrics.current_factor = config.initial_factor

    def update_warmup_on_success(self, account_index: int, latency: float) -> float:
        """Update warm-up state on successful request.

        Args:
            account_index: Account index
            latency: Request latency in seconds

        Returns:
            Updated warm-up factor (0.0-1.0)
        """
        if account_index not in self._configs or account_index not in self._metrics:
            return 1.0  # No warm-up configured, full capacity

        config = self._configs[account_index]
        metrics = self._metrics[account_index]

        # Record latency
        metrics.latency_history.append((time.time(), latency))

        # Update success count
        metrics.successes += 1

        # Transition from initiated to warming_up on first success
        if metrics.state == WarmupState.INITIATED:
            metrics.state = WarmupState.WARMING_UP

        # Check if warm-up should complete
        should_complete = self._should_complete_warmup(account_index)

        if should_complete:
            metrics.state = WarmupState.COMPLETE
            metrics.current_factor = config.max_factor
        else:
            # Update warm-up factor based on profile
            if config.profile == WarmupProfile.CONSERVATIVE:
                metrics.current_factor = min(config.max_factor, metrics.current_factor + config.increment * 0.5)
            elif config.profile == WarmupProfile.AGGRESSIVE:
                metrics.current_factor = min(config.max_factor, metrics.current_factor + config.increment * 1.5)
            elif config.profile == WarmupProfile.MANUAL:
                # Don't auto-increment, keep current factor
                pass
            else:  # ADAPTIVE (default)
                # Adjust increment based on performance
                adjustment = self._calculate_adaptive_adjustment(account_index, latency)
                metrics.current_factor = min(config.max_factor, metrics.current_factor + config.increment * adjustment)

        return metrics.current_factor

    def update_warmup_on_failure(self, account_index: int, error: str) -> float:
        """Update warm-up state on failed request.

        Args:
            account_index: Account index
            error: Error description

        Returns:
            Updated warm-up factor (0.0-1.0)
        """
        if account_index not in self._configs or account_index not in self._metrics:
            return 1.0  # No warm-up configured, full capacity

        metrics = self._metrics[account_index]

        # Record error
        metrics.error_history.append((time.time(), error))

        # Update failure count
        metrics.failures += 1

        # Reduce warm-up factor on failure (conservative approach)
        config = self._configs[account_index]
        reduction_factor = 0.5  # Reduce by 50% on failure

        if metrics.state in [WarmupState.INITIATED, WarmupState.WARMING_UP]:
            metrics.current_factor = max(config.initial_factor, metrics.current_factor * reduction_factor)

        return metrics.current_factor

    def _should_complete_warmup(self, account_index: int) -> bool:
        """Determine if warm-up should be completed.

        Args:
            account_index: Account index

        Returns:
            True if warm-up should complete
        """
        config = self._configs[account_index]
        metrics = self._metrics[account_index]

        # Check success count
        if metrics.successes >= config.min_successes:
            return True

        # Check time limit
        if config.max_duration > 0:
            elapsed = time.time() - metrics.start_time
            if elapsed >= config.max_duration:
                return True

        # Check if already at max factor
        if metrics.current_factor >= config.max_factor:
            return True

        return False

    def _calculate_adaptive_adjustment(self, account_index: int, current_latency: float) -> float:
        """Calculate adaptive adjustment factor based on performance.

        Args:
            account_index: Account index
            current_latency: Current request latency

        Returns:
            Adjustment multiplier (0.5-2.0)
        """
        metrics = self._metrics[account_index]

        # Establish baseline if not exists
        if account_index not in self._performance_baselines:
            self._performance_baselines[account_index] = current_latency
            return 1.0  # Neutral adjustment for first request

        baseline_latency = self._performance_baselines[account_index]

        # Compare current performance to baseline
        if current_latency <= baseline_latency * 0.8:
            # Significantly better performance, increase ramp-up speed
            return 1.5
        elif current_latency <= baseline_latency * 1.2:
            # Similar performance, normal ramp-up
            return 1.0
        else:
            # Worse performance, slow down ramp-up
            return 0.7

    def get_warmup_status(self, account_index: int) -> Dict[str, Any]:
        """Get detailed warm-up status for an account.

        Args:
            account_index: Account index

        Returns:
            Warm-up status dictionary
        """
        if account_index not in self._configs or account_index not in self._metrics:
            return {
                "account_index": account_index,
                "configured": False,
                "state": WarmupState.INACTIVE.value,
                "factor": 1.0,
                "progress": 1.0
            }

        config = self._configs[account_index]
        metrics = self._metrics[account_index]

        progress = 0.0
        if config.profile != WarmupProfile.MANUAL:
            if config.min_successes > 0:
                progress = min(1.0, metrics.successes / config.min_successes)
            elif config.max_duration > 0:
                elapsed = time.time() - metrics.start_time
                progress = min(1.0, elapsed / config.max_duration)

        return {
            "account_index": account_index,
            "configured": True,
            "state": metrics.state.value,
            "factor": metrics.current_factor,
            "progress": progress,
            "profile": config.profile.value,
            "successes": metrics.successes,
            "failures": metrics.failures,
            "elapsed_time": time.time() - metrics.start_time if metrics.start_time > 0 else 0,
            "estimated_completion": self._estimate_completion_time(account_index)
        }

    def _estimate_completion_time(self, account_index: int) -> Optional[float]:
        """Estimate time until warm-up completion.

        Args:
            account_index: Account index

        Returns:
            Estimated seconds until completion, or None if indeterminate
        """
        if account_index not in self._configs or account_index not in self._metrics:
            return None

        config = self._configs[account_index]
        metrics = self._metrics[account_index]

        if metrics.state in [WarmupState.INACTIVE, WarmupState.COMPLETE]:
            return 0.0

        # If using success count method
        if config.min_successes > 0:
            remaining = max(0, config.min_successes - metrics.successes)
            if metrics.successes > 0:
                avg_time_per_success = (time.time() - metrics.start_time) / metrics.successes
                return remaining * avg_time_per_success
            else:
                return None  # Can't estimate without any successes

        # If using time-based method
        if config.max_duration > 0:
            elapsed = time.time() - metrics.start_time
            return max(0, config.max_duration - elapsed)

        return None

    def set_manual_warmup_factor(self, account_index: int, factor: float) -> None:
        """Set manual warm-up factor for an account.

        Args:
            account_index: Account index
            factor: Warm-up factor (0.0-1.0)
        """
        if account_index not in self._configs:
            self._configs[account_index] = WarmupConfig(profile=WarmupProfile.MANUAL)

        if account_index not in self._metrics:
            self._metrics[account_index] = WarmupMetrics()

        metrics = self._metrics[account_index]
        metrics.current_factor = max(0.0, min(1.0, factor))

        # Update state based on factor
        if factor >= 1.0:
            metrics.state = WarmupState.COMPLETE
        elif factor > 0.0:
            metrics.state = WarmupState.WARMING_UP
        else:
            metrics.state = WarmupState.INACTIVE

    def reset_warmup(self, account_index: int) -> None:
        """Reset warm-up state for an account.

        Args:
            account_index: Account index
        """
        if account_index in self._metrics:
            metrics = self._metrics[account_index]
            metrics.state = WarmupState.INACTIVE
            metrics.current_factor = 1.0
            metrics.successes = 0
            metrics.failures = 0
            metrics.start_time = 0.0

    def get_pool_warmup_summary(self, account_indices: list) -> Dict[str, Any]:
        """Get warm-up summary for a pool of accounts.

        Args:
            account_indices: List of account indices

        Returns:
            Warm-up summary dictionary
        """
        summary = {
            "total_accounts": len(account_indices),
            "warming_up": 0,
            "complete": 0,
            "initiated": 0,
            "inactive": 0,
            "average_factor": 0.0,
            "accounts_needing_attention": []
        }

        factors = []
        for idx in account_indices:
            if idx in self._metrics:
                metrics = self._metrics[idx]
                factors.append(metrics.current_factor)

                if metrics.state == WarmupState.WARMING_UP:
                    summary["warming_up"] += 1
                elif metrics.state == WarmupState.COMPLETE:
                    summary["complete"] += 1
                elif metrics.state == WarmupState.INITIATED:
                    summary["initiated"] += 1
                else:
                    summary["inactive"] += 1

                # Flag accounts needing attention (long warm-up, many failures)
                if metrics.failures > 3 or (time.time() - metrics.start_time) > 300:
                    status = self.get_warmup_status(idx)
                    summary["accounts_needing_attention"].append({
                        "account_index": idx,
                        "status": status
                    })

        if factors:
            summary["average_factor"] = sum(factors) / len(factors)

        return summary