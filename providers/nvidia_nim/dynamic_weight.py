"""Dynamic weight adjustment for NVIDIA NIM account pool."""

import time
from typing import List, Dict
from dataclasses import dataclass
from enum import Enum


class WeightAdjustmentStrategy(Enum):
    """Strategies for dynamic weight adjustment."""

    LATENCY_BASED = "latency_based"        # Adjust based on response times
    SUCCESS_RATE = "success_rate"         # Adjust based on success/failure rates
    HYBRID = "hybrid"                     # Combination of latency and success rate
    PERFORMANCE_TREND = "performance_trend"  # Based on performance trends over time


@dataclass
class PerformanceMetrics:
    """Performance metrics for weight calculation."""

    avg_latency: float = 0.0
    success_rate: float = 1.0
    request_count: int = 0
    failure_count: int = 0
    latency_samples: int = 0

    # Trend analysis
    latency_trend: float = 0.0  # Positive = increasing, negative = decreasing
    success_trend: float = 0.0  # Positive = improving, negative = degrading


class DynamicWeightAdjuster:
    """Adjusts account weights dynamically based on performance metrics."""

    def __init__(
        self,
        strategy: WeightAdjustmentStrategy = WeightAdjustmentStrategy.HYBRID,
        min_weight: float = 0.1,
        max_weight: float = 2.0,
        adjustment_sensitivity: float = 1.0,
    ):
        """Initialize dynamic weight adjuster.

        Args:
            strategy: Weight adjustment strategy to use
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
            adjustment_sensitivity: Sensitivity multiplier for adjustments (higher = more aggressive)
        """
        self.strategy = strategy
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.adjustment_sensitivity = adjustment_sensitivity
        self._baseline_metrics: Dict[int, PerformanceMetrics] = {}

    def calculate_dynamic_weight(
        self,
        account_index: int,
        current_weight: float,
        metrics: PerformanceMetrics,
    ) -> float:
        """Calculate dynamic weight based on performance metrics.

        Args:
            account_index: Account index
            current_weight: Current weight
            metrics: Current performance metrics

        Returns:
            Adjusted weight
        """
        # Establish baseline metrics on first call
        if account_index not in self._baseline_metrics:
            self._baseline_metrics[account_index] = metrics
            return current_weight

        baseline = self._baseline_metrics[account_index]

        # Apply strategy-specific adjustment
        if self.strategy == WeightAdjustmentStrategy.LATENCY_BASED:
            return self._adjust_by_latency(current_weight, metrics, baseline)
        elif self.strategy == WeightAdjustmentStrategy.SUCCESS_RATE:
            return self._adjust_by_success_rate(current_weight, metrics, baseline)
        elif self.strategy == WeightAdjustmentStrategy.PERFORMANCE_TREND:
            return self._adjust_by_trends(current_weight, metrics, baseline)
        else:  # HYBRID (default)
            return self._adjust_hybrid(current_weight, metrics, baseline)

    def _adjust_by_latency(
        self,
        current_weight: float,
        metrics: PerformanceMetrics,
        baseline: PerformanceMetrics,
    ) -> float:
        """Adjust weight based on latency performance."""
        if metrics.latency_samples == 0 or baseline.avg_latency == 0:
            return current_weight

        # Calculate latency ratio (current vs baseline)
        latency_ratio = metrics.avg_latency / baseline.avg_latency

        # Invert ratio for weight adjustment (lower latency = higher weight)
        # Apply sensitivity and bounds
        adjustment_factor = max(0.5, min(2.0, 1.0 / latency_ratio))
        adjustment_factor = 1.0 + (adjustment_factor - 1.0) * self.adjustment_sensitivity

        new_weight = current_weight * adjustment_factor
        return max(self.min_weight, min(self.max_weight, new_weight))

    def _adjust_by_success_rate(
        self,
        current_weight: float,
        metrics: PerformanceMetrics,
        baseline: PerformanceMetrics,
    ) -> float:
        """Adjust weight based on success rate."""
        # Avoid division by zero
        if baseline.success_rate == 0:
            return current_weight

        # Calculate success rate ratio
        success_ratio = metrics.success_rate / baseline.success_rate

        # Apply sensitivity and bounds
        adjustment_factor = max(0.5, min(2.0, success_ratio))
        adjustment_factor = 1.0 + (adjustment_factor - 1.0) * self.adjustment_sensitivity

        new_weight = current_weight * adjustment_factor
        return max(self.min_weight, min(self.max_weight, new_weight))

    def _adjust_by_trends(
        self,
        current_weight: float,
        metrics: PerformanceMetrics,
        baseline: PerformanceMetrics,
    ) -> float:
        """Adjust weight based on performance trends."""
        # Combine latency and success trends
        combined_trend = (metrics.latency_trend * -1.0) + metrics.success_trend

        # Convert trend to adjustment factor
        # Positive trend = improving performance = increase weight
        # Negative trend = degrading performance = decrease weight
        adjustment_factor = 1.0 + (combined_trend * 0.1 * self.adjustment_sensitivity)
        adjustment_factor = max(0.5, min(2.0, adjustment_factor))

        new_weight = current_weight * adjustment_factor
        return max(self.min_weight, min(self.max_weight, new_weight))

    def _adjust_hybrid(
        self,
        current_weight: float,
        metrics: PerformanceMetrics,
        baseline: PerformanceMetrics,
    ) -> float:
        """Hybrid adjustment combining latency and success rate."""
        latency_weight = self._adjust_by_latency(current_weight, metrics, baseline)
        success_weight = self._adjust_by_success_rate(current_weight, metrics, baseline)

        # Average the two adjustments
        hybrid_weight = (latency_weight + success_weight) / 2.0
        return max(self.min_weight, min(self.max_weight, hybrid_weight))

    def update_baseline_metrics(self, account_index: int, metrics: PerformanceMetrics) -> None:
        """Update baseline metrics for an account.

        Args:
            account_index: Account index
            metrics: New baseline metrics
        """
        self._baseline_metrics[account_index] = metrics

    def get_weight_adjustment_report(
        self,
        account_index: int,
        current_weight: float,
        metrics: PerformanceMetrics,
    ) -> Dict[str, any]:
        """Get detailed report of weight adjustment calculation.

        Args:
            account_index: Account index
            current_weight: Current weight
            metrics: Current performance metrics

        Returns:
            Weight adjustment report
        """
        if account_index not in self._baseline_metrics:
            return {
                "account_index": account_index,
                "current_weight": current_weight,
                "adjusted_weight": current_weight,
                "adjustment_made": False,
                "reason": "No baseline metrics available",
            }

        baseline = self._baseline_metrics[account_index]
        adjusted_weight = self.calculate_dynamic_weight(account_index, current_weight, metrics)

        # Calculate adjustment factors for each component
        latency_factor = 1.0
        success_factor = 1.0
        trend_factor = 1.0

        if baseline.avg_latency > 0 and metrics.latency_samples > 0:
            latency_ratio = metrics.avg_latency / baseline.avg_latency
            latency_factor = max(0.5, min(2.0, 1.0 / latency_ratio))

        if baseline.success_rate > 0:
            success_ratio = metrics.success_rate / baseline.success_rate
            success_factor = max(0.5, min(2.0, success_ratio))

        if self.strategy == WeightAdjustmentStrategy.PERFORMANCE_TREND:
            combined_trend = (metrics.latency_trend * -1.0) + metrics.success_trend
            trend_factor = max(0.5, min(2.0, 1.0 + (combined_trend * 0.1)))

        return {
            "account_index": account_index,
            "strategy": self.strategy.value,
            "current_weight": current_weight,
            "adjusted_weight": adjusted_weight,
            "adjustment_made": abs(adjusted_weight - current_weight) > 0.01,
            "factors": {
                "latency_factor": latency_factor,
                "success_factor": success_factor,
                "trend_factor": trend_factor,
            },
            "metrics": {
                "current_avg_latency": metrics.avg_latency,
                "baseline_avg_latency": baseline.avg_latency,
                "current_success_rate": metrics.success_rate,
                "baseline_success_rate": baseline.success_rate,
                "latency_trend": metrics.latency_trend,
                "success_trend": metrics.success_trend,
            },
            "bounds": {
                "min_weight": self.min_weight,
                "max_weight": self.max_weight,
            }
        }

    def reset_baseline(self, account_index: int = None) -> None:
        """Reset baseline metrics.

        Args:
            account_index: Specific account to reset, or None for all
        """
        if account_index is not None:
            if account_index in self._baseline_metrics:
                del self._baseline_metrics[account_index]
        else:
            self._baseline_metrics.clear()