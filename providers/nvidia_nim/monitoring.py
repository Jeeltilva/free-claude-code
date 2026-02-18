"""Advanced health monitoring for NVIDIA NIM account pool."""

import time
from typing import List, Dict, Any
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from .account import NimApiAccount


@dataclass
class HealthMetrics:
    """Historical health metrics for an account."""

    # Failure tracking over time
    failure_history: deque = field(default_factory=lambda: deque(maxlen=100))

    # Latency tracking over time
    latency_history: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Request success/failure tracking
    request_outcomes: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Timestamped metrics for trend analysis
    health_trends: deque = field(default_factory=lambda: deque(maxlen=100))


class HealthMonitor:
    """Monitors and analyzes account health metrics."""

    def __init__(self, alert_threshold_failure_rate: float = 0.1, alert_threshold_latency_spike: float = 2.0):
        """Initialize health monitor.

        Args:
            alert_threshold_failure_rate: Failure rate threshold for alerts (0.1 = 10%)
            alert_threshold_latency_spike: Latency multiplier for spike detection
        """
        self.alert_threshold_failure_rate = alert_threshold_failure_rate
        self.alert_threshold_latency_spike = alert_threshold_latency_spike
        self._metrics: Dict[int, HealthMetrics] = {}

    def _get_metrics(self, account_index: int) -> HealthMetrics:
        """Get or create metrics for an account."""
        if account_index not in self._metrics:
            self._metrics[account_index] = HealthMetrics()
        return self._metrics[account_index]

    def record_request(self, account: NimApiAccount, success: bool, latency: float) -> None:
        """Record a request outcome for health analysis.

        Args:
            account: The account that made the request
            success: Whether the request succeeded
            latency: Request latency in seconds
        """
        metrics = self._get_metrics(account.index)
        timestamp = time.time()

        # Record request outcome
        metrics.request_outcomes.append((timestamp, success))

        # Record latency
        metrics.latency_history.append((timestamp, latency))

        # Record failure details if failed
        if not success:
            metrics.failure_history.append((timestamp, latency))

    def record_failure(self, account: NimApiAccount, error_type: str, latency: float) -> None:
        """Record a failed request for health analysis.

        Args:
            account: The account that failed
            error_type: Type of error that occurred
            latency: Request latency before failure
        """
        metrics = self._get_metrics(account.index)
        timestamp = time.time()

        metrics.request_outcomes.append((timestamp, False))
        metrics.failure_history.append((timestamp, error_type, latency))

    def get_failure_rate(self, account: NimApiAccount, window_seconds: int = 300) -> float:
        """Calculate failure rate over a time window.

        Args:
            account: Account to analyze
            window_seconds: Time window in seconds

        Returns:
            Failure rate as ratio (0.0 to 1.0)
        """
        metrics = self._get_metrics(account.index)
        cutoff_time = time.time() - window_seconds

        recent_requests = [
            outcome for outcome in metrics.request_outcomes
            if outcome[0] >= cutoff_time
        ]

        if not recent_requests:
            return 0.0

        failed_requests = sum(1 for _, success in recent_requests if not success)
        return failed_requests / len(recent_requests)

    def get_avg_latency(self, account: NimApiAccount, window_seconds: int = 300) -> float:
        """Calculate average latency over a time window.

        Args:
            account: Account to analyze
            window_seconds: Time window in seconds

        Returns:
            Average latency in seconds
        """
        metrics = self._get_metrics(account.index)
        cutoff_time = time.time() - window_seconds

        recent_latencies = [
            latency for timestamp, latency in metrics.latency_history
            if timestamp >= cutoff_time
        ]

        if not recent_latencies:
            return account.avg_latency  # Fall back to account's EMA

        return sum(recent_latencies) / len(recent_latencies)

    def detect_latency_spike(self, account: NimApiAccount, window_seconds: int = 300) -> bool:
        """Detect if current latency is a significant spike.

        Args:
            account: Account to analyze
            window_seconds: Time window for baseline calculation

        Returns:
            True if current latency is significantly higher than baseline
        """
        metrics = self._get_metrics(account.index)
        if not metrics.latency_history:
            return False

        current_latency = metrics.latency_history[-1][1]
        baseline_latency = self.get_avg_latency(account, window_seconds)

        return current_latency > baseline_latency * self.alert_threshold_latency_spike

    def get_health_alerts(self, account: NimApiAccount) -> List[Dict[str, Any]]:
        """Get health alerts for an account.

        Args:
            account: Account to check

        Returns:
            List of alert dictionaries
        """
        alerts = []
        failure_rate = self.get_failure_rate(account)

        # Check for high failure rate
        if failure_rate > self.alert_threshold_failure_rate:
            alerts.append({
                "type": "high_failure_rate",
                "severity": "warning" if failure_rate < 0.2 else "critical",
                "message": f"Failure rate {failure_rate:.1%} exceeds threshold {self.alert_threshold_failure_rate:.1%}",
                "failure_rate": failure_rate,
                "account_index": account.index,
                "timestamp": time.time()
            })

        # Check for latency spikes
        if self.detect_latency_spike(account):
            alerts.append({
                "type": "latency_spike",
                "severity": "warning",
                "message": f"Latency spike detected",
                "account_index": account.index,
                "timestamp": time.time()
            })

        # Check for circuit breaker state
        if account.circuit_breaker_state.name != "CLOSED":
            alerts.append({
                "type": "circuit_breaker_open",
                "severity": "info" if account.circuit_breaker_state.name == "HALF_OPEN" else "warning",
                "message": f"Circuit breaker state: {account.circuit_breaker_state.name}",
                "account_index": account.index,
                "state": account.circuit_breaker_state.name,
                "timestamp": time.time()
            })

        return alerts

    def get_detailed_health_report(self, account: NimApiAccount) -> Dict[str, Any]:
        """Get detailed health report for an account.

        Args:
            account: Account to analyze

        Returns:
            Dictionary with detailed health metrics
        """
        metrics = self._get_metrics(account.index)

        return {
            "account_index": account.index,
            "current_state": account.circuit_breaker_state.name,
            "failure_rate_5min": self.get_failure_rate(account, 300),
            "failure_rate_1hour": self.get_failure_rate(account, 3600),
            "avg_latency_5min": self.get_avg_latency(account, 300),
            "avg_latency_1hour": self.get_avg_latency(account, 3600),
            "total_requests_tracked": len(metrics.request_outcomes),
            "total_failures_tracked": len(metrics.failure_history),
            "recent_alerts": self.get_health_alerts(account),
            "trend_analysis": self._analyze_trends(account),
        }

    def _analyze_trends(self, account: NimApiAccount) -> Dict[str, Any]:
        """Analyze health trends for predictive insights.

        Args:
            account: Account to analyze

        Returns:
            Trend analysis results
        """
        # Simplified trend analysis - in a real implementation this would be more sophisticated
        failure_rate = self.get_failure_rate(account, 3600)  # 1 hour

        trend = "stable"
        if failure_rate > 0.15:
            trend = "degrading"
        elif failure_rate < 0.02:
            trend = "improving"

        return {
            "failure_rate_trend": trend,
            "predicted_availability": max(0.0, min(1.0, 1.0 - failure_rate)),
            "recommendation": self._get_recommendation(account, failure_rate)
        }

    def _get_recommendation(self, account: NimApiAccount, failure_rate: float) -> str:
        """Get recommendation based on health analysis.

        Args:
            account: Account to analyze
            failure_rate: Current failure rate

        Returns:
            Recommendation string
        """
        if account.circuit_breaker_state.name == "OPEN":
            return "Account is blocked due to failures. Wait for cooldown or manually reset."
        elif failure_rate > 0.2:
            return "High failure rate detected. Consider reducing load or investigating connectivity."
        elif failure_rate > 0.05:
            return "Moderate failure rate. Monitor closely."
        else:
            return "Account health is good."

    def get_pool_health_summary(self, accounts: List[NimApiAccount]) -> Dict[str, Any]:
        """Get health summary for entire account pool.

        Args:
            accounts: List of all accounts

        Returns:
            Pool health summary
        """
        total_accounts = len(accounts)
        healthy_accounts = sum(1 for acc in accounts if acc.is_healthy())
        available_accounts = sum(1 for acc in accounts if acc.is_available())

        # Aggregate alerts
        all_alerts = []
        for account in accounts:
            all_alerts.extend(self.get_health_alerts(account))

        critical_alerts = [alert for alert in all_alerts if alert["severity"] == "critical"]
        warning_alerts = [alert for alert in all_alerts if alert["severity"] == "warning"]

        return {
            "total_accounts": total_accounts,
            "healthy_accounts": healthy_accounts,
            "available_accounts": available_accounts,
            "unhealthy_accounts": total_accounts - healthy_accounts,
            "critical_alerts": len(critical_alerts),
            "warning_alerts": len(warning_alerts),
            "overall_health": "critical" if critical_alerts else "warning" if warning_alerts else "healthy",
            "timestamp": time.time()
        }