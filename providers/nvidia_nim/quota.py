"""Advanced quota management for NVIDIA NIM accounts."""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime, timedelta


@dataclass
class QuotaLimits:
    """Tiered quota limits for an account."""

    hourly_limit: int = 0  # 0 = unlimited
    daily_limit: int = 0   # 0 = unlimited
    monthly_limit: int = 0 # 0 = unlimited


@dataclass
class QuotaUsage:
    """Usage tracking for quota limits."""

    hourly_count: int = 0
    daily_count: int = 0
    monthly_count: int = 0

    # Reset timestamps
    hourly_reset_time: float = 0.0
    daily_reset_time: float = 0.0
    monthly_reset_time: float = 0.0

    # Usage history for trend analysis
    hourly_usage_history: list = field(default_factory=list)  # [(timestamp, count), ...]
    daily_usage_history: list = field(default_factory=list)   # [(timestamp, count), ...]


class QuotaManager:
    """Manages tiered quota limits and rollover policies."""

    def __init__(self, alert_threshold: float = 0.8):
        """Initialize quota manager.

        Args:
            alert_threshold: Percentage threshold for quota exhaustion warnings (0.8 = 80%)
        """
        self.alert_threshold = alert_threshold
        self._quotas: Dict[int, QuotaLimits] = {}
        self._usage: Dict[int, QuotaUsage] = {}

    def set_account_quotas(self, account_index: int, limits: QuotaLimits) -> None:
        """Set quota limits for an account.

        Args:
            account_index: Account index
            limits: Quota limits to set
        """
        self._quotas[account_index] = limits
        if account_index not in self._usage:
            self._usage[account_index] = QuotaUsage()
        self._reset_usage_if_needed(account_index)

    def get_account_limits(self, account_index: int) -> Optional[QuotaLimits]:
        """Get quota limits for an account.

        Args:
            account_index: Account index

        Returns:
            Quota limits or None if not set
        """
        return self._quotas.get(account_index)

    def _reset_usage_if_needed(self, account_index: int) -> None:
        """Reset usage counters if reset time has passed.

        Args:
            account_index: Account index
        """
        usage = self._usage[account_index]
        now = time.time()

        # Reset hourly counter
        if usage.hourly_reset_time <= now:
            usage.hourly_usage_history.append((now, usage.hourly_count))
            usage.hourly_count = 0
            usage.hourly_reset_time = self._get_next_hour_reset()

        # Reset daily counter
        if usage.daily_reset_time <= now:
            usage.daily_usage_history.append((now, usage.daily_count))
            usage.daily_count = 0
            usage.daily_reset_time = self._get_next_day_reset()

        # Reset monthly counter
        if usage.monthly_reset_time <= now:
            usage.monthly_count = 0
            usage.monthly_reset_time = self._get_next_month_reset()

    def _get_next_hour_reset(self) -> float:
        """Get timestamp for next hourly reset."""
        now = datetime.now()
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return next_hour.timestamp()

    def _get_next_day_reset(self) -> float:
        """Get timestamp for next daily reset."""
        now = datetime.now()
        next_day = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return next_day.timestamp()

    def _get_next_month_reset(self) -> float:
        """Get timestamp for next monthly reset."""
        now = datetime.now()
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            next_month = now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
        return next_month.timestamp()

    def increment_usage(self, account_index: int) -> bool:
        """Increment usage counters for an account.

        Args:
            account_index: Account index

        Returns:
            True if increment successful, False if quota exceeded
        """
        if account_index not in self._quotas:
            return True  # No quotas set, unlimited

        self._reset_usage_if_needed(account_index)
        usage = self._usage[account_index]
        limits = self._quotas[account_index]

        # Check hourly limit
        if limits.hourly_limit > 0 and usage.hourly_count >= limits.hourly_limit:
            return False

        # Check daily limit
        if limits.daily_limit > 0 and usage.daily_count >= limits.daily_limit:
            return False

        # Check monthly limit
        if limits.monthly_limit > 0 and usage.monthly_count >= limits.monthly_limit:
            return False

        # Increment all counters
        usage.hourly_count += 1
        usage.daily_count += 1
        usage.monthly_count += 1

        return True

    def check_quota_availability(self, account_index: int) -> bool:
        """Check if account has available quota.

        Args:
            account_index: Account index

        Returns:
            True if quota available, False if exceeded
        """
        if account_index not in self._quotas:
            return True  # No quotas set, unlimited

        self._reset_usage_if_needed(account_index)
        usage = self._usage[account_index]
        limits = self._quotas[account_index]

        # Check hourly limit
        if limits.hourly_limit > 0 and usage.hourly_count >= limits.hourly_limit:
            return False

        # Check daily limit
        if limits.daily_limit > 0 and usage.daily_count >= limits.daily_limit:
            return False

        # Check monthly limit
        if limits.monthly_limit > 0 and usage.monthly_count >= limits.monthly_limit:
            return False

        return True

    def get_quota_utilization(self, account_index: int) -> Dict[str, float]:
        """Get quota utilization percentages for an account.

        Args:
            account_index: Account index

        Returns:
            Dict with utilization percentages for each tier
        """
        if account_index not in self._quotas:
            return {
                "hourly_utilization": 0.0,
                "daily_utilization": 0.0,
                "monthly_utilization": 0.0
            }

        self._reset_usage_if_needed(account_index)
        usage = self._usage[account_index]
        limits = self._quotas[account_index]

        hourly_util = 0.0
        daily_util = 0.0
        monthly_util = 0.0

        if limits.hourly_limit > 0:
            hourly_util = min(1.0, usage.hourly_count / limits.hourly_limit)

        if limits.daily_limit > 0:
            daily_util = min(1.0, usage.daily_count / limits.daily_limit)

        if limits.monthly_limit > 0:
            monthly_util = min(1.0, usage.monthly_count / limits.monthly_limit)

        return {
            "hourly_utilization": hourly_util,
            "daily_utilization": daily_util,
            "monthly_utilization": monthly_util
        }

    def get_quota_alerts(self, account_index: int) -> list:
        """Get quota exhaustion warnings for an account.

        Args:
            account_index: Account index

        Returns:
            List of alert dictionaries
        """
        if account_index not in self._quotas:
            return []

        utilization = self.get_quota_utilization(account_index)
        alerts = []

        # Check for high utilization alerts
        if utilization["hourly_utilization"] >= self.alert_threshold:
            alerts.append({
                "type": "quota_warning",
                "scope": "hourly",
                "severity": "warning",
                "utilization": utilization["hourly_utilization"],
                "threshold": self.alert_threshold,
                "message": f"Hourly quota {utilization['hourly_utilization']:.1%} exceeds warning threshold {self.alert_threshold:.1%}"
            })

        if utilization["daily_utilization"] >= self.alert_threshold:
            alerts.append({
                "type": "quota_warning",
                "scope": "daily",
                "severity": "warning",
                "utilization": utilization["daily_utilization"],
                "threshold": self.alert_threshold,
                "message": f"Daily quota {utilization['daily_utilization']:.1%} exceeds warning threshold {self.alert_threshold:.1%}"
            })

        if utilization["monthly_utilization"] >= self.alert_threshold:
            alerts.append({
                "type": "quota_warning",
                "scope": "monthly",
                "severity": "warning",
                "utilization": utilization["monthly_utilization"],
                "threshold": self.alert_threshold,
                "message": f"Monthly quota {utilization['monthly_utilization']:.1%} exceeds warning threshold {self.alert_threshold:.1%}"
            })

        return alerts

    def get_quota_predictions(self, account_index: int, hours_ahead: int = 24) -> Dict[str, any]:
        """Predict quota exhaustion based on recent usage trends.

        Args:
            account_index: Account index
            hours_ahead: Hours to predict ahead

        Returns:
            Prediction results
        """
        if account_index not in self._quotas or account_index not in self._usage:
            return {
                "predictions": {},
                "trend_analysis": "insufficient_data"
            }

        usage = self._usage[account_index]
        limits = self._quotas[account_index]

        predictions = {}

        # Calculate recent hourly rate (requests per hour)
        recent_hours = []
        now = time.time()
        for timestamp, count in reversed(usage.hourly_usage_history[-24:]):  # Last 24 hours
            if now - timestamp <= 24 * 3600:  # Within last 24 hours
                recent_hours.append(count)

        hourly_rate = sum(recent_hours) / max(1, len(recent_hours)) if recent_hours else 0

        # Predict hourly exhaustion
        if limits.hourly_limit > 0:
            remaining_hourly = max(0, limits.hourly_limit - usage.hourly_count)
            hours_until_hourly_exhaustion = remaining_hourly / max(1, hourly_rate) if hourly_rate > 0 else float('inf')
            predictions["hourly"] = {
                "rate": hourly_rate,
                "remaining": remaining_hourly,
                "hours_until_exhaustion": hours_until_hourly_exhaustion,
                "will_exhaust_soon": hours_until_hourly_exhaustion <= hours_ahead
            }

        # Calculate recent daily rate
        recent_days = []
        for timestamp, count in reversed(usage.daily_usage_history[-7:]):  # Last 7 days
            if now - timestamp <= 7 * 24 * 3600:  # Within last 7 days
                recent_days.append(count)

        daily_rate = sum(recent_days) / max(1, len(recent_days)) if recent_days else 0

        # Predict daily exhaustion
        if limits.daily_limit > 0:
            remaining_daily = max(0, limits.daily_limit - usage.daily_count)
            days_until_daily_exhaustion = remaining_daily / max(1, daily_rate) if daily_rate > 0 else float('inf')
            predictions["daily"] = {
                "rate": daily_rate,
                "remaining": remaining_daily,
                "days_until_exhaustion": days_until_daily_exhaustion,
                "will_exhaust_soon": days_until_daily_exhaustion <= hours_ahead / 24
            }

        return {
            "predictions": predictions,
            "trend_analysis": "increasing" if hourly_rate > 0 and daily_rate > 0 else "stable"
        }

    def get_comprehensive_quota_report(self, account_index: int) -> Dict[str, any]:
        """Get comprehensive quota report for an account.

        Args:
            account_index: Account index

        Returns:
            Comprehensive quota report
        """
        if account_index not in self._quotas:
            return {
                "account_index": account_index,
                "has_quotas": False,
                "message": "No quotas configured for this account"
            }

        limits = self._quotas[account_index]
        utilization = self.get_quota_utilization(account_index)
        alerts = self.get_quota_alerts(account_index)
        predictions = self.get_quota_predictions(account_index)

        return {
            "account_index": account_index,
            "has_quotas": True,
            "limits": {
                "hourly": limits.hourly_limit,
                "daily": limits.daily_limit,
                "monthly": limits.monthly_limit
            },
            "current_usage": {
                "hourly": self._usage[account_index].hourly_count,
                "daily": self._usage[account_index].daily_count,
                "monthly": self._usage[account_index].monthly_count
            },
            "utilization": utilization,
            "alerts": alerts,
            "predictions": predictions,
            "next_resets": {
                "hourly": self._usage[account_index].hourly_reset_time,
                "daily": self._usage[account_index].daily_reset_time,
                "monthly": self._usage[account_index].monthly_reset_time
            }
        }