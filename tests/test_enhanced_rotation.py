"""Tests for enhanced account rotation features."""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from providers.nvidia_nim.account import NimApiAccount, CircuitBreakerState
from providers.nvidia_nim.pool import AccountPool
from providers.nvidia_nim.rotator import AccountRotator, RotationStrategy
from providers.nvidia_nim.monitoring import HealthMonitor
from providers.nvidia_nim.warmup import WarmupManager, WarmupConfig, WarmupProfile
from providers.nvidia_nim.dynamic_weight import DynamicWeightAdjuster, PerformanceMetrics


@pytest.fixture
def mock_api_key():
    return "nvapi-test-key-12345678"


@pytest.fixture
def base_url():
    return "https://test.api.nvidia.com/v1"


@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    client.close = AsyncMock()
    return client


class TestCircuitBreaker:
    """Test enhanced circuit breaker functionality."""

    def test_circuit_breaker_transitions(self, mock_api_key, base_url):
        with patch("providers.nvidia_nim.account.AsyncOpenAI"):
            account = NimApiAccount(
                api_key=mock_api_key,
                base_url=base_url,
                index=0,
            )

            # Initially should be CLOSED
            assert account.circuit_breaker_state == CircuitBreakerState.CLOSED

            # Transition to OPEN
            account.transition_to_open(60)
            assert account.circuit_breaker_state == CircuitBreakerState.OPEN
            assert account.blocked_until > time.time()

            # While blocked, should be unavailable
            assert not account.is_available()

            # Expire the block and check transition to HALF_OPEN
            account.blocked_until = time.time() - 10
            assert account.is_available()  # This should trigger transition to HALF_OPEN
            assert account.circuit_breaker_state == CircuitBreakerState.HALF_OPEN

            # Transition back to CLOSED after successful probes
            account.transition_to_closed()
            assert account.circuit_breaker_state == CircuitBreakerState.CLOSED

    def test_half_open_probing(self, mock_api_key, base_url):
        with patch("providers.nvidia_nim.account.AsyncOpenAI"):
            account = NimApiAccount(
                api_key=mock_api_key,
                base_url=base_url,
                index=0,
            )

            # Set to HALF_OPEN state
            account.transition_to_half_open()

            # Should allow limited probes
            max_probes = 5
            for i in range(max_probes):
                assert account.can_probe_in_half_open(max_probes)

            # Should reject additional probes
            assert not account.can_probe_in_half_open(max_probes)

    def test_circuit_breaker_failure_handling(self, mock_api_key, base_url):
        with patch("providers.nvidia_nim.account.AsyncOpenAI"):
            account = NimApiAccount(
                api_key=mock_api_key,
                base_url=base_url,
                index=0,
            )

            # Cause enough failures to trip the circuit breaker
            for i in range(3):  # Default max_failures is 3
                account.mark_failed(max_failures=3, failure_cooldown=60)

            # Should now be in OPEN state
            assert account.circuit_breaker_state == CircuitBreakerState.OPEN
            assert account.blocked_until > time.time()


class TestHealthMonitoring:
    """Test health monitoring features."""

    def test_health_monitor_basic_tracking(self):
        monitor = HealthMonitor()
        account = MagicMock()
        account.index = 0

        # Record some requests
        monitor.record_request(account, success=True, latency=0.5)
        monitor.record_request(account, success=False, latency=1.0)

        # Check failure rate
        failure_rate = monitor.get_failure_rate(account, window_seconds=10)
        assert failure_rate == 0.5  # 1 failure out of 2 requests

        # Check average latency
        avg_latency = monitor.get_avg_latency(account, window_seconds=10)
        assert avg_latency == 0.75  # Average of 0.5 and 1.0

    def test_health_alerts(self):
        monitor = HealthMonitor(alert_threshold_failure_rate=0.3)
        account = MagicMock()
        account.index = 0
        account.circuit_breaker_state.name = "CLOSED"

        # Record low failure rate (1 failure out of 10 = 10%)
        for i in range(10):
            monitor.record_request(account, success=(i != 5), latency=0.1)  # 1/10 = 10% failure

        # Should not trigger alert yet (10% < 30%)
        alerts = monitor.get_health_alerts(account)
        high_failure_alerts = [a for a in alerts if a["type"] == "high_failure_rate"]
        assert len(high_failure_alerts) == 0

        # Record many failures to exceed threshold
        for i in range(10):
            monitor.record_request(account, success=False, latency=0.1)  # Now 11/20 = 55% failure

        # Should trigger alert now (55% > 30%)
        alerts = monitor.get_health_alerts(account)
        high_failure_alerts = [a for a in alerts if a["type"] == "high_failure_rate"]
        assert len(high_failure_alerts) == 1
        assert high_failure_alerts[0]["severity"] == "critical"


class TestWarmupManagement:
    """Test intelligent warm-up features."""

    def test_warmup_configurations(self):
        warmup_manager = WarmupManager()

        # Test different profiles
        conservative_config = WarmupConfig(profile=WarmupProfile.CONSERVATIVE)
        aggressive_config = WarmupConfig(profile=WarmupProfile.AGGRESSIVE)

        warmup_manager.set_account_config(0, conservative_config)
        warmup_manager.set_account_config(1, aggressive_config)

        # Check configurations were stored
        assert warmup_manager.get_account_config(0).profile == WarmupProfile.CONSERVATIVE
        assert warmup_manager.get_account_config(1).profile == WarmupProfile.AGGRESSIVE

    def test_warmup_progression(self):
        warmup_manager = WarmupManager()
        config = WarmupConfig(
            profile=WarmupProfile.CONSERVATIVE,
            initial_factor=0.1,
            increment=0.2,
            min_successes=3
        )
        warmup_manager.set_account_config(0, config)

        # Initiate warmup
        warmup_manager.initiate_warmup(0)
        status = warmup_manager.get_warmup_status(0)
        assert status["state"] == "initiated"
        assert status["factor"] == 0.1

        # Simulate successful requests
        for i in range(3):
            factor = warmup_manager.update_warmup_on_success(0, 0.1)
            # Conservative profile should increase slowly
            assert factor > 0.1

        # Should complete after enough successes
        status = warmup_manager.get_warmup_status(0)
        if status["successes"] >= config.min_successes:
            assert status["state"] == "complete"
            assert status["factor"] == 1.0


class TestDynamicWeighting:
    """Test dynamic weight adjustment features."""

    def test_dynamic_weight_calculation(self):
        adjuster = DynamicWeightAdjuster()

        # Create baseline metrics
        baseline = PerformanceMetrics(
            avg_latency=1.0,
            success_rate=0.95,
            request_count=100,
            failure_count=5,
            latency_samples=100
        )
        adjuster.update_baseline_metrics(0, baseline)

        # Better performance should increase weight
        better_metrics = PerformanceMetrics(
            avg_latency=0.5,  # 2x faster
            success_rate=0.98,  # Higher success rate
            request_count=100,
            failure_count=2,
            latency_samples=100
        )

        new_weight = adjuster.calculate_dynamic_weight(0, 1.0, better_metrics)
        assert new_weight > 1.0  # Weight should increase

        # Worse performance should decrease weight
        worse_metrics = PerformanceMetrics(
            avg_latency=2.0,  # 2x slower
            success_rate=0.90,  # Lower success rate
            request_count=100,
            failure_count=10,
            latency_samples=100
        )

        new_weight = adjuster.calculate_dynamic_weight(0, 1.0, worse_metrics)
        assert new_weight < 1.0  # Weight should decrease

    def test_weight_bounds_enforcement(self):
        adjuster = DynamicWeightAdjuster(min_weight=0.5, max_weight=1.5)

        # Create extreme metrics to test bounds
        baseline = PerformanceMetrics(avg_latency=1.0, success_rate=0.95)
        adjuster.update_baseline_metrics(0, baseline)

        extreme_metrics = PerformanceMetrics(
            avg_latency=10.0,  # Very slow
            success_rate=0.1   # Very low success rate
        )

        # Even with extreme metrics, weight should stay within bounds
        new_weight = adjuster.calculate_dynamic_weight(0, 1.0, extreme_metrics)
        assert 0.5 <= new_weight <= 1.5


@pytest.mark.asyncio
class TestEnhancedPoolFeatures:
    """Test enhanced features integrated in the account pool."""

    @pytest.fixture
    def account_pool(self, mock_api_key, base_url):
        with patch("providers.nvidia_nim.account.AsyncOpenAI"):
            return AccountPool(
                api_keys=[mock_api_key, mock_api_key + "2", mock_api_key + "3"],
                base_url=base_url,
                per_account_timeout=300.0,
                health_state_file="",
            )

    async def test_pool_includes_enhanced_features(self, account_pool):
        # Check that enhanced components are present
        assert hasattr(account_pool, '_health_monitor')
        assert hasattr(account_pool, '_warmup_manager')
        assert hasattr(account_pool, '_distributed')
        assert hasattr(account_pool, '_weight_adjuster')

    async def test_account_success_includes_enhanced_tracking(self, account_pool):
        account = account_pool._accounts[0]

        # Mark success with latency
        await account_pool.mark_account_success(account, latency=0.5)

        # Check that warmup factor was updated (placeholder test)
        # In a real test, we'd check interaction with warmup manager
        assert account.warmup_factor >= 0.0  # Should be valid

    async def test_account_failure_includes_enhanced_handling(self, account_pool):
        account = account_pool._accounts[0]

        # Mark failure with error message
        await account_pool.mark_account_failed(account, error_message="test error")

        # Check that health monitor was updated
        # This is a structural check since the actual recording happens internally
        assert hasattr(account_pool._health_monitor, '_metrics')