import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from providers.nvidia_nim.account import NimApiAccount
from providers.nvidia_nim.pool import AccountPool
from providers.nvidia_nim.rotator import (
    AccountRotator,
    AllAccountsExhaustedError,
    PerAccountRateLimiter,
    RotationStrategy,
)
from openai import APIError, RateLimitError


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


class TestNimApiAccount:
    """Test NimApiAccount dataclass with health tracking."""

    def test_constructor_creates_client_with_correct_params(self, mock_api_key, base_url):
        with patch("providers.nvidia_nim.account.AsyncOpenAI") as mock_openai:
            account = NimApiAccount(
                api_key=mock_api_key,
                base_url=base_url,
                index=0,
                timeout=300.0,
            )

            mock_openai.assert_called_once_with(
                api_key=mock_api_key,
                base_url=base_url,
                max_retries=0,
                timeout=300.0,
            )
            assert account.api_key == mock_api_key
            assert account.base_url == base_url
            assert account.index == 0
            assert account.timeout == 300.0
            assert account.request_count == 0
            assert account.failure_count == 0

    def test_is_available_returns_true_when_blocked_until_in_past(self, mock_api_key, base_url):
        with patch("providers.nvidia_nim.account.AsyncOpenAI"):
            account = NimApiAccount(
                api_key=mock_api_key,
                base_url=base_url,
                index=0,
            )
            account.blocked_until = time.time() - 10

            assert account.is_available() is True

    def test_is_available_returns_false_when_blocked_until_in_future(self, mock_api_key, base_url):
        with patch("providers.nvidia_nim.account.AsyncOpenAI"):
            account = NimApiAccount(
                api_key=mock_api_key,
                base_url=base_url,
                index=0,
            )
            account.blocked_until = time.time() + 60

            assert account.is_available() is False

    def test_is_healthy_returns_true_when_below_threshold(self, mock_api_key, base_url):
        with patch("providers.nvidia_nim.account.AsyncOpenAI"):
            account = NimApiAccount(
                api_key=mock_api_key,
                base_url=base_url,
                index=0,
            )
            account.failure_count = 2

            assert account.is_healthy(max_failures=3) is True

    def test_is_healthy_auto_recovers_when_blocked_expired(self, mock_api_key, base_url):
        with patch("providers.nvidia_nim.account.AsyncOpenAI"):
            account = NimApiAccount(
                api_key=mock_api_key,
                base_url=base_url,
                index=0,
            )
            account.failure_count = 5
            account.blocked_until = time.time() - 10

            result = account.is_healthy(max_failures=3)

            assert result is True
            assert account.failure_count == 0

    def test_is_healthy_stays_unhealthy_when_still_blocked(self, mock_api_key, base_url):
        with patch("providers.nvidia_nim.account.AsyncOpenAI"):
            account = NimApiAccount(
                api_key=mock_api_key,
                base_url=base_url,
                index=0,
            )
            account.failure_count = 5
            account.blocked_until = time.time() + 60

            result = account.is_healthy(max_failures=3)

            assert result is False
            assert account.failure_count == 5

    def test_mark_used_updates_last_used_and_request_count(self, mock_api_key, base_url):
        with patch("providers.nvidia_nim.account.AsyncOpenAI"):
            account = NimApiAccount(
                api_key=mock_api_key,
                base_url=base_url,
                index=0,
            )
            initial_count = account.request_count
            before_time = time.time()

            account.mark_used()

            after_time = time.time()
            assert account.request_count == initial_count + 1
            assert before_time <= account.last_used <= after_time

    def test_mark_failed_increments_failure_count(self, mock_api_key, base_url):
        with patch("providers.nvidia_nim.account.AsyncOpenAI"):
            account = NimApiAccount(
                api_key=mock_api_key,
                base_url=base_url,
                index=0,
            )
            initial_failures = account.failure_count
            before_time = time.time()

            account.mark_failed()

            after_time = time.time()
            assert account.failure_count == initial_failures + 1
            assert before_time <= account.last_failure <= after_time

    def test_block_sets_blocked_until_correctly(self, mock_api_key, base_url):
        with patch("providers.nvidia_nim.account.AsyncOpenAI"):
            account = NimApiAccount(
                api_key=mock_api_key,
                base_url=base_url,
                index=0,
            )
            before_time = time.time()

            account.block(duration_seconds=60)

            after_time = time.time()
            assert before_time + 60 <= account.blocked_until <= after_time + 60

    def test_reset_failures_resets_failure_count_to_zero(self, mock_api_key, base_url):
        with patch("providers.nvidia_nim.account.AsyncOpenAI"):
            account = NimApiAccount(
                api_key=mock_api_key,
                base_url=base_url,
                index=0,
            )
            account.failure_count = 10

            account.reset_failures()

            assert account.failure_count == 0

    def test_cleanup_old_timestamps_removes_old_timestamps(self, mock_api_key, base_url):
        with patch("providers.nvidia_nim.account.AsyncOpenAI"):
            account = NimApiAccount(
                api_key=mock_api_key,
                base_url=base_url,
                index=0,
            )
            now = time.time()
            # timestamps: 100s ago (expired), 70s ago (expired), 30s ago (kept), now (kept)
            account.request_timestamps = [now - 100, now - 70, now - 30, now]

            account.cleanup_old_timestamps(window_seconds=60)

            assert len(account.request_timestamps) == 2
            assert all(ts > now - 60 for ts in account.request_timestamps)

    def test_add_request_timestamp_adds_current_time(self, mock_api_key, base_url):
        with patch("providers.nvidia_nim.account.AsyncOpenAI"):
            account = NimApiAccount(
                api_key=mock_api_key,
                base_url=base_url,
                index=0,
            )
            before_time = time.time()

            account.add_request_timestamp()

            after_time = time.time()
            assert len(account.request_timestamps) == 1
            assert before_time <= account.request_timestamps[0] <= after_time

    def test_str_masks_api_key(self, mock_api_key, base_url):
        with patch("providers.nvidia_nim.account.AsyncOpenAI"):
            account = NimApiAccount(
                api_key=mock_api_key,
                base_url=base_url,
                index=5,
            )
            account.request_count = 10
            account.failure_count = 2

            result = str(account)

            assert "nvapi-te..." in result
            assert "index=5" in result
            assert "requests=10" in result
            assert "failures=2" in result
            assert mock_api_key not in result


class TestAccountPool:
    """Test AccountPool management."""

    @pytest.fixture
    def api_keys(self):
        return ["key1", "key2", "key3"]

    @pytest.fixture
    def account_pool(self, api_keys, base_url):
        with patch("providers.nvidia_nim.account.AsyncOpenAI"):
            return AccountPool(
                api_keys=api_keys,
                base_url=base_url,
                per_account_timeout=300.0,
                health_state_file="",
            )

    def test_constructor_creates_correct_number_of_accounts(self, account_pool, api_keys):
        assert account_pool.size == len(api_keys)
        assert len(account_pool._accounts) == len(api_keys)

    def test_size_property_returns_correct_count(self, account_pool, api_keys):
        assert account_pool.size == len(api_keys)

    @pytest.mark.asyncio
    async def test_get_available_accounts_returns_only_healthy_available(self, account_pool):
        # Account 0: healthy and available
        account_pool._accounts[0].failure_count = 0
        account_pool._accounts[0].blocked_until = time.time() - 10

        # Account 1: blocked AND has failures (unavailable, so excluded)
        account_pool._accounts[1].failure_count = 5  # exceeds default max_failures=3
        account_pool._accounts[1].blocked_until = time.time() + 60

        # Account 2: blocked (not available)
        account_pool._accounts[2].failure_count = 0
        account_pool._accounts[2].blocked_until = time.time() + 60

        available = await account_pool.get_available_accounts(max_failures=3)

        assert len(available) == 1
        assert available[0].index == 0

    @pytest.mark.asyncio
    async def test_get_available_accounts_excludes_unhealthy_accounts(self, account_pool):
        for account in account_pool._accounts:
            account.failure_count = 5
            account.blocked_until = time.time() + 60

        available = await account_pool.get_available_accounts(max_failures=3)

        assert len(available) == 0

    @pytest.mark.asyncio
    async def test_get_account_by_index_returns_correct_account(self, account_pool):
        account = await account_pool.get_account_by_index(1)

        assert account is not None
        assert account.index == 1

    @pytest.mark.asyncio
    async def test_get_account_by_index_returns_none_for_invalid_index(self, account_pool):
        account = await account_pool.get_account_by_index(10)

        assert account is None

    @pytest.mark.asyncio
    async def test_mark_account_failed_increments_failure_and_blocks(self, account_pool):
        account = account_pool._accounts[0]
        initial_failures = account.failure_count
        before_time = time.time()

        await account_pool.mark_account_failed(account, cooldown_seconds=60)

        assert account.failure_count == initial_failures + 1
        assert account.blocked_until > before_time

    @pytest.mark.asyncio
    async def test_mark_account_success_resets_failures(self, account_pool):
        account = account_pool._accounts[0]
        account.failure_count = 5

        await account_pool.mark_account_success(account)

        assert account.failure_count == 0

    @pytest.mark.asyncio
    async def test_update_account_usage_updates_timestamps(self, account_pool):
        account = account_pool._accounts[0]
        initial_count = account.request_count
        initial_ts_count = len(account.request_timestamps)

        await account_pool.update_account_usage(account)

        assert account.request_count == initial_count + 1
        assert len(account.request_timestamps) == initial_ts_count + 1

    @pytest.mark.asyncio
    async def test_get_health_status_returns_correct_stats(self, account_pool):
        account_pool._accounts[0].failure_count = 0
        account_pool._accounts[0].blocked_until = time.time() - 10

        account_pool._accounts[1].failure_count = 5
        account_pool._accounts[1].blocked_until = time.time() - 10

        account_pool._accounts[2].failure_count = 0
        account_pool._accounts[2].blocked_until = time.time() + 60

        status = await account_pool.get_health_status()

        assert status["total"] == 3
        assert status["available"] == 2
        assert status["blocked"] == 1

    @pytest.mark.asyncio
    async def test_save_and_load_state_persistence_roundtrip(self, api_keys, base_url, tmp_path):
        state_file = str(tmp_path / "health.json")

        with patch("providers.nvidia_nim.account.AsyncOpenAI"):
            pool1 = AccountPool(
                api_keys=api_keys,
                base_url=base_url,
                health_state_file=state_file,
            )
            pool1._accounts[0].request_count = 10
            pool1._accounts[0].failure_count = 2
            pool1._accounts[0].blocked_until = 12345.0

            pool1.save_state()

            pool2 = AccountPool(
                api_keys=api_keys,
                base_url=base_url,
                health_state_file=state_file,
            )

            assert pool2._accounts[0].request_count == 10
            assert pool2._accounts[0].failure_count == 2
            assert pool2._accounts[0].blocked_until == 12345.0

    @pytest.mark.asyncio
    async def test_close_all_closes_all_clients(self, account_pool):
        # Store references to the clients before close_all sets them to None
        clients = []
        for account in account_pool._accounts:
            mock_client = AsyncMock()
            account.client = mock_client
            clients.append(mock_client)

        await account_pool.close_all()

        # Verify each stored client had close called
        for client in clients:
            client.close.assert_called_once()

        # Verify all account clients are now None
        for account in account_pool._accounts:
            assert account.client is None


class TestAccountRotator:
    """Test AccountRotator rotation strategies."""

    @pytest.fixture
    def api_keys(self):
        return ["key1", "key2", "key3"]

    @pytest.fixture
    def pool(self, api_keys, base_url):
        with patch("providers.nvidia_nim.account.AsyncOpenAI"):
            return AccountPool(
                api_keys=api_keys,
                base_url=base_url,
            )

    @pytest.fixture
    def rotator_round_robin(self, pool):
        return AccountRotator(
            pool=pool,
            strategy=RotationStrategy.ROUND_ROBIN,
            requests_per_minute=40,
            max_failures=3,
        )

    @pytest.fixture
    def rotator_least_used(self, pool):
        return AccountRotator(
            pool=pool,
            strategy=RotationStrategy.LEAST_USED,
            requests_per_minute=40,
            max_failures=3,
        )

    @pytest.fixture
    def rotator_on_failure(self, pool):
        return AccountRotator(
            pool=pool,
            strategy=RotationStrategy.ON_FAILURE,
            requests_per_minute=40,
            max_failures=3,
        )

    @pytest.mark.asyncio
    async def test_round_robin_cycles_through_accounts(self, rotator_round_robin, pool):
        for account in pool._accounts:
            account.blocked_until = 0
            account.failure_count = 0

        account1 = await rotator_round_robin.get_next_account()
        account2 = await rotator_round_robin.get_next_account()
        account3 = await rotator_round_robin.get_next_account()
        account4 = await rotator_round_robin.get_next_account()

        assert account1.index == 0
        assert account2.index == 1
        assert account3.index == 2
        assert account4.index == 0

    @pytest.mark.asyncio
    async def test_least_used_picks_oldest_used_account(self, rotator_least_used, pool):
        pool._accounts[0].last_used = time.time() - 100
        pool._accounts[1].last_used = time.time() - 50
        pool._accounts[2].last_used = time.time() - 200

        for account in pool._accounts:
            account.blocked_until = 0
            account.failure_count = 0

        account = await rotator_least_used.get_next_account()

        assert account.index == 2

    @pytest.mark.asyncio
    async def test_on_failure_sticks_to_same_account_until_failure(self, rotator_on_failure, pool):
        for account in pool._accounts:
            account.blocked_until = 0
            account.failure_count = 0

        account1 = await rotator_on_failure.get_next_account()
        account2 = await rotator_on_failure.get_next_account()
        account3 = await rotator_on_failure.get_next_account()

        assert account1.index == account2.index == account3.index

    @pytest.mark.asyncio
    async def test_on_failure_fails_over_on_rate_limit(self, rotator_on_failure, pool):
        for account in pool._accounts:
            account.blocked_until = 0
            account.failure_count = 0

        first_account = await rotator_on_failure.get_next_account()

        first_account.blocked_until = time.time() + 60

        second_account = await rotator_on_failure.get_next_account()

        assert first_account.index != second_account.index

    @pytest.mark.asyncio
    async def test_get_any_available_raises_when_all_exhausted(self, rotator_on_failure, pool):
        for account in pool._accounts:
            account.blocked_until = time.time() + 60
            account.failure_count = 5

        with pytest.raises(AllAccountsExhaustedError) as exc_info:
            await rotator_on_failure.get_any_available(wait=False)

        assert exc_info.value.total == 3

    @pytest.mark.asyncio
    async def test_execute_with_rotation_retries_on_rate_limit_error(self, rotator_on_failure):
        attempts = []

        async def operation(account):
            attempts.append(account.index)
            if len(attempts) < 2:
                raise RateLimitError("Rate limited", response=None, body=None)
            return "success"

        result = await rotator_on_failure.execute_with_rotation(operation)

        assert result == "success"
        assert len(attempts) == 2

    @pytest.mark.asyncio
    async def test_execute_with_rotation_retries_on_api_error(self, rotator_on_failure):
        attempts = []

        async def operation(account):
            attempts.append(account.index)
            if len(attempts) < 2:
                raise APIError("API error", request=None, body=None)
            return "success"

        result = await rotator_on_failure.execute_with_rotation(operation, retry_delay=0.1)

        assert result == "success"
        assert len(attempts) == 2

    @pytest.mark.asyncio
    async def test_execute_with_rotation_succeeds_on_first_try(self, rotator_on_failure):
        async def operation(account):
            return "success"

        result = await rotator_on_failure.execute_with_rotation(operation)

        assert result == "success"


class TestPerAccountRateLimiter:
    """Test PerAccountRateLimiter sliding window implementation."""

    @pytest.fixture
    def rate_limiter(self):
        return PerAccountRateLimiter(requests_per_minute=3)

    @pytest.fixture
    def account(self, mock_api_key, base_url):
        with patch("providers.nvidia_nim.account.AsyncOpenAI"):
            return NimApiAccount(
                api_key=mock_api_key,
                base_url=base_url,
                index=0,
            )

    def test_is_allowed_returns_true_under_limit(self, rate_limiter, account):
        account.request_timestamps = [time.time(), time.time()]

        result = rate_limiter.is_allowed(account)

        assert result is True

    def test_is_allowed_returns_false_at_limit(self, rate_limiter, account):
        account.request_timestamps = [time.time(), time.time(), time.time()]

        result = rate_limiter.is_allowed(account)

        assert result is False

    def test_get_wait_time_returns_zero_when_available(self, rate_limiter, account):
        account.request_timestamps = [time.time()]

        wait_time = rate_limiter.get_wait_time(account)

        assert wait_time == 0.0

    def test_get_wait_time_returns_positive_when_at_limit(self, rate_limiter, account):
        now = time.time()
        account.request_timestamps = [now - 30, now - 10, now]

        wait_time = rate_limiter.get_wait_time(account)

        assert wait_time > 0
        assert wait_time <= 60
