"""Admin endpoints for managing API key rotation at runtime."""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .dependencies import get_provider
from providers.base import BaseProvider
from providers.nvidia_nim.rotator import RotationStrategy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# =============================================================================
# Request/Response Models
# =============================================================================


class AddAccountRequest(BaseModel):
    """Request to add a new API key."""

    api_key: str = Field(..., description="NVIDIA NIM API key")


class AddAccountResponse(BaseModel):
    """Response after adding an account."""

    index: int = Field(..., description="Index of the newly added account")
    total: int = Field(..., description="Total number of accounts in pool")


class RemoveAccountResponse(BaseModel):
    """Response after removing an account."""

    removed: bool = Field(..., description="Whether account was successfully removed")
    total: int = Field(..., description="Total number of accounts remaining")


class SetStrategyRequest(BaseModel):
    """Request to change rotation strategy."""

    strategy: str = Field(
        ..., description="Rotation strategy: round_robin, least_used, or on_failure"
    )


class SetStrategyResponse(BaseModel):
    """Response after changing strategy."""

    strategy: str = Field(..., description="New rotation strategy")


class AccountDetail(BaseModel):
    """Detailed account information."""

    index: int
    api_key_masked: str
    request_count: int
    failure_count: int
    available: bool
    healthy: bool
    last_used: float
    blocked_until: float
    avg_latency: float
    latency_samples: int
    requests_in_window: int
    circuit_breaker_state: str
    half_open_probe_count: int
    consecutive_successes: int
    static_weight: float
    dynamic_weight: float
    failure_rate_5min: float


class ListAccountsResponse(BaseModel):
    """Response with all account details."""

    total: int
    accounts: list[AccountDetail]


class ValidationResult(BaseModel):
    """Validation result for a single account."""

    index: int
    valid: bool
    error: Optional[str] = None


class ValidateAccountsResponse(BaseModel):
    """Response from account validation."""

    results: list[ValidationResult]
    total: int
    valid_count: int


class WarmupStatusResponse(BaseModel):
    """Response with warm-up status information."""

    account_index: int
    configured: bool
    state: str
    factor: float
    progress: float
    profile: str
    successes: int
    failures: int
    elapsed_time: float
    estimated_completion: Optional[float]


class QuotaReportResponse(BaseModel):
    """Response with quota usage report."""

    account_index: int
    has_quotas: bool
    limits: Dict[str, int]
    current_usage: Dict[str, int]
    utilization: Dict[str, float]
    alerts: list
    predictions: Dict[str, Any]


class DynamicWeightReportResponse(BaseModel):
    """Response with dynamic weight adjustment report."""

    account_index: int
    strategy: str
    current_weight: float
    adjusted_weight: float
    adjustment_made: bool
    factors: Dict[str, float]
    metrics: Dict[str, Any]


# =============================================================================
# Admin Routes
# =============================================================================


@router.post("/accounts", response_model=AddAccountResponse)
async def add_account(
    request: AddAccountRequest,
    provider: BaseProvider = Depends(get_provider),
):
    """Add a new API key to the rotation pool.

    Only works in multi-account mode. Returns 400 if running in single-key mode.
    """
    rotator = getattr(provider, "_rotator", None)
    pool = getattr(provider, "_pool", None)

    if rotator is None or pool is None:
        raise HTTPException(
            status_code=400,
            detail="Not running in multi-account mode. Cannot add accounts dynamically.",
        )

    try:
        index = await pool.add_account(request.api_key)
        return AddAccountResponse(index=index, total=pool.size)
    except Exception as e:
        logger.error(f"Failed to add account: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/accounts/{index}", response_model=RemoveAccountResponse)
async def remove_account(
    index: int,
    provider: BaseProvider = Depends(get_provider),
):
    """Remove an account from the pool by index.

    Returns 404 if account not found.
    """
    rotator = getattr(provider, "_rotator", None)
    pool = getattr(provider, "_pool", None)

    if rotator is None or pool is None:
        raise HTTPException(
            status_code=400,
            detail="Not running in multi-account mode. Cannot remove accounts.",
        )

    removed = await pool.remove_account(index)

    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"Account with index {index} not found",
        )

    return RemoveAccountResponse(removed=True, total=pool.size)


@router.put("/strategy", response_model=SetStrategyResponse)
async def set_strategy(
    request: SetStrategyRequest,
    provider: BaseProvider = Depends(get_provider),
):
    """Change the rotation strategy.

    Valid strategies: round_robin, least_used, on_failure
    """
    rotator = getattr(provider, "_rotator", None)

    if rotator is None:
        raise HTTPException(
            status_code=400,
            detail="Not running in multi-account mode. Cannot change strategy.",
        )

    # Validate strategy
    try:
        strategy = RotationStrategy(request.strategy)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy: {request.strategy}. "
            f"Valid options: round_robin, least_used, on_failure",
        )

    rotator.set_strategy(strategy)
    logger.info(f"ADMIN: Changed rotation strategy to {strategy.value}")

    return SetStrategyResponse(strategy=strategy.value)


@router.get("/accounts", response_model=ListAccountsResponse)
async def list_accounts(provider: BaseProvider = Depends(get_provider)):
    """List all accounts with detailed statistics.

    Returns masked API keys and health metrics for each account.
    """
    pool = getattr(provider, "_pool", None)

    if pool is None:
        raise HTTPException(
            status_code=400,
            detail="Not running in multi-account mode.",
        )

    accounts = pool.get_all_accounts()
    account_details = []

    for acc in accounts:
        # Mask API key (show first 8 chars)
        masked_key = acc.api_key[:8] + "..." if len(acc.api_key) > 8 else "***"

        account_details.append(
            AccountDetail(
                index=acc.index,
                api_key_masked=masked_key,
                request_count=acc.request_count,
                failure_count=acc.failure_count,
                available=acc.is_available(),
                healthy=acc.is_healthy(),
                last_used=acc.last_used,
                blocked_until=acc.blocked_until,
                avg_latency=acc.avg_latency,
                latency_samples=acc.latency_samples,
                requests_in_window=acc.get_request_count_in_window(),
                circuit_breaker_state=acc.circuit_breaker_state.value,
                half_open_probe_count=acc.half_open_probe_count,
                consecutive_successes=acc.consecutive_successes,
                static_weight=acc.weight,
                dynamic_weight=getattr(acc, 'dynamic_weight', acc.weight),
                failure_rate_5min=getattr(acc, 'failure_rate_5min', 0.0),
            )
        )

    return ListAccountsResponse(total=len(accounts), accounts=account_details)


@router.post("/validate", response_model=ValidateAccountsResponse)
async def validate_accounts(provider: BaseProvider = Depends(get_provider)):
    """Validate all accounts by making lightweight test requests.

    This is an expensive operation (makes real API calls). Use sparingly.
    """
    pool = getattr(provider, "_pool", None)

    if pool is None:
        raise HTTPException(
            status_code=400,
            detail="Not running in multi-account mode.",
        )

    try:
        results = await pool.validate_accounts()
        valid_count = sum(1 for r in results if r["valid"])

        return ValidateAccountsResponse(
            results=[ValidationResult(**r) for r in results],
            total=len(results),
            valid_count=valid_count,
        )
    except Exception as e:
        logger.error(f"Account validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/warmup/{index}", response_model=WarmupStatusResponse)
async def get_warmup_status(
    index: int,
    provider: BaseProvider = Depends(get_provider),
):
    """Get warm-up status for a specific account."""
    pool = getattr(provider, "_pool", None)

    if pool is None:
        raise HTTPException(
            status_code=400,
            detail="Not running in multi-account mode.",
        )

    # Access warmup manager from pool
    warmup_manager = getattr(pool, "_warmup_manager", None)
    if warmup_manager is None:
        raise HTTPException(
            status_code=500,
            detail="Warmup manager not available.",
        )

    try:
        status = warmup_manager.get_warmup_status(index)
        return WarmupStatusResponse(**status)
    except Exception as e:
        logger.error(f"Warmup status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/warmup/{index}/reset")
async def reset_warmup(
    index: int,
    provider: BaseProvider = Depends(get_provider),
):
    """Reset warm-up state for a specific account."""
    pool = getattr(provider, "_pool", None)

    if pool is None:
        raise HTTPException(
            status_code=400,
            detail="Not running in multi-account mode.",
        )

    # Access warmup manager from pool
    warmup_manager = getattr(pool, "_warmup_manager", None)
    if warmup_manager is None:
        raise HTTPException(
            status_code=500,
            detail="Warmup manager not available.",
        )

    try:
        warmup_manager.reset_warmup(index)
        return {"status": "success", "message": f"Warmup reset for account {index}"}
    except Exception as e:
        logger.error(f"Warmup reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/warmup/{index}/factor/{factor}")
async def set_manual_warmup_factor(
    index: int,
    factor: float,
    provider: BaseProvider = Depends(get_provider),
):
    """Set manual warm-up factor for a specific account."""
    pool = getattr(provider, "_pool", None)

    if pool is None:
        raise HTTPException(
            status_code=400,
            detail="Not running in multi-account mode.",
        )

    # Validate factor range
    if not 0.0 <= factor <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Factor must be between 0.0 and 1.0",
        )

    # Access warmup manager from pool
    warmup_manager = getattr(pool, "_warmup_manager", None)
    if warmup_manager is None:
        raise HTTPException(
            status_code=500,
            detail="Warmup manager not available.",
        )

    try:
        warmup_manager.set_manual_warmup_factor(index, factor)

        # Update account's warmup factor directly
        account = await pool.get_account_by_index(index)
        if account:
            account.warmup_factor = factor

        return {"status": "success", "message": f"Warmup factor set to {factor} for account {index}"}
    except Exception as e:
        logger.error(f"Warmup factor setting failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quota/{index}", response_model=QuotaReportResponse)
async def get_quota_report(
    index: int,
    provider: BaseProvider = Depends(get_provider),
):
    """Get quota usage report for a specific account."""
    pool = getattr(provider, "_pool", None)

    if pool is None:
        raise HTTPException(
            status_code=400,
            detail="Not running in multi-account mode.",
        )

    # Access quota manager (would need to be added to pool)
    # For now, return a placeholder response
    try:
        return QuotaReportResponse(
            account_index=index,
            has_quotas=False,
            limits={"hourly": 0, "daily": 0, "monthly": 0},
            current_usage={"hourly": 0, "daily": 0, "monthly": 0},
            utilization={"hourly": 0.0, "daily": 0.0, "monthly": 0.0},
            alerts=[],
            predictions={"predictions": {}, "trend_analysis": "insufficient_data"},
        )
    except Exception as e:
        logger.error(f"Quota report failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weights/dynamic/{index}", response_model=DynamicWeightReportResponse)
async def get_dynamic_weight_report(
    index: int,
    provider: BaseProvider = Depends(get_provider),
):
    """Get dynamic weight adjustment report for a specific account."""
    pool = getattr(provider, "_pool", None)

    if pool is None:
        raise HTTPException(
            status_code=400,
            detail="Not running in multi-account mode.",
        )

    # Access weight adjuster from pool
    weight_adjuster = getattr(pool, "_weight_adjuster", None)
    if weight_adjuster is None:
        raise HTTPException(
            status_code=500,
            detail="Weight adjuster not available.",
        )

    try:
        # Get account for current weight
        account = await pool.get_account_by_index(index)
        if not account:
            raise HTTPException(
                status_code=404,
                detail=f"Account {index} not found",
            )

        # Create performance metrics for report
        from providers.nvidia_nim.dynamic_weight import PerformanceMetrics

        # In a real implementation, we'd get actual metrics from the health monitor
        metrics = PerformanceMetrics(
            avg_latency=account.avg_latency,
            success_rate=1.0,  # Placeholder
            request_count=account.request_count,
            failure_count=account.failure_count,
            latency_samples=account.latency_samples,
        )

        report = weight_adjuster.get_weight_adjustment_report(
            index, account.weight, metrics
        )

        return DynamicWeightReportResponse(**report)
    except Exception as e:
        logger.error(f"Dynamic weight report failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/weights/update")
async def update_dynamic_weights(provider: BaseProvider = Depends(get_provider)):
    """Trigger update of dynamic weights for all accounts."""
    pool = getattr(provider, "_pool", None)

    if pool is None:
        raise HTTPException(
            status_code=400,
            detail="Not running in multi-account mode.",
        )

    try:
        await pool.update_dynamic_weights()
        return {"status": "success", "message": "Dynamic weights updated"}
    except Exception as e:
        logger.error(f"Dynamic weight update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
