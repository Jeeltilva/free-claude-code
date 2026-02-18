"""Dependency injection for FastAPI."""

from typing import Optional

from config.settings import Settings, get_settings as _get_settings, NVIDIA_NIM_BASE_URL
from providers.base import BaseProvider, ProviderConfig


# Global provider instance (singleton)
_provider: Optional[BaseProvider] = None


def get_settings() -> Settings:
    """Get application settings via dependency injection."""
    return _get_settings()


def get_provider() -> BaseProvider:
    """Get or create the provider instance based on settings.provider_type."""
    global _provider
    if _provider is None:
        settings = get_settings()

        if settings.provider_type == "nvidia_nim":
            from providers.nvidia_nim import NvidiaNimProvider

            # Get all API keys (supports both single and multiple)
            api_keys = settings.get_api_keys()

            if not api_keys:
                raise ValueError(
                    "No NVIDIA NIM API keys configured. "
                    "Set NVIDIA_NIM_API_KEY or NVIDIA_NIM_API_KEYS"
                )

            config = ProviderConfig(
                api_key=api_keys[0],  # Primary key for backward compatibility
                base_url=NVIDIA_NIM_BASE_URL,
                rate_limit=settings.nvidia_nim_rate_limit,
                rate_window=settings.nvidia_nim_rate_window,
                nim_settings=settings.nim,
            )
            _provider = NvidiaNimProvider(config, api_keys=api_keys)
        else:
            raise ValueError(
                f"Unknown provider_type: '{settings.provider_type}'. "
                f"Supported: 'nvidia_nim'"
            )
    return _provider


async def cleanup_provider():
    """Cleanup provider resources."""
    global _provider
    if _provider:
        # Close multi-account pool if present
        pool = getattr(_provider, "_pool", None)
        if pool:
            await pool.close_all()
        # Close single-key client if present
        client = getattr(_provider, "_client", None)
        if client:
            await client.aclose()
        _provider = None
