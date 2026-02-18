"""Centralized configuration using Pydantic Settings."""

from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

from .nim import NimSettings

load_dotenv()

# Fixed base URL for NVIDIA NIM
NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ==================== Provider Selection ====================
    provider_type: str = "nvidia_nim"

    # ==================== Messaging Platform Selection ====================
    messaging_platform: str = "telegram"

    # ==================== NVIDIA NIM Config ====================
    nvidia_nim_api_key: str = ""
    nvidia_nim_api_keys: str = ""  # Comma-separated keys for multi-account rotation
    nvidia_nim_rotation_strategy: Literal["round_robin", "least_used", "on_failure"] = "round_robin"
    nvidia_nim_failure_cooldown: int = 60
    nvidia_nim_max_failures: int = 3
    nvidia_nim_max_retries: int = 3
    nvidia_nim_per_account_timeout: float = 300.0
    nvidia_nim_health_state_file: str = ""
    nvidia_nim_default_weight: float = 1.0
    nvidia_nim_weights: str = ""  # Comma-separated per-account weights, overrides default
    nvidia_nim_daily_quota_limit: int = 0  # 0 = unlimited daily requests per account
    nvidia_nim_warmup_enabled: bool = True
    nvidia_nim_warmup_increment: float = 0.1
    nvidia_nim_warmup_successes_needed: int = 5
    nvidia_nim_half_open_success_threshold: int = 3
    nvidia_nim_half_open_max_probes: int = 5

    # Distributed persistence
    nvidia_nim_node_id: str = "default"

    # Dynamic weighting
    nvidia_nim_dynamic_weight_enabled: bool = True
    nvidia_nim_dynamic_weight_min: float = 0.1
    nvidia_nim_dynamic_weight_max: float = 2.0
    nvidia_nim_dynamic_weight_sensitivity: float = 1.0

    # ==================== Model ====================
    # All Claude model requests are mapped to this single model
    model: str = "moonshotai/kimi-k2-thinking"

    # ==================== Rate Limiting ====================
    nvidia_nim_rate_limit: int = 40
    nvidia_nim_rate_window: int = 60

    # ==================== Fast Prefix Detection ====================
    fast_prefix_detection: bool = True

    # ==================== Optimizations ====================
    enable_network_probe_mock: bool = True
    enable_title_generation_skip: bool = True
    enable_suggestion_mode_skip: bool = True
    enable_filepath_extraction_mock: bool = True

    # ==================== NIM Settings ====================
    nim: NimSettings = Field(default_factory=NimSettings)

    # ==================== Bot Wrapper Config ====================
    telegram_bot_token: Optional[str] = None
    allowed_telegram_user_id: Optional[str] = None
    claude_workspace: str = "./agent_workspace"
    allowed_dir: str = ""
    max_cli_sessions: int = 10

    # ==================== Server ====================
    host: str = "0.0.0.0"
    port: int = 8082
    log_file: str = "server.log"

    def get_api_keys(self) -> List[str]:
        """Get all configured API keys (single + multi-key support)."""
        keys = []
        if self.nvidia_nim_api_key:
            keys.append(self.nvidia_nim_api_key)
        if self.nvidia_nim_api_keys:
            keys.extend([k.strip() for k in self.nvidia_nim_api_keys.split(",")])
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for k in keys:
            if k and k not in seen:
                seen.add(k)
                result.append(k)
        return result

    # Handle empty strings for optional string fields
    @field_validator(
        "telegram_bot_token",
        "allowed_telegram_user_id",
        mode="before",
    )
    @classmethod
    def parse_optional_str(cls, v):
        if v == "":
            return None
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
