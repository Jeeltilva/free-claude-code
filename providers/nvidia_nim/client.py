"""NVIDIA NIM provider implementation."""

import logging
import json
import time
import uuid
from typing import Any, AsyncIterator, List, Optional

from openai import AsyncOpenAI

from providers.base import BaseProvider, ProviderConfig
from providers.rate_limit import GlobalRateLimiter
from .request import build_request_body
from .response import convert_response
from .errors import map_error
from .metrics import MetricsCollector
from .utils import (
    SSEBuilder,
    map_stop_reason,
    ThinkTagParser,
    HeuristicToolParser,
    ContentType,
)

logger = logging.getLogger(__name__)


class NvidiaNimProvider(BaseProvider):
    """NVIDIA NIM provider using official OpenAI client with multi-account rotation."""

    def __init__(self, config: ProviderConfig, api_keys: Optional[List[str]] = None):
        super().__init__(config)
        self._nim_settings = config.nim_settings

        # Support both single key (backward compatible) and multiple keys
        keys = api_keys or ([config.api_key] if config.api_key else [])
        if not keys:
            raise ValueError("At least one API key is required")

        self._base_url = (
            config.base_url or "https://integrate.api.nvidia.com/v1"
        ).rstrip("/")

        # Keep compatibility for tests that check these attributes
        self._api_key = keys[0] if keys else ""

        # Initialize multi-account rotation if multiple keys provided
        if len(keys) > 1:
            from .pool import AccountPool
            from .rotator import AccountRotator, RotationStrategy
            from config.settings import get_settings

            settings = get_settings()

            self._pool = AccountPool(
                keys,
                self._base_url,
                per_account_timeout=settings.nvidia_nim_per_account_timeout,
                health_state_file=settings.nvidia_nim_health_state_file,
            )

            # Map strategy string to enum
            strategy_str = getattr(settings, "nvidia_nim_rotation_strategy", "on_failure")
            try:
                strategy = RotationStrategy(strategy_str)
            except ValueError:
                strategy = RotationStrategy.ON_FAILURE

            self._rotator = AccountRotator(
                pool=self._pool,
                strategy=strategy,
                requests_per_minute=config.rate_limit or 40,
                max_failures=settings.nvidia_nim_max_failures,
                failure_cooldown=settings.nvidia_nim_failure_cooldown,
                max_retries=settings.nvidia_nim_max_retries,
                warmup_enabled=settings.nvidia_nim_warmup_enabled,
                warmup_increment=settings.nvidia_nim_warmup_increment,
                half_open_success_threshold=getattr(settings, "nvidia_nim_half_open_success_threshold", 3),
                half_open_max_probes=getattr(settings, "nvidia_nim_half_open_max_probes", 5),
            )
            self._client = None  # Not used with multi-account
            self._global_rate_limiter = None  # Not used with multi-account
        else:
            # Single key mode (backward compatible)
            self._pool = None
            self._rotator = None
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                max_retries=0,
                timeout=300.0,
            )
            self._global_rate_limiter = GlobalRateLimiter.get_instance(
                rate_limit=config.rate_limit,
                rate_window=config.rate_window,
            )

    def _build_request_body(self, request: Any, stream: bool = False) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(request, self._nim_settings, stream=stream)

    async def stream_response(
        self, request: Any, input_tokens: int = 0
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""
        message_id = f"msg_{uuid.uuid4()}"
        sse = SSEBuilder(message_id, request.model, input_tokens)

        body = self._build_request_body(request, stream=True)
        logger.info(
            f"NIM_STREAM: model={body.get('model')} msgs={len(body.get('messages', []))} tools={len(body.get('tools', []))}"
        )

        yield sse.message_start()

        think_parser = ThinkTagParser()
        heuristic_parser = HeuristicToolParser()

        finish_reason = None
        usage_info = None
        error_occurred = False
        error_message = ""

        # Metrics tracking
        metrics = MetricsCollector.get_instance()
        stream_start = time.monotonic()
        first_content_time: Optional[float] = None
        stream_account_ref: List[int] = [0]  # mutable container for account index

        try:
            async for chunk in self._call_nim_streaming(body, stream_account_ref):
                # OpenAI client returns objects, not JSON
                if getattr(chunk, "usage", None):
                    usage_info = chunk.usage

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                    logger.debug(f"NIM finish_reason: {finish_reason}")

                # Handle reasoning content from delta
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    if first_content_time is None:
                        first_content_time = time.monotonic()
                    for event in sse.ensure_thinking_block():
                        yield event
                    yield sse.emit_thinking_delta(reasoning)

                # Handle text content
                if delta.content:
                    if first_content_time is None:
                        first_content_time = time.monotonic()
                    for part in think_parser.feed(delta.content):
                        if part.type == ContentType.THINKING:
                            for event in sse.ensure_thinking_block():
                                yield event
                            yield sse.emit_thinking_delta(part.content)
                        else:
                            filtered_text, detected_tools = heuristic_parser.feed(
                                part.content
                            )

                            if filtered_text:
                                for event in sse.ensure_text_block():
                                    yield event
                                yield sse.emit_text_delta(filtered_text)

                            for tool_use in detected_tools:
                                for event in sse.close_content_blocks():
                                    yield event

                                block_idx = sse.blocks.allocate_index()
                                yield sse.content_block_start(
                                    block_idx,
                                    "tool_use",
                                    id=tool_use["id"],
                                    name=tool_use["name"],
                                )
                                yield sse.content_block_delta(
                                    block_idx,
                                    "input_json_delta",
                                    json.dumps(tool_use["input"]),
                                )
                                yield sse.content_block_stop(block_idx)

                # Handle native tool calls
                if delta.tool_calls:
                    for event in sse.close_content_blocks():
                        yield event
                    for tc in delta.tool_calls:
                        # Convert OpenAI tool call object to dict for existing logic
                        tc_info = {
                            "index": tc.index,
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for event in self._process_tool_call(tc_info, sse):
                            yield event

        except Exception as e:
            logger.error(f"NIM_ERROR: {type(e).__name__}: {e}")
            mapped_e = map_error(e)
            error_occurred = True
            error_message = str(mapped_e)
            logger.info(f"NIM_STREAM: Emitting SSE error event for {type(e).__name__}")
            # Ensure open blocks are closed before emitting error to follow Anthropic protocol
            for event in sse.close_content_blocks():
                yield event
            for event in sse.emit_error(error_message):
                yield event

        # Flush remaining content
        remaining = think_parser.flush()
        if remaining:
            if remaining.type == ContentType.THINKING:
                for event in sse.ensure_thinking_block():
                    yield event
                yield sse.emit_thinking_delta(remaining.content)
            else:
                for event in sse.ensure_text_block():
                    yield event
                yield sse.emit_text_delta(remaining.content)

        for tool_use in heuristic_parser.flush():
            for event in sse.close_content_blocks():
                yield event

            block_idx = sse.blocks.allocate_index()
            yield sse.content_block_start(
                block_idx,
                "tool_use",
                id=tool_use["id"],
                name=tool_use["name"],
            )
            yield sse.content_block_delta(
                block_idx,
                "input_json_delta",
                json.dumps(tool_use["input"]),
            )
            yield sse.content_block_stop(block_idx)

        if (
            not error_occurred
            and sse.blocks.text_index == -1
            and not sse.blocks.tool_indices
        ):
            for event in sse.ensure_text_block():
                yield event
            yield sse.emit_text_delta(" ")

        for event in sse.close_all_blocks():
            yield event

        output_tokens = (
            usage_info.completion_tokens
            if usage_info and hasattr(usage_info, "completion_tokens")
            else sse.estimate_output_tokens()
        )
        yield sse.message_delta(map_stop_reason(finish_reason), output_tokens)
        yield sse.message_stop()
        yield sse.done()

        # Record throughput metrics
        stream_end = time.monotonic()
        total_latency = stream_end - stream_start
        ttft_ms = (
            (first_content_time - stream_start) * 1000
            if first_content_time is not None
            else 0.0
        )
        streaming_duration = (
            stream_end - first_content_time
            if first_content_time is not None
            else total_latency
        )
        min_streaming_duration = 0.01  # 10ms floor to avoid inflated TPS
        tps = (
            output_tokens / max(streaming_duration, min_streaming_duration)
            if streaming_duration > 0 and output_tokens > 0
            else 0.0
        )

        if not error_occurred:
            metrics.record_request(
                account_index=stream_account_ref[0],
                ttft_ms=ttft_ms,
                tps=tps,
                total_latency_s=total_latency,
                output_tokens=output_tokens,
            )

    async def _call_nim_streaming(
        self, body: dict, account_ref: Optional[List[int]] = None
    ):
        """Call NIM streaming API - used by multi-account rotator.

        This is an async generator that yields raw OpenAI chunks.
        The rotator handles account selection and failover.

        Args:
            body: Request body dict.
            account_ref: Mutable list to write the account index into for the caller.
        """
        metrics = MetricsCollector.get_instance()

        if self._rotator:
            # Multi-account mode - use rotator for failover
            async def _stream_op(account):
                if account_ref is not None:
                    account_ref[0] = account.index
                metrics.increment_active_streams(account.index)
                try:
                    stream = await account.client.chat.completions.create(**body, stream=True)
                    async for chunk in stream:
                        yield chunk
                finally:
                    metrics.decrement_active_streams(account.index)

            async for chunk in self._rotator.execute_streaming_with_rotation(_stream_op):
                yield chunk
        else:
            # Single key mode (backward compatible)
            if account_ref is not None:
                account_ref[0] = 0
            stream = await self._global_rate_limiter.execute_with_retry(
                self._client.chat.completions.create, **body, stream=True
            )
            async for chunk in stream:
                yield chunk

    async def complete(self, request: Any) -> dict:
        """Make a non-streaming completion request."""
        body = self._build_request_body(request, stream=False)
        logger.info(
            f"NIM_COMPLETE: model={body.get('model')} msgs={len(body.get('messages', []))} tools={len(body.get('tools', []))}"
        )

        try:
            if self._rotator:
                # Multi-account rotation
                async def _complete(account):
                    response = await account.client.chat.completions.create(**body)
                    return response

                response = await self._rotator.execute_with_rotation(_complete)
                return response.model_dump()
            else:
                # Single key mode (backward compatible)
                response = await self._global_rate_limiter.execute_with_retry(
                    self._client.chat.completions.create, **body
                )
                return response.model_dump()
        except Exception as e:
            logger.error(f"NIM_ERROR: {type(e).__name__}: {e}")
            raise map_error(e)

    def convert_response(self, response_json: dict, original_request: Any) -> Any:
        """Convert provider response to Anthropic format."""
        return convert_response(response_json, original_request)

    def _process_tool_call(self, tc: dict, sse: Any):
        """Process a single tool call delta and yield SSE events.

        Args:
            tc: Tool call delta info dict
            sse: SSEBuilder instance
        """

        tc_index = tc.get("index", 0)
        if tc_index < 0:
            tc_index = len(sse.blocks.tool_indices)

        fn_delta = tc.get("function", {})
        if fn_delta.get("name") is not None:
            sse.blocks.tool_names[tc_index] = (
                sse.blocks.tool_names.get(tc_index, "") + fn_delta["name"]
            )

        if tc_index not in sse.blocks.tool_indices:
            name = sse.blocks.tool_names.get(tc_index, "")
            if name or tc.get("id"):
                tool_id = tc.get("id") or f"tool_{uuid.uuid4()}"
                yield sse.start_tool_block(tc_index, tool_id, name)
                sse.blocks.tool_started[tc_index] = True
        elif not sse.blocks.tool_started.get(tc_index) and sse.blocks.tool_names.get(
            tc_index
        ):
            tool_id = tc.get("id") or f"tool_{uuid.uuid4()}"
            name = sse.blocks.tool_names[tc_index]
            yield sse.start_tool_block(tc_index, tool_id, name)
            sse.blocks.tool_started[tc_index] = True

        args = fn_delta.get("arguments", "")
        if args:
            if not sse.blocks.tool_started.get(tc_index):
                tool_id = tc.get("id") or f"tool_{uuid.uuid4()}"
                name = sse.blocks.tool_names.get(tc_index, "tool_call") or "tool_call"

                yield sse.start_tool_block(tc_index, tool_id, name)
                sse.blocks.tool_started[tc_index] = True

            # INTERCEPTION: If this is a Task tool, force background=False
            current_name = sse.blocks.tool_names.get(tc_index, "")
            if current_name == "Task":
                try:
                    args_json = json.loads(args)
                    if args_json.get("run_in_background") is not False:
                        logger.info(
                            f"NIM_INTERCEPT: Forcing run_in_background=False for Task {tc.get('id', 'unknown')}"
                        )
                        args_json["run_in_background"] = False
                        args = json.dumps(args_json)
                except Exception as e:
                    logger.warning(
                        f"NIM_INTERCEPT: Failed to parse/modify Task args: {e}"
                    )

            yield sse.emit_tool_delta(tc_index, args)
