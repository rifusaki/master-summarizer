"""
OpenCode server HTTP client.

Manages the lifecycle of an OpenCode server process and provides
methods to create sessions, send prompts, and receive responses
from different LLM models.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from typing import Any

import httpx

from src.config import ModelConfig, PipelineConfig, pipeline_config

logger = logging.getLogger(__name__)


class OpenCodeError(Exception):
    """Base exception for OpenCode client errors."""


class OpenCodeTimeoutError(OpenCodeError):
    """Server or request timed out."""


class OpenCodeRateLimitError(OpenCodeError):
    """Server returned 429 rate limit response."""

    def __init__(self, message: str = "", retry_after: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class OpenCodeHTTPError(OpenCodeError):
    """Server returned a non-429 HTTP error response."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class OpenCodeServerError(OpenCodeHTTPError):
    """Legacy alias for backward compatibility."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message, status_code=status_code)


class OpenCodeClient:
    """
    HTTP client for the OpenCode server.

    Manages server lifecycle (auto-start) and provides typed methods
    for session management and LLM prompt dispatch.
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or pipeline_config
        self.base_url = self.config.opencode_base_url
        self._http: httpx.AsyncClient | None = None
        self._server_process: subprocess.Popen | None = None
        self._sessions: dict[str, str] = {}  # role -> session_id

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the OpenCode server and HTTP client."""
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=float(self.config.llm_timeout),
                write=10.0,
                pool=10.0,
            ),
        )

        # Check if server is already running
        if await self._health_check():
            logger.info("OpenCode server already running at %s", self.base_url)
            return

        # Auto-start the server
        logger.info("Starting OpenCode server on port %d...", self.config.opencode_port)
        self._server_process = subprocess.Popen(
            [
                "opencode",
                "serve",
                "--port",
                str(self.config.opencode_port),
                "--hostname",
                self.config.opencode_host,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to become healthy
        deadline = time.time() + self.config.server_startup_timeout
        while time.time() < deadline:
            if await self._health_check():
                logger.info("OpenCode server started successfully")
                return
            await asyncio.sleep(0.5)

        raise OpenCodeTimeoutError(
            f"OpenCode server did not start within {self.config.server_startup_timeout}s"
        )

    async def stop(self) -> None:
        """Shut down the HTTP client and server process."""
        if self._http:
            await self._http.aclose()
            self._http = None

        if self._server_process:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            self._server_process = None
            logger.info("OpenCode server stopped")

    async def _health_check(self) -> bool:
        """Check if the server is healthy."""
        try:
            assert self._http is not None
            resp = await self._http.get("/global/health")
            data = resp.json()
            return data.get("healthy", False)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def get_or_create_session(self, role: str, title: str = "") -> str:
        """
        Get existing session for a role, or create a new one.

        Each agent role gets its own session to keep context separated.
        """
        if role in self._sessions:
            return self._sessions[role]

        session_title = title or f"pipeline-{role}"
        resp = await self._request(
            "POST",
            "/session",
            json={"title": session_title},
        )
        session_id = resp["id"]
        self._sessions[role] = session_id
        logger.info("Created session %s for role '%s'", session_id, role)
        return session_id

    async def create_fresh_session(self, title: str = "") -> str:
        """Create a new session without caching it to a role."""
        resp = await self._request(
            "POST",
            "/session",
            json={"title": title or "pipeline-task"},
        )
        return resp["id"]

    # ------------------------------------------------------------------
    # Core LLM interaction
    # ------------------------------------------------------------------

    async def send_prompt(
        self,
        session_id: str,
        model: ModelConfig,
        user_prompt: str,
        system_prompt: str | None = None,
        image_base64: str | None = None,
        image_media_type: str = "image/png",
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Send a prompt to the LLM via OpenCode and wait for the response.

        Args:
            session_id: The session to send the prompt in.
            model: Model configuration specifying provider and model ID.
            user_prompt: The user message text.
            system_prompt: Optional system prompt override.
            image_base64: Optional base64-encoded image for multimodal.
            image_media_type: MIME type of the image.
            json_schema: Optional JSON schema for structured output.

        Returns:
            Dict with 'text' (response content) and 'usage' (token counts).
        """
        # Build message parts
        parts: list[dict[str, Any]] = []

        if image_base64 and model.supports_images:
            # OpenCode uses FilePartInput for images with a data URL
            data_url = f"data:{image_media_type};base64,{image_base64}"
            parts.append(
                {
                    "type": "file",
                    "mime": image_media_type,
                    "url": data_url,
                }
            )

        parts.append({"type": "text", "text": user_prompt})

        # Build request body
        body: dict[str, Any] = {
            "model": {
                "providerID": model.provider_id,
                "modelID": model.model_id,
            },
            "parts": parts,
            "tools": {},  # Disable all tools for pure inference
        }

        if system_prompt:
            body["system"] = system_prompt

        if json_schema:
            body["format"] = {
                "type": "json_schema",
                "schema": json_schema,
                "retryCount": 2,
            }

        # Send and wait for response
        resp = await self._request(
            "POST",
            f"/session/{session_id}/message",
            json=body,
        )

        # Parse response
        return self._parse_response(resp)

    async def send_prompt_for_role(
        self,
        role: str,
        user_prompt: str,
        system_prompt: str | None = None,
        image_base64: str | None = None,
        image_media_type: str = "image/png",
        json_schema: dict[str, Any] | None = None,
        model_override: ModelConfig | None = None,
    ) -> dict[str, Any]:
        """
        High-level method: send a prompt using the model assigned to a role.

        Automatically manages sessions per role.
        """
        from src.config import MODELS

        model = model_override or MODELS[role]
        session_id = await self.get_or_create_session(role)

        return await self.send_prompt(
            session_id=session_id,
            model=model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            image_base64=image_base64,
            image_media_type=image_media_type,
            json_schema=json_schema,
        )

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    async def send_prompts_batch(
        self,
        role: str,
        prompts: list[dict[str, Any]],
        concurrency: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Send multiple prompts concurrently with controlled parallelism.

        Each dict in prompts should have keys matching send_prompt_for_role args:
        'user_prompt', and optionally 'system_prompt', 'image_base64', 'json_schema'.
        """
        semaphore = asyncio.Semaphore(concurrency)
        results: list[dict[str, Any]] = []

        async def _process(idx: int, prompt_kwargs: dict) -> tuple[int, dict]:
            async with semaphore:
                # Each batch item gets a fresh session to avoid context bleed
                session_id = await self.create_fresh_session(
                    title=f"batch-{role}-{idx}"
                )
                from src.config import MODELS

                model = MODELS[role]
                result = await self.send_prompt(
                    session_id=session_id,
                    model=model,
                    **prompt_kwargs,
                )
                return idx, result

        tasks = [_process(i, p) for i, p in enumerate(prompts)]
        completed: list[tuple[int, dict] | BaseException] = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        # Sort by original index and handle errors
        result_map: dict[int, dict] = {}
        error_idx = 0
        for item in completed:
            if isinstance(item, BaseException):
                logger.error("Batch prompt failed: %s", item)
                result_map[error_idx] = {
                    "text": "",
                    "usage": {},
                    "error": str(item),
                }
                error_idx += 1
            else:
                assert isinstance(item, tuple)
                idx, result = item
                result_map[idx] = result

        return [result_map[i] for i in sorted(result_map.keys())]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an HTTP request to the OpenCode server."""
        assert self._http is not None, "Client not started. Call start() first."

        try:
            resp = await self._http.request(method, path, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException as exc:
            raise OpenCodeTimeoutError(f"Request timed out: {method} {path}") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            body = exc.response.text[:500]
            if status == 429:
                retry_after: int | None = None
                try:
                    retry_after = int(exc.response.headers.get("retry-after", ""))
                except (ValueError, TypeError):
                    pass
                raise OpenCodeRateLimitError(
                    f"Rate limited (429): {body}", retry_after=retry_after
                ) from exc
            raise OpenCodeHTTPError(
                f"Server error {status}: {body}", status_code=status
            ) from exc

    @staticmethod
    def _parse_response(resp: dict[str, Any]) -> dict[str, Any]:
        """Extract text content and usage from an OpenCode message response."""
        text_parts: list[str] = []
        structured_output = None
        usage = {}

        # Response format: {"info": {...}, "parts": [...]}
        info = resp.get("info", {})
        parts = resp.get("parts", [])

        # Check for structured output in info (primary location)
        if "structured" in info:
            structured_output = info["structured"]

        # Extract text and tool results from parts
        for part in parts:
            part_type = part.get("type", "")
            if part_type == "text":
                text_parts.append(part.get("text", ""))
            elif part_type == "tool":
                # Structured output comes as a StructuredOutput tool call
                if part.get("tool") == "StructuredOutput":
                    state = part.get("state", {})
                    tool_input = state.get("input")
                    if tool_input and structured_output is None:
                        structured_output = tool_input
                else:
                    # Other tool results — try to extract text content
                    state = part.get("state", {})
                    output = state.get("output", "")
                    if output and isinstance(output, str):
                        text_parts.append(output)

        # Extract token usage from info
        # OpenCode returns tokens as info.tokens with fields:
        # total, input, output, reasoning, cache.read, cache.write
        if "tokens" in info:
            tokens = info["tokens"]
            usage = {
                "input_tokens": tokens.get("input", 0),
                "output_tokens": tokens.get("output", 0),
                "total_tokens": tokens.get("total", 0),
                "reasoning_tokens": tokens.get("reasoning", 0),
                "cache_read_tokens": tokens.get("cache", {}).get("read", 0),
                "cache_write_tokens": tokens.get("cache", {}).get("write", 0),
            }
        elif "usage" in info:
            usage = info["usage"]

        result: dict[str, Any] = {
            "text": "\n".join(text_parts).strip(),
            "usage": usage,
        }

        if structured_output is not None:
            result["structured_output"] = structured_output

        # Check for errors
        error = info.get("error")
        if error:
            error_name = error.get("name", "")
            if error_name == "StructuredOutputError":
                result["structured_output_error"] = error.get("message", "")
            else:
                result["error"] = error.get("message", str(error))

        return result
