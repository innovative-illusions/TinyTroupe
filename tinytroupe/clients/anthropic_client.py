"""
Anthropic Claude client for TinyTroupe.

Requires the `anthropic` Python package:
    pip install anthropic

Set your API key via the ANTHROPIC_API_KEY environment variable (or
ANTHROPIC_BASE_URL for a custom endpoint).
"""

import logging
import os
import pickle
import time

from tinytroupe import config_manager, utils

logger = logging.getLogger("tinytroupe")


class AnthropicClient:
    """
    Client for Anthropic's Claude API.

    Translates between the OpenAI-style message format used internally by
    TinyTroupe and the Anthropic Messages API format, so the rest of the
    codebase does not need to know which backend is active.

    Message format conversion
    -------------------------
    TinyTroupe (OpenAI-style) → Anthropic:
      * The first ``role: system`` message (if any) is extracted and passed
        as the top-level ``system`` parameter.
      * Remaining messages are forwarded as-is (Anthropic accepts
        ``role: user`` / ``role: assistant``).

    Response conversion
    -------------------
    Anthropic → TinyTroupe (OpenAI-style):
      Returns ``{"role": "assistant", "content": "<text>"}`` to match what
      the rest of the stack expects.
    """

    @config_manager.config_defaults(
        cache_api_calls="cache_api_calls",
        cache_file_name="cache_file_name",
    )
    def __init__(self, cache_api_calls=None, cache_file_name=None) -> None:
        logger.debug("Initializing AnthropicClient")
        self._client = None  # lazy-initialised in _get_client()
        self.set_api_cache(cache_api_calls, cache_file_name)

    # ------------------------------------------------------------------
    # Cache helpers (same pattern as the other clients)
    # ------------------------------------------------------------------

    def set_api_cache(self, cache_api_calls, cache_file_name=None):
        self.cache_api_calls = cache_api_calls
        self.cache_file_name = cache_file_name
        if self.cache_api_calls:
            self.api_cache = self._load_cache()

    def _load_cache(self):
        if self.cache_file_name and os.path.exists(self.cache_file_name):
            try:
                with open(self.cache_file_name, "rb") as fh:
                    return pickle.load(fh)
            except (EOFError, pickle.UnpicklingError) as exc:
                logger.warning("Cache file unreadable (%s). Starting fresh.", exc)
        return {}

    def _save_cache(self):
        with open(self.cache_file_name, "wb") as fh:
            pickle.dump(self.api_cache, fh)

    # ------------------------------------------------------------------
    # Client initialisation (lazy so the import error is deferred until
    # the user actually tries to use the Anthropic provider)
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required to use the Anthropic provider. "
                "Install it with:  pip install anthropic"
            ) from exc

        api_key = os.getenv("ANTHROPIC_API_KEY")
        base_url = config_manager.get("base_url") or os.getenv("ANTHROPIC_BASE_URL")

        kwargs = {"api_key": api_key, "max_retries": 0}
        if base_url:
            kwargs["base_url"] = base_url

        self._client = anthropic.Anthropic(**kwargs)
        return self._client

    # ------------------------------------------------------------------
    # Message sending
    # ------------------------------------------------------------------

    @config_manager.config_defaults(
        model="model",
        temperature="temperature",
        max_completion_tokens="max_completion_tokens",
        top_p="top_p",
        timeout="timeout",
        max_attempts="max_attempts",
        waiting_time="waiting_time",
        exponential_backoff_factor="exponential_backoff_factor",
    )
    def send_message(
        self,
        current_messages,
        dedent_messages=True,
        model=None,
        temperature=None,
        max_completion_tokens=None,
        top_p=None,
        stop=None,
        timeout=None,
        max_attempts=None,
        waiting_time=None,
        exponential_backoff_factor=None,
        n=1,
        response_format=None,
        enable_pydantic_model_return=False,
        echo=False,
    ):
        """Send a chat message to the Anthropic API and return the reply."""

        from tinytroupe.clients import InvalidRequestError, NonTerminalError  # avoid circular

        def aux_exponential_backoff():
            nonlocal waiting_time
            if waiting_time <= 0:
                waiting_time = 2
            logger.info("Request failed. Waiting %s seconds…", waiting_time)
            time.sleep(waiting_time)
            waiting_time = waiting_time * exponential_backoff_factor

        # Dedent message content if requested
        if dedent_messages:
            current_messages = [
                {**m, "content": utils.dedent(m["content"])} if "content" in m else m
                for m in current_messages
            ]

        # ------------------------------------------------------------------
        # Convert OpenAI-style messages → Anthropic format
        # ------------------------------------------------------------------
        system_prompt = None
        anthropic_messages = []

        for msg in current_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                # Anthropic takes a single system string; concatenate if multiple
                system_prompt = (system_prompt + "\n" + content) if system_prompt else content
            else:
                anthropic_messages.append({"role": role, "content": content})

        # Build API call parameters
        api_params = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_completion_tokens or 1024,
        }
        if system_prompt:
            api_params["system"] = system_prompt
        if temperature is not None:
            api_params["temperature"] = temperature
        if top_p is not None:
            api_params["top_p"] = top_p
        if stop:
            api_params["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        cache_key = str((model, api_params))

        i = 0
        while i < max_attempts:
            try:
                i += 1
                start_time = time.monotonic()
                logger.debug("Sending request to Anthropic API. Attempt %d", i)

                if self.cache_api_calls and cache_key in self.api_cache:
                    raw_response = self.api_cache[cache_key]
                else:
                    logger.info("Waiting %s seconds before next API request…", waiting_time)
                    time.sleep(waiting_time)

                    client = self._get_client()
                    raw_response = client.messages.create(**api_params)

                    if self.cache_api_calls:
                        self.api_cache[cache_key] = raw_response
                        self._save_cache()

                elapsed = time.monotonic() - start_time
                logger.debug("Got response in %.2fs after %d attempt(s)", elapsed, i)

                return utils.sanitize_dict(self._extract_response(raw_response))

            except Exception as exc:
                error_str = str(exc)
                logger.error("[%d] Anthropic API error: %s", i, error_str)
                if "invalid_request_error" in error_str.lower():
                    raise InvalidRequestError(error_str)
                aux_exponential_backoff()

        logger.error("Failed to get response after %d attempts", max_attempts)
        return None

    # ------------------------------------------------------------------
    # Response normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_response(response) -> dict:
        """Convert an Anthropic Message object to TinyTroupe's expected dict."""
        try:
            # anthropic SDK returns a Message object with a content list
            text = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            return {"role": "assistant", "content": text}
        except Exception as exc:
            logger.error("Error extracting Anthropic response: %s | response: %s", exc, response)
            raise ValueError("Unexpected response format from Anthropic API") from exc

    # ------------------------------------------------------------------
    # Token counting (approximate — Anthropic charges per token server-side)
    # ------------------------------------------------------------------

    def _count_tokens(self, messages: list, model: str):
        """Best-effort token count using the Anthropic SDK's count_tokens helper."""
        try:
            client = self._get_client()
            system_prompt = None
            anthropic_messages = []
            for m in messages:
                if m.get("role") == "system":
                    system_prompt = m.get("content", "")
                else:
                    anthropic_messages.append({"role": m["role"], "content": m.get("content", "")})

            kwargs = {"model": model, "messages": anthropic_messages}
            if system_prompt:
                kwargs["system"] = system_prompt

            result = client.messages.count_tokens(**kwargs)
            return result.input_tokens
        except Exception as exc:
            logger.debug("Token counting unavailable: %s", exc)
            return None
