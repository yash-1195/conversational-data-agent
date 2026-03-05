"""
LLMClient — thin wrapper around the OpenAI-compatible Gemini API.

Responsibilities:
  - Read auth + model settings from LLMConfig (no hardcoded values)
  - Expose a single call() method that accepts a system prompt + message
    list and returns the assistant reply as a plain string
  - Expose a call_structured() method that returns the raw Choice object
    for callers that need finish_reason, token counts, etc.
  - Raise LLMClientError (not raw OpenAI SDK exceptions) so callers have
    one exception type to handle regardless of provider

Usage
-----
    from src.core.config_loader import load_config
    from src.core.llm_client import LLMClient

    cfg = load_config()
    client = LLMClient(cfg.llm)

    reply = client.call(
        system_prompt="You are a data analysis assistant.",
        messages=[{"role": "user", "content": "How many rows?"}],
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion import Choice
from src.core.config_loader import LLMConfig

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMClientError(Exception):
    """Base class for all LLMClient failures.

    Catch this to handle any API-level error without caring about the
    specific cause. Catch the subclasses for targeted handling.
    """


class LLMAuthError(LLMClientError):
    """Raised when the API key is missing or rejected (HTTP 401)."""


class LLMRateLimitError(LLMClientError):
    """Raised when the API returns a rate-limit response (HTTP 429)."""


class LLMTimeoutError(LLMClientError):
    """Raised when the API request times out."""


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """Stateless wrapper around the OpenAI-compatible Gemini client.

    Stateless here means no conversation history is stored — that is the
    responsibility of ContextManager. One instance can be created at
    application startup and shared across all agent nodes.

    Parameters
    ----------
    config:
        An LLMConfig instance (populated from configs/config.yaml).
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = self._build_client(config)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_client(config: LLMConfig) -> OpenAI:
        """Resolve the API key and build the SDK client.

        Raises LLMAuthError immediately if the key env var is unset,
        so failures are caught at startup rather than mid-run.
        """
        try:
            api_key = config.api_key          # property on LLMConfig; raises EnvironmentError if unset
        except EnvironmentError as exc:
            raise LLMAuthError(str(exc)) from exc

        return OpenAI(
            api_key=api_key,
            base_url=config.base_url,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(
        self,
        system_prompt: str,
        messages: list[ChatCompletionMessageParam],
    ) -> str:
        """Send a system prompt + messages to the LLM; return the reply string.

        The system prompt is prepended as a ``{"role": "system", ...}``
        message before the caller-supplied messages. This matches the
        contract of PromptBuilder: it builds the system prompt and message
        list separately, and this method combines them.

        Parameters
        ----------
        system_prompt:
            The full system prompt string from PromptBuilder.build_system_prompt().
        messages:
            Message list from PromptBuilder.build_messages(). Must not
            already contain a system message.

        Returns
        -------
        str
            The assistant message content.

        Raises
        ------
        LLMAuthError
            If the API key is missing or rejected (HTTP 401).
        LLMRateLimitError
            If the API returns a rate-limit response (HTTP 429).
        LLMTimeoutError
            If the request times out.
        LLMClientError
            For all other API-level failures.
        ValueError
            If the LLM returns an empty content field (content-policy
            refusal or malformed request).
        """
        choice = self.call_structured(system_prompt, messages)
        content = choice.message.content

        if content is None:
            raise ValueError(
                f"LLM returned an empty response "
                f"(finish_reason='{choice.finish_reason}'). "
                "This may indicate a content-policy refusal or a malformed request."
            )

        return content

    def call_structured(
        self,
        system_prompt: str,
        messages: list[ChatCompletionMessageParam],
    ) -> Choice:
        """Send messages and return the raw ChatCompletionChoice object.

        Use this when you need access to finish_reason, token usage,
        or other metadata beyond the reply string. call() delegates to
        this method internally.

        Raises the same LLMClientError subclasses as call().
        """
        system_msg: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": system_prompt,
        }
        full_messages: list[ChatCompletionMessageParam] = [system_msg, *messages]

        try:
            response = self._client.chat.completions.create(
                model=self._config.model_name,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                messages=full_messages,
            )
        except RateLimitError as exc:
            raise LLMRateLimitError(
                f"Rate limit reached for model '{self._config.model_name}': {exc}"
            ) from exc
        except APITimeoutError as exc:
            raise LLMTimeoutError(
                f"Request timed out for model '{self._config.model_name}': {exc}"
            ) from exc
        except APIConnectionError as exc:
            raise LLMClientError(
                f"Connection error while calling '{self._config.model_name}': {exc}"
            ) from exc
        except APIStatusError as exc:
            if exc.status_code == 401:
                raise LLMAuthError(
                    f"Authentication failed for model '{self._config.model_name}'. "
                    f"Check the value of '{self._config.api_key_env_var}'."
                ) from exc
            raise LLMClientError(
                f"API error (status={exc.status_code}) for model "
                f"'{self._config.model_name}': {exc}"
            ) from exc
        except APIError as exc:
            raise LLMClientError(
                f"API error for model '{self._config.model_name}': {exc}"
            ) from exc

        if not response.choices:
            raise LLMClientError(
                f"Model '{self._config.model_name}' returned an empty choices list. "
                "This may indicate a content-policy refusal or a provider-side error."
            )

        return response.choices[0]
