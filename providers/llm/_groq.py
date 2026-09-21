"""Groq adapter."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from backend.models import ProviderTokenBudgetExceededError
from backend.utils import get_logger, estimate_tokens

from backend.providers.llm._types import (
    LLMMessage,
    LLMRequestRequirements,
    LLMResponse,
)
from backend.providers.llm._openai import OpenAICompatibleLLM

logger = get_logger(__name__)


class GroqLLM(OpenAICompatibleLLM):
    """Text and vision generation through GroqCloud."""

    name = "groq"
    max_tokens_field = "max_completion_tokens"

    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = "openai/gpt-oss-20b"
    DEFAULT_VISION_MODEL = "qwen/qwen3.6-27b"
    VISION_MODELS = frozenset({DEFAULT_VISION_MODEL})

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = "",
        temperature: float = 0.1,
        max_output_tokens: int = 1400,
        timeout_s: float = 90.0,
        vision_model: str = DEFAULT_VISION_MODEL,
        retry_attempts: int = 2,
        supports_images_override: Optional[bool] = None,
        max_rate_limit_wait_seconds: float = 20.0,
        tpm_limit: int = 8000,
        estimated_image_tokens: int = 2048,
        focused_vision_max_output_tokens: int = 1024,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model or self.DEFAULT_MODEL,
            base_url=base_url or self.DEFAULT_BASE_URL,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
            vision_model=vision_model or self.DEFAULT_VISION_MODEL,
            retry_attempts=retry_attempts,
        )
        self._supports_images_override = supports_images_override
        self.retry_max_delay = max(0.0, max_rate_limit_wait_seconds)
        self.skip_if_retry_after_exceeds_max = True
        self.tpm_limit = max(0, tpm_limit)
        self.estimated_image_tokens = max(0, estimated_image_tokens)
        self.focused_vision_max_output_tokens = max(
            128, focused_vision_max_output_tokens
        )
        self.supports_vision = self.supports_images()

    def _groq_model_supports_images(model: str) -> bool:
        return (model or "").strip().lower() in GroqLLM.VISION_MODELS

    def supports_images(self, model: Optional[str] = None) -> bool:
        if self._supports_images_override is not None:
            return self._supports_images_override
        return (model or "").strip().lower() in self.VISION_MODELS

    def _provider_payload(
        self,
        model: str,
        *,
        json_mode: bool,
        requirements: Optional[LLMRequestRequirements],
    ) -> Dict[str, Any]:
        normalized = (model or "").strip().lower()
        if normalized.startswith("qwen/"):
            return {"reasoning_format": "hidden"}
        if normalized.startswith("openai/gpt-oss-"):
            return {"include_reasoning": False}
        return {}

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        model: Optional[str] = None,
        json_mode: bool = False,
        requirements: Optional[LLMRequestRequirements] = None,
    ) -> LLMResponse:
        requested = max_output_tokens or self.max_output_tokens
        operation = requirements.operation if requirements else "unspecified"
        image_count = sum(len(message.images) for message in messages)
        input_estimate = estimate_tokens(system or "") + sum(
            estimate_tokens(message.text)
            + len(message.images) * self.estimated_image_tokens
            for message in messages
        )
        effective = requested
        if image_count and operation in {
            "final_answer",
            "final_answer_continuation",
        }:
            effective = min(effective, self.focused_vision_max_output_tokens)
        if self.tpm_limit:
            safe_limit = max(256, int(self.tpm_limit * 0.90))
            available = safe_limit - input_estimate - 128
            if available < 128:
                raise ProviderTokenBudgetExceededError(
                    "Groq request skipped before HTTP: estimated input exceeds TPM window",
                    provider=self.name,
                )
            effective = min(effective, max(128, available))
            if effective < requested:
                if requested > self.max_output_tokens:
                    raise ProviderTokenBudgetExceededError(
                        "Groq request skipped before HTTP: exhaustive output budget "
                        "does not fit the estimated TPM window",
                        provider=self.name,
                    )
                logger.info(
                    "LLM operation=%s provider=groq token_budget requested=%d "
                    "input_estimate=%d effective=%d",
                    operation,
                    requested,
                    input_estimate,
                    effective,
                )
        try:
            response = super().complete(
                messages,
                system=system,
                temperature=temperature,
                max_output_tokens=effective,
                model=model,
                json_mode=json_mode,
                requirements=requirements,
            )
        except ProviderTokenBudgetExceededError:
            if not image_count or operation not in {
                "final_answer",
                "final_answer_continuation",
            } or effective <= 512:
                raise
            reduced = max(256, min(512, effective // 2))
            logger.warning(
                "LLM operation=%s provider=groq retry=token_budget_reduction "
                "previous_output=%d reduced_output=%d",
                operation,
                effective,
                reduced,
            )
            response = super().complete(
                messages,
                system=system,
                temperature=temperature,
                max_output_tokens=reduced,
                model=model,
                json_mode=json_mode,
                requirements=requirements,
            )
            effective = reduced

        response.diagnostics.setdefault("estimated_input_tokens", input_estimate)
        response.diagnostics.setdefault("requested_output_tokens", requested)
        response.diagnostics.setdefault("effective_output_tokens", effective)
        response.diagnostics.setdefault(
            "estimated_total_tokens", input_estimate + effective
        )
        response.diagnostics.setdefault("selected_visuals", image_count)
        return response
