"""OpenRouter adapter."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from backend.models import (
    ProviderBadRequestError,
    ProviderCapabilityError,
    ProviderPaymentRequiredError,
    ProviderUnavailableError,
)
from backend.utils import get_logger

from backend.providers.llm._types import (
    LLMMessage,
    LLMRequestRequirements,
    LLMResponse,
)
from backend.providers.llm._openai import OpenAICompatibleLLM

logger = get_logger(__name__)


class OpenRouterLLM(OpenAICompatibleLLM):
    name = "openrouter"

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "openrouter/free"
    FREE_ROUTER_MODEL = "openrouter/free"

    VISION_MODEL_MARKERS: tuple[str, ...] = (
        "gemini", "gpt-4o", "gpt-4.1", "gpt-4-turbo", "gpt-5",
        "o3", "o4-mini", "claude-3", "claude-sonnet", "claude-opus", "claude-haiku",
        "pixtral", "llama-3.2-11b", "llama-3.2-90b", "llama-4",
        "qwen-vl", "qwen2-vl", "qwen2.5-vl", "internvl", "molmo",
        "grok-2-vision", "grok-4", "mistral-medium-3",
        "-vision", "vision-", "multimodal",
    )
    TEXT_ONLY_MARKERS: tuple[str, ...] = (
        "gemma-2", "deepseek-r1-distill", "text-embedding",
    )

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = "",
        temperature: float = 0.1,
        max_output_tokens: int = 1400,
        timeout_s: float = 90.0,
        vision_model: str = "",
        retry_attempts: int = 2,
        supports_images_override: Optional[bool] = None,
        app_url: str = "https://github.com/",
        app_title: str = "OmniRAG",
        free_fallback: bool = True,
    ):
        super().__init__(
            api_key=api_key,
            model=model or self.DEFAULT_MODEL,
            base_url=base_url or self.DEFAULT_BASE_URL,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
            vision_model=vision_model,
            retry_attempts=retry_attempts,
        )
        self._supports_images_override = supports_images_override
        self.app_url = app_url
        self.app_title = app_title
        self.free_fallback = free_fallback
        self.supports_vision = True

    def _openrouter_model_supports_images(model: str) -> bool:
        slug = (model or "").lower()
        if not slug:
            return False
        if slug == OpenRouterLLM.FREE_ROUTER_MODEL:
            return True
        if any(marker in slug for marker in OpenRouterLLM.TEXT_ONLY_MARKERS):
            return False
        return any(marker in slug for marker in OpenRouterLLM.VISION_MODEL_MARKERS)

    def supports_images(self, model: Optional[str] = None) -> bool:
        if self._supports_images_override is not None:
            return self._supports_images_override
        slug = (model or self.vision_model or self.model).lower()
        if not slug:
            return False
        if slug == self.FREE_ROUTER_MODEL:
            return True
        if any(marker in slug for marker in self.TEXT_ONLY_MARKERS):
            return False
        return any(marker in slug for marker in self.VISION_MODEL_MARKERS)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.app_url,
            "X-Title": self.app_title,
        }

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
        has_images = any(m.has_images for m in messages)
        target_model = model or (self.vision_model if has_images else self.model)
        requirements = requirements or LLMRequestRequirements(
            requires_images=has_images,
            requires_structured_output=json_mode,
        )

        if has_images and not self.supports_images(target_model):
            raise ProviderCapabilityError(
                f"OpenRouter model '{target_model}' does not accept image input",
                provider="openrouter",
                capability="images",
                user_message=(
                    f"The configured OpenRouter model `{target_model}` cannot read images, "
                    "so the visual evidence for this request could not be analysed. "
                    "Set `OPENROUTER_MODEL` to a vision-capable model."
                ),
            )

        try:
            return self._complete_route(
                messages,
                system=system,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                model=target_model,
                json_mode=json_mode,
                requirements=requirements,
            )
        except ProviderPaymentRequiredError as exc:
            if target_model != self.FREE_ROUTER_MODEL and self.free_fallback:
                logger.warning(
                    "OpenRouter model=%s requires credits; retrying once via %s",
                    target_model,
                    self.FREE_ROUTER_MODEL,
                )
                try:
                    response = self._complete_route(
                        messages,
                        system=system,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        model=self.FREE_ROUTER_MODEL,
                        json_mode=json_mode,
                        requirements=requirements,
                    )
                    response.fallback_used = True
                    response.diagnostics["openrouter_free_fallback"] = True
                    return response
                except ProviderPaymentRequiredError as free_exc:
                    free_exc.user_message = self._free_unavailable_message(requirements)
                    raise
            exc.user_message = self._free_unavailable_message(requirements)
            raise

    def _complete_route(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: Optional[str],
        temperature: Optional[float],
        max_output_tokens: Optional[int],
        model: str,
        json_mode: bool,
        requirements: LLMRequestRequirements,
    ) -> LLMResponse:
        try:
            return super().complete(
                messages,
                system=system,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                model=model,
                json_mode=json_mode,
                requirements=requirements,
            )
        except (ProviderUnavailableError, ProviderBadRequestError) as exc:
            if model == self.FREE_ROUTER_MODEL and requirements.requires_images:
                raise ProviderCapabilityError(
                    "No compatible free OpenRouter multimodal route was available",
                    provider=self.name,
                    capability="images",
                    user_message=self._free_unavailable_message(requirements),
                ) from exc
            raise

    @staticmethod
    def _free_unavailable_message(requirements: LLMRequestRequirements) -> str:
        if requirements.requires_images:
            return "No free multimodal OpenRouter model is currently available for this request."
        return "No compatible free OpenRouter model is currently available. Please retry shortly."
