"""OpenAI / OpenAI-compatible chat-completions adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from backend.models import (
    LLMError,
    ProviderPolicyError,
    Role,
)
from backend.utils import to_data_url, retry_call

from backend.providers.llm._types import (
    LLMMessage,
    LLMRequestRequirements,
    LLMResponse,
    _TEXT_ONLY_HINTS,
)
from backend.providers.llm._http import post_json
from backend.providers.llm._base import (
    BaseLLMProvider,
    _visible_message_text,
    _without_leading_thinking,
    raise_gateway_error,
)


class OpenAICompatibleLLM(BaseLLMProvider):
    """OpenAI / OpenAI-compatible chat-completions adapter."""

    name = "openai_compatible"
    supports_vision = True
    max_tokens_field = "max_tokens"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "",
        temperature: float = 0.1,
        max_output_tokens: int = 1400,
        timeout_s: float = 90.0,
        vision_model: str = "",
        retry_attempts: int = 2,
    ):
        super().__init__(model=model, temperature=temperature, max_output_tokens=max_output_tokens)
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.timeout_s = timeout_s
        self.vision_model = vision_model or model
        self.retry_attempts = max(1, retry_attempts)
        self.supports_vision = not any(h in model.lower() for h in _TEXT_ONLY_HINTS)

    def model_for_request(
        self, messages: Sequence[LLMMessage], model: Optional[str] = None
    ) -> str:
        return model or (
            self.vision_model if any(message.has_images for message in messages) else self.model
        )

    def supports_images(self, model: Optional[str] = None) -> bool:
        target = (model or self.vision_model or self.model).lower()
        return not any(hint in target for hint in _TEXT_ONLY_HINTS)

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
        target_model = self.model_for_request(messages, model)

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": self._build_messages(messages, system),
            "temperature": self.temperature if temperature is None else temperature,
            self.max_tokens_field: max_output_tokens or self.max_output_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
            if self.name == "openrouter":
                payload["provider"] = {"require_parameters": True}
        payload.update(
            self._provider_payload(
                target_model,
                json_mode=json_mode,
                requirements=requirements,
            )
        )

        headers = self._headers()
        diagnostics: Dict[str, Any] = {}

        def _call() -> Dict[str, Any]:
            return post_json(
                f"{self.base_url}/chat/completions",
                payload,
                headers=headers,
                timeout_s=self.timeout_s,
                provider=self.name,
                diagnostics=diagnostics,
            )

        body = retry_call(
            _call,
            attempts=self.retry_attempts,
            operation=f"{self.name}/chat-completions ({target_model})",
            max_delay=getattr(self, "retry_max_delay", 2.0),
            skip_if_retry_after_exceeds_max=getattr(
                self, "skip_if_retry_after_exceeds_max", False
            ),
        )
        response = self._parse(body, target_model)
        response.diagnostics.update(diagnostics)
        response.diagnostics.setdefault("provider_raw_chars", len(response.text))
        response.diagnostics.setdefault("parsed_chars", len(response.text))
        return response

    def _provider_payload(
        self,
        model: str,
        *,
        json_mode: bool,
        requirements: Optional[LLMRequestRequirements],
    ) -> Dict[str, Any]:
        return {}

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_messages(
        self, messages: Sequence[LLMMessage], system: Optional[str]
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})

        for message in messages:
            role = "assistant" if message.role == Role.ASSISTANT else (
                "system" if message.role == Role.SYSTEM else "user"
            )
            if not message.images:
                out.append({"role": role, "content": message.text})
                continue

            parts: List[Dict[str, Any]] = []
            if message.text:
                parts.append({"type": "text", "text": message.text})
            for image in message.images:
                if image.label:
                    parts.append({"type": "text", "text": image.label})
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": to_data_url(image.data, image.media_type)},
                    }
                )
            out.append({"role": role, "content": parts})
        return out

    def _parse(self, body: Dict[str, Any], model: str) -> LLMResponse:
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            raise_gateway_error(error, provider=self.name)

        try:
            choice = (body.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            raw_text, structured_reasoning_chars = _visible_message_text(message)
            text, tagged_reasoning_chars = _without_leading_thinking(raw_text)
            reasoning_chars = structured_reasoning_chars + tagged_reasoning_chars
            finish_reason = choice.get("finish_reason") or ""
        except Exception as exc:
            raise LLMError(
                f"Unexpected chat/completions payload: {str(body)[:300]}",
                provider=self.name,
            ) from exc

        if finish_reason == "content_filter":
            raise ProviderPolicyError(
                f"{self.name} blocked the request (content_filter)",
                provider=self.name,
                reason="content_filter",
            )

        if not text:
            raise LLMError(
                "The model returned an empty completion",
                provider=self.name,
                user_message="The model returned an empty answer. Try rephrasing your question.",
            )

        return LLMResponse(
            text=text,
            model=body.get("model", model),
            finish_reason=finish_reason,
            usage=body.get("usage") or {},
            provider=self.name,
            diagnostics={
                "provider_raw_chars": len(raw_text),
                "parsed_chars": len(text),
                "reasoning_chars": reasoning_chars,
                "reasoning_suppressed": bool(reasoning_chars),
            },
        )
