"""Consolidated provider tests: LLM fallback, Groq transport, and embeddings."""

from __future__ import annotations

import httpx
import pytest

from backend.config import build_settings, reset_settings_cache
from backend.models import (
    AllProvidersFailedError,
    Chunk,
    EmbeddingError,
    FileType,
    MissingCredentialError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderCapabilityError,
    ProviderPaymentRequiredError,
    ProviderPolicyError,
    ProviderTokenBudgetExceededError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from backend.providers.llm import (
    BaseLLMProvider,
    FallbackLLMProvider,
    GeminiLLM,
    GroqLLM,
    ImagePart,
    LLMMessage,
    LLMRequestRequirements,
    LLMResponse,
    OpenRouterLLM,
    build_llm_provider,
    classify,
    get_llm_provider,
    llm_session,
    model_supports_images,
    should_failover,
)
from backend.providers import FailureClass
from backend.providers.embeddings import (
    BaseEmbeddingProvider,
    DEFAULT_DIMENSIONS,
    GeminiEmbeddings,
    HashingEmbeddings,
    ResilientEmbeddings,
)
from backend.rag.vector_store import InMemoryVectorStore

GEMINI_DEFAULT_MODEL = GeminiLLM.DEFAULT_MODEL
OPENROUTER_DEFAULT_MODEL = OpenRouterLLM.DEFAULT_MODEL
DEFAULT_BASE_URL = GroqLLM.DEFAULT_BASE_URL
DEFAULT_MODEL = GroqLLM.DEFAULT_MODEL
DEFAULT_VISION_MODEL = GroqLLM.DEFAULT_VISION_MODEL
EMBEDDING_DEFAULT_BASE_URL = GeminiEmbeddings.DEFAULT_BASE_URL
EMBEDDING_DEFAULT_MODEL = GeminiEmbeddings.DEFAULT_MODEL


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class ScriptedProvider(BaseLLMProvider):
    """Provider that raises a queued exception or returns a canned answer."""

    def __init__(self, name: str, outcomes, *, images: bool = True, model: str = "m"):
        super().__init__(model=model)
        self.name = name
        self._images = images
        self.outcomes = list(outcomes)
        self.call_count = 0
        self.last_messages = None

    def supports_images(self, model=None) -> bool:
        return self._images

    def complete(self, messages, *, system=None, temperature=None,
                 max_output_tokens=None, model=None, json_mode=False,
                 requirements=None):
        self.call_count += 1
        self.last_messages = messages
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        if isinstance(outcome, BaseException):
            raise outcome
        return LLMResponse(text=str(outcome), model=self.model, provider=self.name)


def message(text: str = "hi", *, with_image: bool = False) -> list[LLMMessage]:
    images = [ImagePart(data=b"\x89PNG-fake", media_type="image/png")] if with_image else []
    return [LLMMessage(text=text, images=images)]


def configure(monkeypatch, **env):
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    return build_settings()


class FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = "", headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text or str(self._payload)
        self.headers = headers or {}

    def json(self):
        return self._payload


# --------------------------------------------------------------------------- #
# Error classification
# --------------------------------------------------------------------------- #
class TestErrorClassification:
    @pytest.mark.parametrize(
        "exc",
        [
            RateLimitError("429", provider="gemini"),
            RateLimitError("quota", provider="gemini", quota_exhausted=True),
            ProviderTimeoutError("timeout", provider="gemini"),
            ProviderUnavailableError("503", provider="gemini"),
        ],
    )
    def test_recoverable_errors_fail_over(self, exc):
        assert classify(exc) is FailureClass.RECOVERABLE
        assert should_failover(exc) is True

    @pytest.mark.parametrize(
        "exc,expected",
        [
            (ProviderAuthError("401", provider="gemini"), FailureClass.AUTH),
            (ProviderBadRequestError("400", provider="gemini"), FailureClass.BAD_REQUEST),
            (ProviderPolicyError("safety", provider="gemini"), FailureClass.POLICY),
            (ProviderCapabilityError("no images", provider="openrouter"), FailureClass.CAPABILITY),
            (ValueError("bug in our code"), FailureClass.BUG),
            (TypeError("bug"), FailureClass.BUG),
        ],
    )
    def test_non_recoverable_errors_do_not_fail_over(self, exc, expected):
        assert classify(exc) is expected
        assert should_failover(exc) is False

    def test_quota_exhaustion_is_not_retried_on_the_same_provider(self):
        from backend.providers.llm import should_retry_same_provider

        transient = RateLimitError("slow down", provider="gemini")
        exhausted = RateLimitError("quota", provider="gemini", quota_exhausted=True)

        assert should_retry_same_provider(transient) is True
        # Waiting will not refill a quota; switch provider instead of hanging.
        assert should_retry_same_provider(exhausted) is False


# --------------------------------------------------------------------------- #
# Router failover
# --------------------------------------------------------------------------- #
class TestRouterFailover:
    def test_gemini_success_never_touches_the_fallback(self):
        gemini = ScriptedProvider("gemini", outcomes=["primary answer"])
        openrouter = ScriptedProvider("openrouter", outcomes=["fallback answer"])
        router = FallbackLLMProvider([gemini, openrouter])

        response = router.complete(message())

        assert response.text == "primary answer"
        assert response.provider == "gemini"
        assert response.fallback_used is False
        assert openrouter.call_count == 0
        assert router.stats.failovers == 0

    @pytest.mark.parametrize(
        "failure",
        [
            RateLimitError("429 rate limited", provider="gemini"),
            RateLimitError("RESOURCE_EXHAUSTED", provider="gemini", quota_exhausted=True),
            ProviderTimeoutError("deadline exceeded", provider="gemini"),
            ProviderUnavailableError("503 model overloaded", provider="gemini"),
            ProviderUnavailableError("connection reset", provider="gemini"),
        ],
        ids=["rate_limit", "quota", "timeout", "server_error", "network"],
    )
    def test_recoverable_gemini_failure_falls_back_to_openrouter(self, failure):
        gemini = ScriptedProvider("gemini", outcomes=[failure])
        openrouter = ScriptedProvider("openrouter", outcomes=["fallback answer"])
        router = FallbackLLMProvider([gemini, openrouter])

        response = router.complete(message())

        assert response.text == "fallback answer"
        assert response.provider == "openrouter"
        assert response.fallback_used is True
        assert gemini.call_count == 1
        assert openrouter.call_count == 1
        assert router.stats.failovers == 1
        assert len(response.attempts) == 2

    def test_invalid_api_key_does_not_fall_back(self):
        gemini = ScriptedProvider(
            "gemini", outcomes=[ProviderAuthError("401 invalid key", provider="gemini")]
        )
        openrouter = ScriptedProvider("openrouter", outcomes=["fallback answer"])
        router = FallbackLLMProvider([gemini, openrouter])

        with pytest.raises(ProviderAuthError):
            router.complete(message())

        # A second vendor cannot fix a bad key — it must not be charged for it.
        assert openrouter.call_count == 0

    def test_policy_refusal_does_not_fall_back(self):
        gemini = ScriptedProvider(
            "gemini", outcomes=[ProviderPolicyError("blocked", provider="gemini", reason="SAFETY")]
        )
        openrouter = ScriptedProvider("openrouter", outcomes=["fallback answer"])
        router = FallbackLLMProvider([gemini, openrouter])

        with pytest.raises(ProviderPolicyError):
            router.complete(message())
        assert openrouter.call_count == 0

    def test_malformed_request_does_not_fall_back(self):
        gemini = ScriptedProvider(
            "gemini", outcomes=[ProviderBadRequestError("400 bad model", provider="gemini")]
        )
        openrouter = ScriptedProvider("openrouter", outcomes=["fallback"])
        router = FallbackLLMProvider([gemini, openrouter])

        with pytest.raises(ProviderBadRequestError):
            router.complete(message())
        assert openrouter.call_count == 0

    def test_programming_bug_propagates_untouched(self):
        gemini = ScriptedProvider("gemini", outcomes=[ValueError("our bug")])
        openrouter = ScriptedProvider("openrouter", outcomes=["fallback"])
        router = FallbackLLMProvider([gemini, openrouter])

        with pytest.raises(ValueError):
            router.complete(message())
        assert openrouter.call_count == 0

    def test_fallback_failure_reports_every_provider(self):
        gemini = ScriptedProvider("gemini", outcomes=[RateLimitError("429", provider="gemini")])
        openrouter = ScriptedProvider(
            "openrouter", outcomes=[ProviderUnavailableError("503", provider="openrouter")]
        )
        router = FallbackLLMProvider([gemini, openrouter])

        with pytest.raises(AllProvidersFailedError) as excinfo:
            router.complete(message())

        failures = dict(excinfo.value.failures)
        assert set(failures) == {"gemini", "openrouter"}
        assert excinfo.value.user_message

    def test_nonrecoverable_fallback_failure_preserves_primary_failure(self):
        primary_error = ProviderUnavailableError("503", provider="gemini")
        fallback_error = ProviderBadRequestError("404 model", provider="openrouter")
        router = FallbackLLMProvider([
            ScriptedProvider("gemini", outcomes=[primary_error]),
            ScriptedProvider("openrouter", outcomes=[fallback_error]),
        ])

        with pytest.raises(AllProvidersFailedError) as excinfo:
            router.complete(message())

        assert excinfo.value.failures == [
            ("gemini", primary_error),
            ("openrouter", fallback_error),
        ]
        assert "gemini:" in excinfo.value.user_message
        assert "openrouter:" in excinfo.value.user_message

    def test_fallback_can_be_disabled(self):
        gemini = ScriptedProvider("gemini", outcomes=[RateLimitError("429", provider="gemini")])
        openrouter = ScriptedProvider("openrouter", outcomes=["fallback"])
        router = FallbackLLMProvider([gemini, openrouter], enable_fallback=False)

        with pytest.raises(AllProvidersFailedError):
            router.complete(message())
        assert openrouter.call_count == 0

    def test_rate_limit_cooldown_is_per_session_and_expires(self):
        now = [100.0]
        gemini = ScriptedProvider(
            "gemini", outcomes=[RateLimitError("429", provider="gemini"), "recovered"]
        )
        fallback = ScriptedProvider("openrouter", outcomes=["s1 fallback", "s1 fallback 2"])
        router = FallbackLLMProvider(
            [gemini, fallback], rate_limit_cooldown_seconds=60, clock=lambda: now[0]
        )

        with llm_session("s1"):
            assert router.complete(message()).provider == "openrouter"
            assert router.complete(message()).provider == "openrouter"
        assert gemini.call_count == 1

        with llm_session("s2"):
            assert router.complete(message()).provider == "gemini"
        assert gemini.call_count == 2

        now[0] += 61
        gemini.outcomes.append("after cooldown")
        with llm_session("s1"):
            assert router.complete(message()).provider == "gemini"


# --------------------------------------------------------------------------- #
# Multimodal capability
# --------------------------------------------------------------------------- #
class TestMultimodalCapability:
    def test_text_only_fallback_is_skipped_for_image_requests(self):
        gemini = ScriptedProvider("gemini", outcomes=[RateLimitError("429", provider="gemini")])
        text_only = ScriptedProvider("openrouter", outcomes=["fallback"], images=False)
        router = FallbackLLMProvider([gemini, text_only])

        with pytest.raises(AllProvidersFailedError):
            router.complete(message(with_image=True))

        # Never sent: dropping the image silently would produce a confident
        # answer about a picture the model never saw.
        assert text_only.call_count == 0

    def test_text_only_fallback_still_serves_text_requests(self):
        gemini = ScriptedProvider("gemini", outcomes=[RateLimitError("429", provider="gemini")])
        text_only = ScriptedProvider("openrouter", outcomes=["fallback"], images=False)
        router = FallbackLLMProvider([gemini, text_only])

        response = router.complete(message(with_image=False))
        assert response.text == "fallback"
        assert text_only.call_count == 1

    def test_capability_error_when_no_provider_can_see_images(self):
        text_only = ScriptedProvider("openrouter", outcomes=["never"], images=False)
        router = FallbackLLMProvider([text_only])

        with pytest.raises(ProviderCapabilityError) as excinfo:
            router.complete(message(with_image=True))

        assert "cannot read images" in excinfo.value.user_message
        assert text_only.call_count == 0

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("google/gemini-3.6-flash", True),
            ("openrouter/free", True),
            ("openai/gpt-4o-mini", True),
            ("anthropic/claude-sonnet-4.5", True),
            ("mistralai/pixtral-12b", True),
            ("qwen/qwen2.5-vl-72b-instruct", True),
            ("meta-llama/llama-3.1-8b-instruct", False),
            ("deepseek/deepseek-r1-distill-llama-70b", False),
            ("google/gemma-2-27b-it", False),
            ("", False),
        ],
    )
    def test_openrouter_model_capability_detection(self, model, expected):
        assert model_supports_images(model) is expected

    def test_openrouter_raises_before_calling_a_text_only_model(self):
        provider = OpenRouterLLM(
            api_key="test-key", model="meta-llama/llama-3.1-8b-instruct"
        )
        with pytest.raises(ProviderCapabilityError) as excinfo:
            provider.complete(message(with_image=True))
        assert "OPENROUTER_MODEL" in excinfo.value.user_message

    def test_capability_override_is_honoured(self):
        provider = OpenRouterLLM(
            api_key="k", model="some/unknown-model", supports_images_override=True
        )
        assert provider.supports_images() is True


# --------------------------------------------------------------------------- #
# Configuration combinations (fallback)
# --------------------------------------------------------------------------- #
class TestConfigurationCombinations:
    def test_both_configured_gives_gemini_primary_openrouter_fallback(self, monkeypatch):
        settings = configure(
            monkeypatch, GEMINI_API_KEY="g-key", OPENROUTER_API_KEY="or-key"
        )
        router = build_llm_provider(settings.llm)

        assert [p.name for p in router.chain] == ["gemini", "openrouter"]
        assert router.enable_fallback is True
        assert settings.llm.fallback_active is True

    def test_only_gemini_configured_works_without_fallback(self, monkeypatch):
        settings = configure(monkeypatch, GEMINI_API_KEY="g-key")
        router = build_llm_provider(settings.llm)

        assert [p.name for p in router.chain] == ["gemini"]
        assert router.enable_fallback is False
        assert settings.llm.is_configured is True

    def test_only_openrouter_configured_becomes_the_active_provider(self, monkeypatch):
        settings = configure(monkeypatch, OPENROUTER_API_KEY="or-key")
        router = build_llm_provider(settings.llm)

        assert [p.name for p in router.chain] == ["openrouter"]
        assert settings.llm.provider == "openrouter"
        assert settings.llm.is_configured is True

    def test_neither_configured_reports_a_clear_error(self, monkeypatch):
        settings = configure(monkeypatch)

        assert settings.llm.is_configured is False
        issues = settings.validation_issues()
        assert issues and "GEMINI_API_KEY" in issues[0]

        with pytest.raises(MissingCredentialError):
            build_llm_provider(settings.llm)

    def test_fallback_can_be_disabled_by_configuration(self, monkeypatch):
        settings = configure(
            monkeypatch,
            GEMINI_API_KEY="g-key",
            OPENROUTER_API_KEY="or-key",
            ENABLE_PROVIDER_FALLBACK="false",
        )
        router = build_llm_provider(settings.llm)

        assert router.enable_fallback is False
        assert len(router.chain) == 1

    def test_models_are_configurable(self, monkeypatch):
        settings = configure(
            monkeypatch,
            GEMINI_API_KEY="g",
            GEMINI_MODEL="gemini-2.5-pro",
            OPENROUTER_API_KEY="o",
            OPENROUTER_MODEL="openai/gpt-4o",
        )
        chain = {e.provider: e.model for e in settings.llm.configured_endpoints}
        assert chain["gemini"] == "gemini-2.5-pro"
        assert chain["openrouter"] == "openai/gpt-4o"

    def test_current_default_model_identifiers(self, monkeypatch):
        settings = configure(
            monkeypatch,
            GEMINI_API_KEY="g",
            OPENROUTER_API_KEY="o",
        )
        chain = {e.provider: e.model for e in settings.llm.configured_endpoints}

        assert GEMINI_DEFAULT_MODEL == "gemini-3.6-flash"
        assert OPENROUTER_DEFAULT_MODEL == "openrouter/free"
        assert chain == {
            "gemini": "gemini-3.6-flash",
            "openrouter": "openrouter/free",
        }

    def test_no_api_key_is_ever_hardcoded(self, monkeypatch):
        settings = configure(monkeypatch)
        for endpoint in settings.llm.endpoints:
            assert endpoint.api_key == ""


# --------------------------------------------------------------------------- #
# HTTP-level classification (adapters, still no network)
# --------------------------------------------------------------------------- #
class TestHTTPClassification:
    @pytest.fixture(autouse=True)
    def _patch_client(self, monkeypatch):
        self.responses = []
        self.requests = []

        class FakeClient:
            is_closed = False

            def post(_self, url, json=None, headers=None):
                self.requests.append({"url": url, "json": json})
                if not self.responses:
                    raise AssertionError("unexpected extra HTTP call")
                item = self.responses.pop(0)
                if isinstance(item, BaseException):
                    raise item
                return item

        monkeypatch.setattr(
            "backend.providers.llm._http.get_client", lambda timeout_s=60.0: FakeClient()
        )

    def test_429_becomes_a_rate_limit_error(self):
        self.responses = [FakeResponse(429, text="Too many requests")]
        provider = GeminiLLM(api_key="k", model="gemini-3.6-flash", retry_attempts=1)
        with pytest.raises(RateLimitError) as excinfo:
            provider.complete(message())
        assert excinfo.value.quota_exhausted is False

    def test_429_with_resource_exhausted_marks_quota(self):
        self.responses = [FakeResponse(429, text='{"error":{"status":"RESOURCE_EXHAUSTED"}}')]
        provider = GeminiLLM(api_key="k", model="gemini-3.6-flash", retry_attempts=1)
        with pytest.raises(RateLimitError) as excinfo:
            provider.complete(message())
        assert excinfo.value.quota_exhausted is True

    def test_401_becomes_an_auth_error(self):
        self.responses = [FakeResponse(401, text="API key not valid")]
        provider = GeminiLLM(api_key="bad", model="gemini-3.6-flash", retry_attempts=1)
        with pytest.raises(ProviderAuthError):
            provider.complete(message())

    def test_503_becomes_a_provider_unavailable_error(self):
        self.responses = [FakeResponse(503, text="model overloaded")]
        provider = GeminiLLM(api_key="k", model="gemini-3.6-flash", retry_attempts=1)
        with pytest.raises(ProviderUnavailableError) as excinfo:
            provider.complete(message())
        assert excinfo.value.status_code == 503
        assert excinfo.value.safe_body == "model overloaded"

    def test_timeout_becomes_a_provider_timeout_error(self):
        self.responses = [httpx.ReadTimeout("timed out")]
        provider = GeminiLLM(api_key="k", model="gemini-3.6-flash", retry_attempts=1)
        with pytest.raises(ProviderTimeoutError):
            provider.complete(message())

    def test_gemini_safety_block_becomes_a_policy_error(self):
        self.responses = [
            FakeResponse(200, {"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []})
        ]
        provider = GeminiLLM(api_key="k", model="gemini-3.6-flash", retry_attempts=1)
        with pytest.raises(ProviderPolicyError):
            provider.complete(message())

    def test_gemini_success_is_parsed(self):
        self.responses = [
            FakeResponse(
                200,
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "Hello"}, {"text": " world"}]
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 5,
                        "candidatesTokenCount": 7,
                        "totalTokenCount": 12,
                    },
                },
                headers={"content-length": "321"},
            )
        ]
        provider = GeminiLLM(api_key="k", model="gemini-3.6-flash", retry_attempts=1)
        response = provider.complete(message())

        assert response.text == "Hello world"
        assert response.provider == "gemini"
        assert response.finish_reason == "STOP"
        assert response.usage["candidatesTokenCount"] == 7
        assert response.diagnostics == {
            "http_status": 200,
            "content_length": "321",
            "response_fully_received": True,
            "json_parsed": True,
            "provider_raw_chars": 11,
            "parsed_chars": 11,
            "candidate_count": 1,
            "content_parts_count": 2,
            "prompt_token_count": 5,
            "candidates_token_count": 7,
            "total_token_count": 12,
        }
        assert self.requests[0]["url"].endswith(
            "/models/gemini-3.6-flash:generateContent"
        )
        assert "temperature" not in self.requests[0]["json"]["generationConfig"]

    def test_openrouter_content_filter_becomes_a_policy_error(self):
        self.responses = [
            FakeResponse(
                200,
                {
                    "choices": [
                        {"message": {"content": ""}, "finish_reason": "content_filter"}
                    ]
                },
            )
        ]
        provider = OpenRouterLLM(api_key="k", model="google/gemini-3.6-flash", retry_attempts=1)
        with pytest.raises(ProviderPolicyError):
            provider.complete(message())

    def test_openrouter_gateway_error_inside_200_is_classified(self):
        self.responses = [
            FakeResponse(200, {"error": {"code": 429, "message": "rate limit exceeded"}})
        ]
        provider = OpenRouterLLM(api_key="k", model="google/gemini-3.6-flash", retry_attempts=1)
        with pytest.raises(RateLimitError) as excinfo:
            provider.complete(message())
        assert excinfo.value.status_code == 429
        assert excinfo.value.safe_body == "rate limit exceeded"

    def test_openrouter_402_is_typed_payment_required(self):
        self.responses = [FakeResponse(402, text="This request requires more credits")]
        provider = OpenRouterLLM(
            api_key="k", model="openrouter/free", retry_attempts=1
        )
        with pytest.raises(ProviderPaymentRequiredError) as excinfo:
            provider.complete(message())
        assert excinfo.value.status_code == 402
        assert "free" in excinfo.value.user_message.lower()

    def test_free_multimodal_route_unavailable_is_controlled(self):
        self.responses = [FakeResponse(503, text="No endpoints found that support image input")]
        provider = OpenRouterLLM(
            api_key="k", model="openrouter/free", retry_attempts=1
        )
        with pytest.raises(ProviderCapabilityError) as excinfo:
            provider.complete(message(with_image=True))
        assert "free multimodal" in excinfo.value.user_message.lower()

    def test_paid_model_402_retries_free_once(self):
        self.responses = [
            FakeResponse(402, text="Insufficient credits"),
            FakeResponse(200, {
                "model": "qwen/qwen-vl-plus:free",
                "choices": [{"message": {"content": "free answer"}, "finish_reason": "stop"}],
            }),
        ]
        provider = OpenRouterLLM(
            api_key="k",
            model="google/gemini-3.6-flash",
            retry_attempts=1,
            free_fallback=True,
        )
        response = provider.complete(message(with_image=True))
        assert [request["json"]["model"] for request in self.requests] == [
            "google/gemini-3.6-flash", "openrouter/free"
        ]
        assert response.model == "qwen/qwen-vl-plus:free"
        assert response.fallback_used is True
        assert response.diagnostics["openrouter_free_fallback"] is True
        assert isinstance(self.requests[1]["json"]["messages"][0]["content"], list)

    def test_structured_free_route_requires_parameter_support(self):
        self.responses = [FakeResponse(200, {
            "model": "free/model",
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        })]
        provider = OpenRouterLLM(api_key="k", model="openrouter/free", retry_attempts=1)
        provider.complete(message(), json_mode=True)
        assert self.requests[0]["json"]["provider"] == {"require_parameters": True}


# --------------------------------------------------------------------------- #
# Isolation from the rest of the system
# --------------------------------------------------------------------------- #
def test_llm_failure_does_not_touch_embeddings_or_vector_store(monkeypatch, session_id):
    """A dead LLM chain must not destroy an existing index."""
    store = InMemoryVectorStore()
    chunk = Chunk(
        document_id="d1",
        session_id=session_id,
        filename="a.pdf",
        file_type=FileType.PDF,
        block_ids=["b1"],
        text="Revenue was 8,400,000 USD.",
    )
    store.upsert(session_id, [chunk], [[0.1] * 8])
    assert store.count(session_id) == 1

    gemini = ScriptedProvider("gemini", outcomes=[RateLimitError("429", provider="gemini")])
    openrouter = ScriptedProvider(
        "openrouter", outcomes=[ProviderUnavailableError("503", provider="openrouter")]
    )
    router = FallbackLLMProvider([gemini, openrouter])

    with pytest.raises(AllProvidersFailedError):
        router.complete(message())

    # The index is untouched by the LLM outage.
    assert store.count(session_id) == 1
    assert store.list_chunks(session_id)[0].text == "Revenue was 8,400,000 USD."


# --------------------------------------------------------------------------- #
# Groq configuration
# --------------------------------------------------------------------------- #
class TestGroqConfiguration:
    def test_default_chain_and_current_models(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
        monkeypatch.setenv("GROQ_API_KEY", "groq-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")

        settings = build_settings()

        assert [e.provider for e in settings.llm.configured_endpoints] == [
            "gemini",
            "groq",
            "openrouter",
        ]
        groq = next(e for e in settings.llm.endpoints if e.provider == "groq")
        assert groq.model == DEFAULT_MODEL == "openai/gpt-oss-20b"
        assert groq.vision_model == DEFAULT_VISION_MODEL == "qwen/qwen3.6-27b"

    def test_explicit_chain_skips_missing_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER_CHAIN", "gemini,groq,openrouter")
        monkeypatch.setenv("GEMINI_API_KEY", "g")
        monkeypatch.setenv("OPENROUTER_API_KEY", "o")

        settings = build_settings()

        assert [e.provider for e in settings.llm.configured_endpoints] == [
            "gemini",
            "openrouter",
        ]

    def test_groq_and_openrouter_work_without_gemini(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "q")
        monkeypatch.setenv("OPENROUTER_API_KEY", "o")

        assert [e.provider for e in build_settings().llm.configured_endpoints] == [
            "groq",
            "openrouter",
        ]

    def test_grok_alias_is_accepted(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER_CHAIN", "groq")
        monkeypatch.setenv("GROK_API_KEY", "legacy-key")

        endpoint = build_settings().llm.configured_endpoints[0]

        assert endpoint.provider == "groq"
        assert endpoint.api_key == "legacy-key"

    def test_canonical_groq_key_wins_over_alias(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER_CHAIN", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "canonical-key")
        monkeypatch.setenv("GROK_API_KEY", "legacy-key")

        assert build_settings().llm.configured_endpoints[0].api_key == "canonical-key"

    def test_same_length_key_change_rebuilds_cached_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER_CHAIN", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "first-key-1")
        first = get_llm_provider(build_settings())

        monkeypatch.setenv("GROQ_API_KEY", "other-key-2")
        reset_settings_cache()
        second = get_llm_provider(build_settings())

        assert second is not first
        assert second.primary.api_key == "other-key-2"

    def test_model_and_chain_changes_rebuild_streamlit_engine(self, monkeypatch):
        from frontend import ui as state

        monkeypatch.setenv("LLM_PROVIDER_CHAIN", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "key")
        monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
        reset_settings_cache()
        first = state.engine()

        monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
        reset_settings_cache()
        second = state.engine()

        assert second is not first
        assert second.settings.llm.primary_endpoint.model == "openai/gpt-oss-120b"


# --------------------------------------------------------------------------- #
# Groq transport
# --------------------------------------------------------------------------- #
@pytest.fixture
def groq_http(monkeypatch):
    calls = []
    responses = []

    class FakeClient:
        is_closed = False

        def post(self, url, json=None, headers=None):
            calls.append({"url": url, "json": json, "headers": headers})
            item = responses.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

    monkeypatch.setattr(
        "backend.providers.llm._http.get_client", lambda timeout_s=60.0: FakeClient()
    )
    return calls, responses


class TestGroqTransport:
    def test_text_response_metadata_and_official_endpoint(self, groq_http):
        calls, responses = groq_http
        responses.append(FakeResponse(
            200,
            {
                "id": "chatcmpl-safe",
                "model": DEFAULT_MODEL,
                "choices": [
                    {"message": {"content": "answer"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
            headers={"content-length": "180"},
        ))
        provider = GroqLLM(api_key="test-key", retry_attempts=1)

        response = provider.complete(message(), max_output_tokens=321)

        assert calls[0]["url"] == f"{DEFAULT_BASE_URL}/chat/completions"
        assert calls[0]["json"]["model"] == DEFAULT_MODEL
        assert calls[0]["json"]["max_completion_tokens"] == 321
        assert "max_tokens" not in calls[0]["json"]
        assert calls[0]["json"]["include_reasoning"] is False
        assert response.text == "answer"
        assert response.model == DEFAULT_MODEL
        assert response.finish_reason == "stop"
        assert response.usage["completion_tokens"] == 2
        assert response.provider == "groq"
        assert response.diagnostics["http_status"] == 200

    def test_multimodal_request_uses_vision_model_and_preserves_mime(self, groq_http):
        calls, responses = groq_http
        responses.append(FakeResponse(200, {
            "model": DEFAULT_VISION_MODEL,
            "choices": [{"message": {"content": "visual answer"}, "finish_reason": "stop"}],
        }))
        provider = GroqLLM(api_key="test-key", retry_attempts=1)

        response = provider.complete(message(with_image=True))

        payload = calls[0]["json"]
        assert payload["model"] == DEFAULT_VISION_MODEL
        assert payload["reasoning_format"] == "hidden"
        parts = payload["messages"][0]["content"]
        image = next(part for part in parts if part["type"] == "image_url")
        assert image["image_url"]["url"].startswith("data:image/png;base64,")
        assert response.text == "visual answer"

    def test_qwen_separate_reasoning_field_never_enters_answer(self, groq_http):
        calls, responses = groq_http
        final_answer = "بناءً على الصورة المرفقة للصفحة 3 [1]، يوجد رسمان توضيحيان."
        reasoning = (
            "1. Analyze the Source Image:\n"
            "2. Draft the Explanation (Internal Monologue/Drafting):\n"
            "3. Refine the Output (Arabic):\n"
            "4. Final Polish\n"
            "5. Check against constraints:\n"
            "6. Final Output Generation:"
        )
        responses.append(
            FakeResponse(
                200,
                {
                    "model": DEFAULT_VISION_MODEL,
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "reasoning": reasoning,
                                "content": final_answer,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )
        provider = GroqLLM(
            api_key="test-key",
            model=DEFAULT_VISION_MODEL,
            retry_attempts=1,
        )

        response = provider.complete(message())

        assert calls[0]["json"]["reasoning_format"] == "hidden"
        assert response.text == final_answer
        assert response.diagnostics["reasoning_suppressed"] is True
        assert response.diagnostics["reasoning_chars"] == len(reasoning)
        assert all(marker not in response.text for marker in (
            "Analyze the Source Image",
            "Internal Monologue",
            "Draft the Explanation",
            "Check against constraints",
            "Final Output Generation",
        ))

    @pytest.mark.parametrize(
        "content",
        [
            [
                {
                    "type": "reasoning",
                    "text": "Analyze the Source Image and draft internally.",
                },
                {
                    "type": "text",
                    "text": "بناءً على الصورة المرفقة للصفحة 3 [1]، يوجد رسمان.",
                },
            ],
            (
                "<think>\n1. Analyze the Source Image:\n"
                "2. Draft the Explanation (Internal Monologue/Drafting):\n"
                "5. Check against constraints:\n"
                "6. Final Output Generation:\n</think>\n"
                "بناءً على الصورة المرفقة للصفحة 3 [1]، يوجد رسمان."
            ),
        ],
        ids=["separate-content-block", "documented-raw-think-envelope"],
    )
    def test_qwen_reasoning_shapes_keep_only_final_content(self, groq_http, content):
        _, responses = groq_http
        responses.append(
            FakeResponse(
                200,
                {
                    "model": DEFAULT_VISION_MODEL,
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )
        provider = GroqLLM(
            api_key="test-key",
            model=DEFAULT_VISION_MODEL,
            retry_attempts=1,
        )

        response = provider.complete(message())

        assert response.text.startswith("بناءً على الصورة المرفقة")
        assert response.diagnostics["reasoning_suppressed"] is True
        assert "Analyze the Source Image" not in response.text
        assert "Internal Monologue" not in response.text
        assert "Draft the Explanation" not in response.text
        assert "Check against constraints" not in response.text
        assert "Final Output Generation" not in response.text

    @pytest.mark.parametrize(
        "response,error_type",
        [
            (FakeResponse(401, text="invalid key"), ProviderAuthError),
            (FakeResponse(404, text="model not found"), ProviderBadRequestError),
            (FakeResponse(429, text="rate limit"), RateLimitError),
            (
                FakeResponse(
                    413,
                    text=(
                        "type=tokens code=rate_limit_exceeded Request too large "
                        "for qwen/qwen3.6-27b TPM Limit 8000 Requested 10468"
                    ),
                ),
                ProviderTokenBudgetExceededError,
            ),
            (FakeResponse(503, text="temporarily unavailable"), ProviderUnavailableError),
            (httpx.ReadTimeout("timed out"), ProviderTimeoutError),
        ],
    )
    def test_typed_errors(self, groq_http, response, error_type):
        _, responses = groq_http
        responses.append(response)
        provider = GroqLLM(api_key="test-key", retry_attempts=1)

        with pytest.raises(error_type):
            provider.complete(message())

    def test_capability_detection_is_conservative(self):
        assert model_supports_images(DEFAULT_VISION_MODEL) is True
        assert model_supports_images(DEFAULT_MODEL) is False
        assert model_supports_images("unknown/future-model") is False

    def test_retry_after_ten_seconds_waits_once_then_succeeds(
        self, groq_http, monkeypatch
    ):
        calls, responses = groq_http
        sleeps = []
        monkeypatch.setattr("backend.utils.time.sleep", sleeps.append)
        responses.extend(
            [
                FakeResponse(
                    429,
                    text="TPM limit reached; try again in 10s",
                    headers={"retry-after": "10"},
                ),
                FakeResponse(
                    200,
                    {
                        "model": DEFAULT_MODEL,
                        "choices": [
                            {"message": {"content": "ok"}, "finish_reason": "stop"}
                        ],
                    },
                ),
            ]
        )
        provider = GroqLLM(
            api_key="test-key",
            retry_attempts=2,
            max_rate_limit_wait_seconds=20,
        )

        assert provider.complete(message()).text == "ok"
        assert len(calls) == 2
        assert sleeps == [10]

    def test_retry_after_over_bound_fails_over_without_sleep(
        self, groq_http, monkeypatch
    ):
        calls, responses = groq_http
        sleeps = []
        monkeypatch.setattr("backend.utils.time.sleep", sleeps.append)
        responses.append(
            FakeResponse(
                429,
                text="TPM limit reached; try again in 30s",
                headers={"retry-after": "30"},
            )
        )
        groq = GroqLLM(
            api_key="test-key",
            retry_attempts=2,
            max_rate_limit_wait_seconds=20,
        )
        fallback = ScriptedProvider("openrouter", ["fallback"])

        response = FallbackLLMProvider([groq, fallback]).complete(message())

        assert response.provider == "openrouter"
        assert len(calls) == 1
        assert sleeps == []

    def test_router_logs_actual_groq_vision_model(self, groq_http, caplog):
        _, responses = groq_http
        responses.append(
            FakeResponse(
                200,
                {
                    "model": DEFAULT_VISION_MODEL,
                    "choices": [
                        {
                            "message": {"content": "visual"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )
        router = FallbackLLMProvider(
            [GroqLLM(api_key="test-key", retry_attempts=1)]
        )

        with caplog.at_level("INFO"):
            router.complete(message(with_image=True))

        assert any(
            f"provider=groq role=primary model={DEFAULT_VISION_MODEL}" in record.message
            for record in caplog.records
        )
        assert any(
            f"requested_model={DEFAULT_VISION_MODEL} "
            f"response_model={DEFAULT_VISION_MODEL}" in record.message
            for record in caplog.records
        )

    def test_focused_vision_413_retries_once_with_smaller_output_and_image(
        self, groq_http
    ):
        calls, responses = groq_http
        responses.extend(
            [
                FakeResponse(
                    413,
                    text=(
                        "type=tokens code=rate_limit_exceeded Request too large "
                        "TPM Limit 8000 Requested 8500"
                    ),
                ),
                FakeResponse(
                    200,
                    {
                        "model": DEFAULT_VISION_MODEL,
                        "choices": [
                            {
                                "message": {"content": "focused answer"},
                                "finish_reason": "stop",
                            }
                        ],
                    },
                ),
            ]
        )
        provider = GroqLLM(
            api_key="test-key",
            retry_attempts=1,
            max_output_tokens=2048,
        )
        msgs = message(with_image=True)

        response = provider.complete(
            msgs,
            max_output_tokens=2048,
            requirements=LLMRequestRequirements(
                requires_images=True,
                operation="final_answer",
            ),
        )

        assert response.text == "focused answer"
        assert [
            call["json"]["max_completion_tokens"] for call in calls
        ] == [1024, 512]
        assert all(
            any(
                part.get("type") == "image_url"
                for part in call["json"]["messages"][0]["content"]
            )
            for call in calls
        )


# --------------------------------------------------------------------------- #
# Three-provider routing
# --------------------------------------------------------------------------- #
class TestThreeProviderRouting:
    def test_gemini_success_does_not_call_fallbacks(self):
        gemini = ScriptedProvider("gemini", ["primary"])
        groq = ScriptedProvider("groq", ["middle"])
        openrouter = ScriptedProvider("openrouter", ["last"])

        response = FallbackLLMProvider([gemini, groq, openrouter]).complete(message())

        assert response.provider == "gemini"
        assert (gemini.call_count, groq.call_count, openrouter.call_count) == (1, 0, 0)

    def test_gemini_429_uses_groq_without_openrouter(self):
        gemini = ScriptedProvider("gemini", [RateLimitError("429", provider="gemini")])
        groq = ScriptedProvider("groq", ["fast fallback"])
        openrouter = ScriptedProvider("openrouter", ["last"])

        response = FallbackLLMProvider([gemini, groq, openrouter]).complete(message())

        assert response.provider == "groq"
        assert response.text == "fast fallback"
        assert openrouter.call_count == 0
        assert response.diagnostics["fallback_position"] == 1

    def test_gemini_and_groq_429_use_openrouter(self):
        gemini = ScriptedProvider("gemini", [RateLimitError("429", provider="gemini")])
        groq = ScriptedProvider("groq", [RateLimitError("429", provider="groq")])
        openrouter = ScriptedProvider("openrouter", ["final fallback"])

        response = FallbackLLMProvider([gemini, groq, openrouter]).complete(message())

        assert response.provider == "openrouter"
        assert response.text == "final fallback"
        assert response.diagnostics["fallback_position"] == 2

    def test_gemini_cooldown_then_groq_413_advances_to_openrouter(self):
        gemini = ScriptedProvider(
            "gemini",
            [
                RateLimitError(
                    "hard quota",
                    provider="gemini",
                    quota_exhausted=True,
                )
            ],
        )
        groq = ScriptedProvider(
            "groq",
            [
                "prime cooldown state",
                ProviderTokenBudgetExceededError(
                    "413 TPM request too large",
                    provider="groq",
                ),
            ],
        )
        openrouter = ScriptedProvider("openrouter", ["fallback answer"])
        router = FallbackLLMProvider([gemini, groq, openrouter])

        with llm_session("production-shaped-413"):
            assert router.complete(message()).provider == "groq"
            response = router.complete(message())

        assert response.provider == "openrouter"
        assert response.text == "fallback answer"
        assert gemini.call_count == 1
        assert groq.call_count == 2
        assert openrouter.call_count == 1

    def test_payment_required_uses_next_provider_without_same_route_retry(self):
        gemini = ScriptedProvider(
            "gemini", [ProviderPaymentRequiredError("402", provider="gemini")]
        )
        groq = ScriptedProvider("groq", ["answer"])
        openrouter = ScriptedProvider("openrouter", ["never"])

        response = FallbackLLMProvider([gemini, groq, openrouter]).complete(message())

        assert response.provider == "groq"
        assert (gemini.call_count, groq.call_count, openrouter.call_count) == (1, 1, 0)

    def test_gemini_cooldown_routes_directly_to_groq(self):
        gemini = ScriptedProvider("gemini", [RateLimitError("429", provider="gemini")])
        groq = ScriptedProvider("groq", ["first", "second"])
        openrouter = ScriptedProvider("openrouter", ["never"])
        router = FallbackLLMProvider([gemini, groq, openrouter])

        with llm_session("session"):
            assert router.complete(message()).provider == "groq"
            assert router.complete(message()).provider == "groq"

        assert gemini.call_count == 1
        assert groq.call_count == 2
        assert openrouter.call_count == 0

    def test_gemini_and_groq_cooldowns_route_directly_to_openrouter(self):
        gemini = ScriptedProvider("gemini", [RateLimitError("429", provider="gemini")])
        groq = ScriptedProvider("groq", [RateLimitError("429", provider="groq")])
        openrouter = ScriptedProvider("openrouter", ["first", "second"])
        router = FallbackLLMProvider([gemini, groq, openrouter])

        with llm_session("session"):
            assert router.complete(message()).provider == "openrouter"
            assert router.complete(message()).provider == "openrouter"

        assert gemini.call_count == 1
        assert groq.call_count == 1
        assert openrouter.call_count == 2

    @pytest.mark.parametrize(
        "failure",
        [
            ProviderUnavailableError("503", provider="groq"),
            ProviderTimeoutError("timeout", provider="groq"),
        ],
    )
    def test_recoverable_groq_failure_uses_openrouter(self, failure):
        gemini = ScriptedProvider("gemini", [RateLimitError("429", provider="gemini")])
        groq = ScriptedProvider("groq", [failure])
        openrouter = ScriptedProvider("openrouter", ["answer"])

        response = FallbackLLMProvider([gemini, groq, openrouter]).complete(message())

        assert response.provider == "openrouter"
        assert openrouter.call_count == 1

    def test_groq_auth_error_is_nonrecoverable_configuration_failure(self):
        gemini = ScriptedProvider("gemini", [RateLimitError("429", provider="gemini")])
        groq = ScriptedProvider("groq", [ProviderAuthError("401", provider="groq")])
        openrouter = ScriptedProvider("openrouter", ["must not run"])

        with pytest.raises(AllProvidersFailedError) as excinfo:
            FallbackLLMProvider([gemini, groq, openrouter]).complete(message())

        assert [name for name, _ in excinfo.value.failures] == ["gemini", "groq"]
        assert openrouter.call_count == 0

    def test_visual_request_skips_text_only_groq_and_preserves_image(self):
        gemini = ScriptedProvider("gemini", [RateLimitError("429", provider="gemini")])
        groq = ScriptedProvider("groq", ["must not run"], images=False)
        openrouter = ScriptedProvider("openrouter", ["visual answer"], images=True)

        response = FallbackLLMProvider([gemini, groq, openrouter]).complete(
            message(with_image=True)
        )

        assert response.provider == "openrouter"
        assert groq.call_count == 0
        assert openrouter.last_messages[0].images[0].media_type == "image/png"

    def test_visual_request_uses_capable_groq(self):
        gemini = ScriptedProvider("gemini", [RateLimitError("429", provider="gemini")])
        groq = ScriptedProvider("groq", ["visual answer"], images=True)
        openrouter = ScriptedProvider("openrouter", ["never"])

        response = FallbackLLMProvider([gemini, groq, openrouter]).complete(
            message(with_image=True)
        )

        assert response.provider == "groq"
        assert groq.last_messages[0].images
        assert openrouter.call_count == 0

    def test_safety_refusal_never_falls_back(self):
        gemini = ScriptedProvider(
            "gemini", [ProviderPolicyError("safety", provider="gemini")]
        )
        groq = ScriptedProvider("groq", ["must not run"])
        openrouter = ScriptedProvider("openrouter", ["must not run"])

        with pytest.raises(ProviderPolicyError):
            FallbackLLMProvider([gemini, groq, openrouter]).complete(message())

        assert groq.call_count == openrouter.call_count == 0

    def test_all_provider_failures_keep_complete_attempt_trail(self):
        router = FallbackLLMProvider([
            ScriptedProvider("gemini", [RateLimitError("429", provider="gemini")]),
            ScriptedProvider("groq", [ProviderUnavailableError("503", provider="groq")]),
            ScriptedProvider(
                "openrouter", [ProviderUnavailableError("503", provider="openrouter")]
            ),
        ])

        with pytest.raises(AllProvidersFailedError) as excinfo:
            router.complete(message())

        assert [name for name, _ in excinfo.value.failures] == [
            "gemini",
            "groq",
            "openrouter",
        ]


# --------------------------------------------------------------------------- #
# Embedding provider selection and fallback
# --------------------------------------------------------------------------- #
class FailingEmbeddings(BaseEmbeddingProvider):
    name = "remote"

    def __init__(self, exc: BaseException):
        super().__init__(model="remote-model", batch_size=2)
        self.exc = exc
        self.calls = 0

    def embed_batch(self, texts, *, is_query=False):
        self.calls += 1
        raise self.exc


def test_gemini_embedding_success_uses_supported_model_and_endpoint(monkeypatch):
    captured = {}

    def fake_post(url, payload, **kwargs):
        captured.update(url=url, payload=payload, headers=kwargs["headers"])
        return {"embeddings": [{"values": [0.25, 0.75]}]}

    monkeypatch.setattr("backend.providers.embeddings.post_json", fake_post)
    provider = GeminiEmbeddings(api_key="test-key")

    assert provider.embed_documents(["hello"]) == [[0.25, 0.75]]
    assert EMBEDDING_DEFAULT_MODEL == "gemini-embedding-001"
    assert captured["url"] == (
        f"{EMBEDDING_DEFAULT_BASE_URL}/models/gemini-embedding-001:batchEmbedContents"
    )
    assert captured["payload"]["requests"][0]["model"] == "models/gemini-embedding-001"
    assert captured["payload"]["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"


@pytest.mark.parametrize(
    "failure",
    [
        RateLimitError("quota", provider="remote", quota_exhausted=True),
        ProviderUnavailableError("server error", provider="remote"),
    ],
)
def test_provider_429_or_5xx_activates_sticky_hash_fallback(failure):
    primary = FailingEmbeddings(failure)
    provider = ResilientEmbeddings(primary)

    vectors = provider.embed_documents(["alpha", "beta"])
    query = provider.embed_query("alpha")

    assert provider.fallback_active
    assert provider.name == "hash"
    assert len(vectors) == 2
    assert all(len(vector) == DEFAULT_DIMENSIONS for vector in vectors)
    assert len(query) == DEFAULT_DIMENSIONS
    assert primary.calls == 1  # query remains in the hash vector space


def test_malformed_gemini_response_is_not_silently_hidden(monkeypatch):
    monkeypatch.setattr(
        "backend.providers.embeddings.post_json",
        lambda *args, **kwargs: {"unexpected": []},
    )
    provider = ResilientEmbeddings(GeminiEmbeddings(api_key="test-key"))

    with pytest.raises(EmbeddingError, match="Unexpected Gemini"):
        provider.embed_documents(["hello"])

    assert not provider.fallback_active


def test_programming_error_is_not_silently_hidden():
    provider = ResilientEmbeddings(FailingEmbeddings(ValueError("bug")))

    with pytest.raises(ValueError, match="bug"):
        provider.embed_documents(["hello"])

    assert not provider.fallback_active


def test_hash_embeddings_have_fixed_expected_dimension():
    provider = HashingEmbeddings()
    vectors = provider.embed_documents(["alpha", "", "مرحبا"])

    assert provider.dimensions == DEFAULT_DIMENSIONS
    assert len(vectors) == 3
    assert all(len(vector) == DEFAULT_DIMENSIONS for vector in vectors)
