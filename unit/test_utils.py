"""Consolidated utility tests: language/text, models, config, and token efficiency."""

from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from backend.config import apply_secrets, build_settings, get_settings, load_dotenv, reset_settings_cache
from backend.ingestion import PDFProcessor, ProcessingContext
from backend.intelligence import OCREngine, VisionAnalyzer
from backend.models import (
    AllProvidersFailedError,
    BlockType,
    BoundingBox,
    Chunk,
    ConfigurationError,
    ContentBlock,
    Document,
    DocumentSummary,
    FileType,
    Language,
    MissingCredentialError,
    Page,
    RateLimitError,
    RetrievalResult,
    Role,
    SearchResult,
    SourceKind,
    TableData,
    VisualRef,
)
from backend.providers.llm import (
    BaseLLMProvider,
    FallbackLLMProvider,
    GroqLLM,
    ImagePart,
    LLMMessage,
    LLMRequestRequirements,
    LLMResponse,
    llm_session,
)
from backend.providers.ocr import VisionOCRProvider
from backend.rag.generation import AnswerGenerator, GenerationRequest
from backend.rag.retrieval import parse_query
from backend.utils import (
    clean_text,
    content_hash,
    contains_arabic,
    detect_language,
    detect_repeated_lines,
    estimate_tokens,
    file_extension,
    is_meaningful,
    is_rtl,
    normalize_arabic,
    normalize_for_search,
    remove_lines,
    sanitize_filename,
    script_ratios,
    split_paragraphs,
    split_sentences,
    stable_id,
    text_hash,
    tokenize,
    truncate,
)

DEFAULT_VISION_MODEL = GroqLLM.DEFAULT_VISION_MODEL


# --- Language & Text ---


class TestLanguageDetection:
    @pytest.mark.parametrize("text,expected", [
        ("Total revenue reached 8.4 million USD.", Language.ENGLISH),
        ("بلغت الإيرادات الإجمالية ثمانية ملايين دولار", Language.ARABIC),
        ("الإيرادات Revenue بلغت 8.4 million دولار أمريكي", Language.MIXED),
        ("", Language.UNKNOWN),
        ("12345 6789 ... !!!", Language.UNKNOWN),
    ])
    def test_detection(self, text, expected):
        assert detect_language(text) == expected

    def test_script_ratios(self):
        arabic, latin = script_ratios("abc أبج")
        assert arabic == pytest.approx(0.5)
        assert latin == pytest.approx(0.5)

    def test_contains_arabic(self):
        assert contains_arabic("hello مرحبا") is True
        assert contains_arabic("hello world") is False

    def test_rtl_detection(self):
        assert is_rtl("بلغت الإيرادات الإجمالية") is True
        assert is_rtl("Total revenue") is False


class TestArabicNormalization:
    @pytest.mark.parametrize("raw,expected_contains", [
        ("الإيرادات", "الايرادات"),   # hamza forms unified
        ("أحمد", "احمد"),
        ("مكتبة", "مكتبه"),           # teh marbuta -> heh
        ("علــــى", "علي"),           # tatweel removed, alef maqsura -> yeh
    ])
    def test_normalization_unifies_variants(self, raw, expected_contains):
        assert normalize_arabic(raw) == expected_contains

    def test_diacritics_are_stripped(self):
        assert normalize_arabic("مُحَمَّد") == normalize_arabic("محمد")

    def test_search_normalization_is_lowercase(self):
        assert normalize_for_search("  Revenue   GREW  ") == "revenue grew"

    def test_normalization_never_mutates_stored_text(self):
        original = "الإيرادات"
        normalize_arabic(original)
        assert original == "الإيرادات"


class TestTokenization:
    def test_arabic_and_english_tokens(self):
        tokens = tokenize("Revenue بلغت 8400 USD")
        assert "revenue" in tokens
        assert "8400" in tokens
        assert any(contains_arabic(t) for t in tokens)

    def test_tokens_are_normalized_for_matching(self):
        assert tokenize("الإيرادات") == tokenize("الايرادات")


class TestTextCleaning:
    def test_hyphenated_line_breaks_are_rejoined(self):
        assert "revenue" in clean_text("reve-\nnue increased")

    def test_control_characters_are_removed(self):
        assert "\x00" not in clean_text("bad\x00text here")

    def test_excessive_whitespace_is_collapsed(self):
        assert clean_text("a    b\n\n\n\nc") == "a b\n\nc"

    def test_numbers_are_never_altered(self):
        text = "Revenue was 8,400,000.50 USD (+12.3%)"
        assert "8,400,000.50" in clean_text(text)
        assert "12.3%" in clean_text(text)

    def test_arabic_text_survives_cleaning(self):
        assert "الإيرادات" in clean_text("  الإيرادات   الإجمالية  ")

    def test_repeated_headers_are_detected_and_removable(self):
        pages = [
            f"ACME CONFIDENTIAL\nPage body {i} with unique content here.\nFooter line"
            for i in range(6)
        ]
        repeated = detect_repeated_lines(pages)

        assert "ACME CONFIDENTIAL" in repeated
        assert "Footer line" in repeated

        cleaned = remove_lines(pages[0], repeated)
        assert "ACME CONFIDENTIAL" not in cleaned
        assert "Page body 0" in cleaned

    def test_body_text_repeating_by_chance_is_not_stripped(self):
        pages = ["Header\n" + "unique %d\n" % i + "Total revenue grew." for i in range(6)]
        repeated = detect_repeated_lines(pages)
        assert "Header" in repeated


class TestTextHelpers:
    def test_sentence_splitting_handles_arabic_punctuation(self):
        sentences = split_sentences("ما هي الإيرادات؟ بلغت 8.4 مليون. نعم.")
        assert len(sentences) >= 2

    def test_paragraph_splitting(self):
        assert len(split_paragraphs("one\n\ntwo\n\nthree")) == 3

    def test_truncate_respects_word_boundaries(self):
        result = truncate("the quick brown fox jumps over", 15)
        assert len(result) <= 16
        assert result.endswith("…")

    def test_token_estimation_is_positive(self):
        assert estimate_tokens("hello world") > 0
        assert estimate_tokens("") == 0

    @pytest.mark.parametrize("text,expected", [
        ("Total revenue reached 8.4 million.", True),
        ("...", False),
        ("a", False),
        ("|||||||||||||", False),
        ("بلغت الإيرادات الإجمالية", True),
    ])
    def test_meaningfulness_filter(self, text, expected):
        assert is_meaningful(text) is expected


class TestHashing:
    def test_content_hash_is_stable_and_distinct(self):
        assert content_hash(b"abc") == content_hash(b"abc")
        assert content_hash(b"abc") != content_hash(b"abd")

    def test_text_hash_ignores_case_and_whitespace(self):
        assert text_hash("Hello   World") == text_hash("hello world")

    def test_stable_id_is_deterministic(self):
        assert stable_id("a", "b", "c") == stable_id("a", "b", "c")
        assert stable_id("a", "b") != stable_id("a", "c")

    @pytest.mark.parametrize("raw,checks", [
        ("../../etc/passwd", lambda n: "/" not in n and ".." not in n),
        ("C:\\Windows\\system32\\evil.txt", lambda n: "\\" not in n),
        ("report<>:\"|?*.pdf", lambda n: not set(n) & set('<>:"|?*')),
        ("", lambda n: n == "untitled"),
        ("CON.txt", lambda n: n.startswith("_")),
        ("تقرير سنوي.pdf", lambda n: "تقرير" in n),
        ("normal_file-1.pdf", lambda n: n == "normal_file-1.pdf"),
    ])
    def test_filename_sanitisation(self, raw, checks):
        assert checks(sanitize_filename(raw))

    def test_long_filenames_are_truncated_but_keep_the_extension(self):
        result = sanitize_filename("x" * 500 + ".pdf")
        assert len(result) <= 130
        assert result.endswith(".pdf")

    def test_file_extension(self):
        assert file_extension("Report.PDF") == "pdf"
        assert file_extension("noext") == ""


class TestRetryPolicy:
    def test_transient_failures_are_retried(self):
        from backend.utils import retry_call

        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RateLimitError("429", provider="test")
            return "ok"

        assert retry_call(flaky, attempts=3, sleep=lambda _: None) == "ok"
        assert attempts["n"] == 3

    def test_permanent_failures_are_not_retried(self):
        from backend.models import ProviderAuthError
        from backend.utils import retry_call

        attempts = {"n": 0}

        def failing():
            attempts["n"] += 1
            raise ProviderAuthError("401", provider="test")

        with pytest.raises(ProviderAuthError):
            retry_call(failing, attempts=4, sleep=lambda _: None)
        assert attempts["n"] == 1

    def test_backoff_is_bounded(self):
        from backend.utils import backoff_delay

        for attempt in range(1, 8):
            assert 0 <= backoff_delay(attempt, base=1.0, cap=8.0) <= 8.0


class TestLogRedaction:
    @pytest.mark.parametrize("secret", [
        "sk-abcdefghijklmnop",
        "sk-ant-abcdefghijklmnop",
        "AIzaSyABCDEFGHIJKLMNOPQRSTUV",
    ])
    def test_credentials_are_redacted_from_log_messages(self, secret):
        from backend.utils import redact

        assert secret not in redact(f"calling provider with key {secret}")

    def test_bearer_tokens_are_redacted(self):
        from backend.utils import redact

        assert "abcdef1234567890" not in redact("Authorization: Bearer abcdef1234567890")


# --- Models ---


class TestContentBlock:
    def test_search_text_combines_every_representation(self):
        block = ContentBlock(
            document_id="d",
            session_id="s",
            block_type=BlockType.CHART,
            text="Revenue by quarter",
            visual_description="Bar chart showing Q4 at 8400",
            parent_section="Financials",
        )
        text = block.search_text

        assert "Financials" in text
        assert "Revenue by quarter" in text
        assert "Bar chart showing Q4 at 8400" in text

    def test_table_contributes_summary_and_markdown(self):
        table = TableData.from_rows([["Region", "Q4"], ["EMEA", "3400"]])
        table.summary = "Columns: Region, Q4."
        block = ContentBlock(
            document_id="d", session_id="s", block_type=BlockType.TABLE, table=table
        )

        assert "Columns: Region, Q4." in block.search_text
        assert "| EMEA | 3400 |" in block.search_text

    def test_empty_block_is_detected(self):
        assert ContentBlock(document_id="d", session_id="s").is_empty is True

    def test_confidence_is_clamped(self):
        assert ContentBlock(document_id="d", session_id="s", confidence=1.7).confidence == 1.0
        assert ContentBlock(document_id="d", session_id="s", confidence=-3).confidence == 0.0

    def test_visual_blocks_are_recognised(self):
        assert BlockType.CHART.is_visual is True
        assert BlockType.DIAGRAM.is_visual is True
        assert BlockType.HANDWRITING.is_visual is True
        assert BlockType.TEXT.is_visual is False


class TestChunkTraceability:
    def test_chunk_requires_at_least_one_source_block(self):
        with pytest.raises(ValidationError):
            Chunk(
                document_id="d",
                session_id="s",
                filename="a.pdf",
                block_ids=[],
                text="orphan",
            )

    def test_citation_label_is_human_readable(self):
        chunk = Chunk(
            document_id="d",
            session_id="s",
            filename="annual_report.pdf",
            page_number=18,
            page_label="Page 18",
            block_ids=["b1"],
            text="…",
        )
        assert chunk.citation_label == "[annual_report.pdf — Page 18]"

    def test_payload_round_trip_preserves_provenance(self):
        original = Chunk(
            document_id="doc-1",
            session_id="sess-1",
            filename="deck.pptx",
            file_type=FileType.PPTX,
            page_number=7,
            page_label="Slide 7",
            block_ids=["b1", "b2"],
            block_type=BlockType.CHART,
            source_kind=SourceKind.VISION,
            text="Chart showing growth",
            section="Market",
            language=Language.ENGLISH,
            visual=VisualRef(asset_id="asset-9", media_type="image/jpeg", origin="embedded"),
            confidence=0.82,
            uncertain=True,
        )

        restored = Chunk.from_payload(original.to_payload())

        assert restored.chunk_id == original.chunk_id
        assert restored.block_ids == ["b1", "b2"]
        assert restored.page_label == "Slide 7"
        assert restored.block_type == BlockType.CHART
        assert restored.source_kind == SourceKind.VISION
        assert restored.visual is not None
        assert restored.visual.asset_id == "asset-9"
        assert restored.visual.media_type == "image/jpeg"
        assert restored.uncertain is True

    def test_payload_always_carries_the_session_id(self):
        chunk = Chunk(
            document_id="d", session_id="s-42", filename="f", block_ids=["b"], text="t"
        )
        assert chunk.to_payload()["session_id"] == "s-42"


class TestTableData:
    def test_from_rows_detects_a_header_and_builds_markdown(self):
        table = TableData.from_rows(
            [["Region", "Q3", "Q4"], ["EMEA", "2100", "3400"], ["APAC", "1800", "2600"]]
        )

        assert table.header == ["Region", "Q3", "Q4"]
        assert table.n_rows == 2
        assert table.n_cols == 3
        assert "| Region | Q3 | Q4 |" in table.markdown
        assert "| EMEA | 2100 | 3400 |" in table.markdown

    def test_pipes_in_cells_are_escaped(self):
        table = TableData.from_rows([["a|b", "c"], ["1", "2"]])
        assert r"a\|b" in table.markdown

    def test_ragged_rows_are_padded(self):
        table = TableData.from_rows([["a", "b", "c"], ["1"]])
        assert table.n_cols == 3
        assert table.markdown.count("|") > 0


class TestDocument:
    def test_blocks_are_flattened_across_pages(self):
        document = Document(session_id="s", filename="f.pdf")
        for number in (1, 2):
            page = Page(document_id=document.document_id, session_id="s", page_number=number)
            page.blocks.append(
                ContentBlock(
                    document_id=document.document_id,
                    session_id="s",
                    page_number=number,
                    text=f"page {number}",
                )
            )
            document.pages.append(page)

        assert len(document.blocks) == 2
        assert document.page(2) is not None
        assert document.page(9) is None

    def test_page_display_label_falls_back_to_page_number(self):
        page = Page(document_id="d", session_id="s", page_number=4)
        assert page.display_label == "Page 4"
        page.label = "Slide 4"
        assert page.display_label == "Slide 4"


class TestDocumentSummary:
    @pytest.mark.parametrize("size,expected", [
        (512, "512 B"), (2048, "2.0 KB"), (5 * 1024 * 1024, "5.0 MB"),
    ])
    def test_size_label_is_human_readable(self, size, expected):
        summary = DocumentSummary(
            document_id="d", session_id="s", filename="f", file_type=FileType.PDF,
            size_bytes=size,
        )
        assert summary.size_label == expected


class TestBoundingBox:
    def test_geometry(self):
        box = BoundingBox(x0=10, y0=20, x1=110, y1=70)
        assert box.width == 100
        assert box.height == 50
        assert box.area == 5000
        assert box.as_tuple() == (10, 20, 110, 70)


# --- Config ---


class TestDefaults:
    def test_defaults_are_sensible_without_any_environment(self):
        settings = build_settings()

        assert settings.llm.provider == "gemini"
        assert settings.retrieval.top_k > settings.retrieval.rerank_top_k
        assert settings.chunking.chunk_overlap < settings.chunking.chunk_size
        assert settings.upload.max_upload_mb > 0
        assert "pdf" in settings.upload.allowed_extensions

    def test_no_provider_means_not_ready_with_a_helpful_message(self):
        settings = build_settings()

        assert settings.is_ready is False
        issues = settings.validation_issues()
        assert any("GEMINI_API_KEY" in issue for issue in issues)

    def test_hash_embeddings_are_selected_offline_and_warned_about(self):
        settings = build_settings()

        assert settings.embedding.provider == "hash"
        assert any("hash" in w for w in settings.warnings())


class TestParsing:
    def test_numeric_settings_are_parsed(self, monkeypatch):
        monkeypatch.setenv("TOP_K", "40")
        monkeypatch.setenv("RERANK_TOP_K", "6")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
        monkeypatch.setenv("MAX_UPLOAD_MB", "12.5")

        settings = build_settings()

        assert settings.retrieval.top_k == 40
        assert settings.retrieval.rerank_top_k == 6
        assert settings.llm.temperature == 0.7
        assert settings.upload.max_upload_mb == 12.5

    @pytest.mark.parametrize("value,expected", [
        ("true", True), ("1", True), ("yes", True), ("on", True),
        ("false", False), ("0", False), ("no", False), ("off", False),
    ])
    def test_boolean_settings_are_parsed(self, monkeypatch, value, expected):
        monkeypatch.setenv("VISION_ENABLED", value)
        assert build_settings().vision.enabled is expected

    def test_invalid_number_raises_a_helpful_configuration_error(self, monkeypatch):
        monkeypatch.setenv("TOP_K", "not-a-number")

        with pytest.raises(ConfigurationError) as excinfo:
            build_settings()
        assert "TOP_K" in excinfo.value.user_message

    def test_overlap_larger_than_chunk_size_is_rejected(self, monkeypatch):
        monkeypatch.setenv("CHUNK_SIZE", "200")
        monkeypatch.setenv("CHUNK_OVERLAP", "400")

        with pytest.raises(ConfigurationError) as excinfo:
            build_settings()
        assert "CHUNK_OVERLAP" in excinfo.value.user_message


class TestSecretHandling:
    def test_redacted_view_masks_every_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSuperSecretValue123")
        monkeypatch.setenv("QDRANT_API_KEY", "qdrant-secret-value")

        redacted = build_settings().redacted()
        serialized = str(redacted)

        assert "AIzaSuperSecretValue123" not in serialized
        assert "qdrant-secret-value" not in serialized

    def test_require_key_raises_a_missing_credential_error(self):
        settings = build_settings()
        with pytest.raises(MissingCredentialError):
            settings.llm.require_key()

    def test_apply_secrets_maps_flat_keys(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        applied = apply_secrets({"GEMINI_API_KEY": "from-secrets"})

        assert applied == 1
        assert build_settings().llm.endpoints[0].api_key == "from-secrets"

    def test_apply_secrets_maps_sections(self, monkeypatch):
        monkeypatch.delenv("QDRANT_URL", raising=False)
        apply_secrets({"qdrant": {"url": "https://example.qdrant.io", "collection": "c"}})

        settings = build_settings()
        assert settings.vector_store.url == "https://example.qdrant.io"
        assert settings.vector_store.collection == "c"

    def test_real_environment_wins_over_secrets(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "from-env")
        apply_secrets({"GEMINI_API_KEY": "from-secrets"})

        assert build_settings().llm.endpoints[0].api_key == "from-env"

    def test_streamlit_secret_value_refreshes_without_overriding_host_env(
        self, monkeypatch
    ):
        monkeypatch.delenv("LLM_MAX_OUTPUT_TOKENS", raising=False)
        apply_secrets({"LLM_MAX_OUTPUT_TOKENS": "1400"})
        assert build_settings().llm.max_output_tokens == 1400

        apply_secrets({"LLM_MAX_OUTPUT_TOKENS": "4096"})
        assert build_settings().llm.max_output_tokens == 4096

        monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "7777")
        apply_secrets({"LLM_MAX_OUTPUT_TOKENS": "8192"})
        assert build_settings().llm.max_output_tokens == 7777

    def test_missing_dotenv_file_is_not_an_error(self):
        assert load_dotenv("definitely-not-a-real-file.env") == 0


class TestCaching:
    def test_get_settings_is_cached_and_resettable(self, monkeypatch):
        monkeypatch.setenv("TOP_K", "11")
        first = get_settings()
        assert first.retrieval.top_k == 11

        monkeypatch.setenv("TOP_K", "22")
        assert get_settings().retrieval.top_k == 11  # still cached

        reset_settings_cache()
        assert get_settings().retrieval.top_k == 22

    def test_streamlit_engine_rebuilds_when_output_budget_changes(self, monkeypatch):
        from backend.services import reset_engine
        from frontend import ui as state

        reset_engine()
        monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "1400")
        monkeypatch.setenv("LLM_EXHAUSTIVE_MAX_OUTPUT_TOKENS", "1400")
        reset_settings_cache()
        first = state.engine()

        monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "4096")
        monkeypatch.setenv("LLM_EXHAUSTIVE_MAX_OUTPUT_TOKENS", "8192")
        reset_settings_cache()
        second = state.engine()

        assert second is not first
        assert second.settings.llm.max_output_tokens == 4096
        assert second.settings.llm.exhaustive_max_output_tokens == 8192

    def test_generation_debug_mode_is_disabled_by_default_and_configurable(
        self, monkeypatch
    ):
        assert build_settings().debug_generation is False
        monkeypatch.setenv("OMNIRAG_DEBUG_GENERATION", "true")
        assert build_settings().debug_generation is True


class TestEmbeddingIndependence:
    def test_embeddings_use_gemini_key_when_present(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        settings = build_settings()

        assert settings.embedding.provider == "gemini"
        assert settings.embedding.api_key == "g-key"

    def test_dedicated_embedding_key_is_preferred(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        monkeypatch.setenv("EMBEDDING_API_KEY", "e-key")

        settings = build_settings()
        assert settings.embedding.api_key == "e-key"

    def test_openrouter_only_falls_back_to_hash_embeddings_with_a_warning(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        settings = build_settings()

        assert settings.embedding.provider == "hash"
        assert any("hash" in w for w in settings.warnings())

    def test_explicit_remote_embedding_provider_without_any_key_uses_hash(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")

        settings = build_settings()

        assert settings.embedding.provider == "hash"
        assert settings.embedding.model == "hash-1024"
        assert any("hash" in warning for warning in settings.warnings())


# --- Token Efficiency ---


class CountingVisionLLM(BaseLLMProvider):
    name = "counting"
    supports_vision = True

    def __init__(self):
        super().__init__(model="counting-vision")
        self.calls = []

    def supports_images(self, model=None):
        return True

    def complete(self, messages, **kwargs):
        self.calls.append(
            {
                "images": sum(len(message.images) for message in messages),
                "max_output_tokens": kwargs.get("max_output_tokens"),
                "json_mode": kwargs.get("json_mode", False),
            }
        )
        if kwargs.get("json_mode"):
            text = (
                '{"type":"diagram","title":"Page 3","description":'
                '"A labelled process diagram.","text":"مرحلة ١ ثم مرحلة ٢",'
                '"entities":[],"data_points":[],"confidence":0.9,'
                '"unreadable":false}'
            )
        else:
            text = "يوضح الرسم مرحلتين مترابطتين. [1]"
        return LLMResponse(text=text, model=self.model, provider=self.name)


class OutcomeProvider(BaseLLMProvider):
    def __init__(self, name, outcomes):
        super().__init__(model=f"{name}-model")
        self.name = name
        self.outcomes = list(outcomes)
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return LLMResponse(text=outcome, model=self.model, provider=self.name)


def test_production_default_operation_budgets(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_CHAIN", "mock")
    settings = build_settings()

    assert settings.vision.analysis_max_output_tokens == 800
    assert settings.ocr.max_output_tokens == 1200
    assert settings.retrieval.query_rewrite_max_output_tokens == 256
    assert settings.retrieval.rerank_max_output_tokens == 256
    assert settings.llm.max_output_tokens == 2048
    assert settings.llm.exhaustive_max_output_tokens == 4096
    assert settings.llm.groq_max_rate_limit_wait_seconds == 20
    assert settings.llm.groq_estimated_image_tokens == 2048
    assert settings.llm.groq_focused_vision_max_output_tokens == 1024


def test_scanned_pdf_ingestion_makes_zero_vision_calls(
    settings, file_store, sample_png
):
    pymupdf = pytest.importorskip("pymupdf")
    pdf = pymupdf.open()
    for _ in range(8):
        page = pdf.new_page(width=500, height=500)
        page.insert_image(page.rect, stream=sample_png)
    data = pdf.tobytes()
    pdf.close()

    llm = CountingVisionLLM()
    lazy_settings = replace(
        settings,
        vision=replace(settings.vision, enabled=True, lazy_analysis=True),
    )
    ctx = ProcessingContext(
        session_id="lazy-session",
        document_id="lazy-document",
        filename="scan.pdf",
        settings=lazy_settings,
        file_store=file_store,
        vision=VisionAnalyzer(llm, min_image_pixels=1),
        ocr=OCREngine(VisionOCRProvider(llm, max_output_tokens=1200)),
    )

    document = PDFProcessor().parse(data, ctx)

    assert document.page_count == 8
    assert len(document.blocks) == 8
    assert all(block.visual is not None for block in document.blocks)
    assert all(
        block.metadata["visual_analysis_pending"] is True
        for block in document.blocks
    )
    assert llm.calls == []


def test_page_three_visual_analysis_is_cached_across_generator_rebuilds(
    settings, file_store, sample_png
):
    llm = CountingVisionLLM()
    visual = file_store.put("session", sample_png, media_type="image/png")
    chunk = Chunk(
        document_id="page-three-document",
        session_id="session",
        filename="scan.pdf",
        file_type=FileType.PDF,
        page_number=3,
        page_label="Page 3",
        block_ids=["page-three-block"],
        block_type=BlockType.PAGE_SNAPSHOT,
        source_kind=SourceKind.VISION,
        text="Image on page 3 of scan.pdf.",
        visual=VisualRef(
            asset_id=visual.asset_id,
            media_type="image/png",
            origin="page_render",
            page_number=3,
        ),
        metadata={"visual_analysis_pending": True},
    )
    retrieval = RetrievalResult(
        query="page 3 اشرحلي الرسومات الموجودة في",
        results=[
            SearchResult(chunk=chunk, rank=0),
            SearchResult(
                chunk=chunk.model_copy(
                    update={
                        "chunk_id": "page-four-chunk",
                        "page_number": 4,
                        "page_label": "Page 4",
                        "block_ids": ["page-four-block"],
                    }
                ),
                rank=1,
            ),
        ],
    )
    plan = parse_query(retrieval.query)
    vision = VisionAnalyzer(llm, min_image_pixels=1, max_output_tokens=800)
    vision.clear_cache()

    for _ in range(2):
        generator = AnswerGenerator(
            llm, file_store=file_store, vision=vision, settings=replace(
                settings,
                vision=replace(settings.vision, enabled=True, lazy_analysis=True),
            )
        )
        result = generator.generate(
            GenerationRequest(
                question=retrieval.query,
                retrieval=retrieval,
                session_id="session",
                plan=plan,
            )
        )
        assert "[1]" in result.answer

    vision_calls = [call for call in llm.calls if call["json_mode"]]
    final_calls = [call for call in llm.calls if not call["json_mode"]]
    assert len(vision_calls) == 1
    assert vision_calls[0]["max_output_tokens"] == 800
    assert len(final_calls) == 2
    assert all(call["max_output_tokens"] == 2048 for call in final_calls)
    assert all(call["images"] == 1 for call in [*vision_calls, *final_calls])
    assert vision.cache_hits == 1


def test_focused_page_compaction_keeps_evidence_and_drops_unrelated_pages(
    settings, file_store, sample_png
):
    llm = CountingVisionLLM()
    asset = file_store.put("session", sample_png, media_type="image/png")
    visual = VisualRef(
        asset_id=asset.asset_id,
        media_type="image/png",
        origin="page_render",
        page_number=3,
    )
    first = Chunk(
        document_id="doc",
        session_id="session",
        filename="scan.pdf",
        page_number=3,
        block_ids=["vision"],
        block_type=BlockType.PAGE_SNAPSHOT,
        source_kind=SourceKind.VISION,
        text="Visual description of the coordination diagram.",
        visual=visual,
    )
    duplicate = first.model_copy(
        update={
            "chunk_id": "ocr-copy",
            "block_ids": ["ocr"],
            "text": "OCR labels: salary processing and bank information.",
        }
    )
    unrelated = first.model_copy(
        update={
            "chunk_id": "page-four",
            "page_number": 4,
            "page_label": "Page 4",
            "block_ids": ["page-four"],
            "text": "Unrelated Page 4 evidence.",
        }
    )
    generator = AnswerGenerator(llm, file_store=file_store, settings=settings)
    plan = parse_query("اشرح بيدج 3")

    compacted = generator._focused_page_results(
        [
            SearchResult(chunk=first, rank=0),
            SearchResult(chunk=duplicate, rank=1),
            SearchResult(chunk=unrelated, rank=2),
        ],
        plan,
    )

    assert len(compacted) == 1
    assert compacted[0].chunk.page_number == 3
    assert "coordination diagram" in compacted[0].chunk.text
    assert "salary processing" in compacted[0].chunk.text
    assert compacted[0].chunk.block_ids == ["vision", "ocr"]


def test_standalone_vision_ocr_obeys_1200_token_ceiling(sample_png):
    llm = CountingVisionLLM()
    provider = VisionOCRProvider(llm, max_output_tokens=1200)

    provider.recognize(sample_png)

    assert llm.calls == [
        {"images": 1, "max_output_tokens": 1200, "json_mode": True}
    ]


def test_groq_multimodal_diagnostics_select_actual_vision_model():
    provider = GroqLLM(api_key="not-sent", retry_attempts=1)
    messages = [
        LLMMessage(
            role=Role.USER,
            text="page 3",
            images=[ImagePart(data=b"image", media_type="image/png")],
        )
    ]

    assert provider.model_for_request(messages) == DEFAULT_VISION_MODEL


def test_groq_tpm_preflight_caps_output_without_dropping_evidence(monkeypatch):
    captured = {}

    def fake_complete(self, messages, **kwargs):
        captured.update(kwargs)
        captured["messages"] = messages
        return LLMResponse(text="ok", model=self.model, provider="groq")

    monkeypatch.setattr(
        "backend.providers.llm.OpenAICompatibleLLM.complete",
        fake_complete,
    )
    provider = GroqLLM(
        api_key="not-sent",
        tpm_limit=1000,
        estimated_image_tokens=100,
    )
    messages = [LLMMessage(text="evidence " * 120)]

    provider.complete(messages, max_output_tokens=900)

    assert captured["messages"] is messages
    assert 128 <= captured["max_output_tokens"] < 900


def test_focused_groq_page_vision_stays_below_tpm_and_preserves_image(monkeypatch):
    captured = {}

    def fake_complete(self, messages, **kwargs):
        captured.update(kwargs)
        captured["messages"] = messages
        return LLMResponse(text="ok", model=self.vision_model, provider="groq")

    monkeypatch.setattr(
        "backend.providers.llm.OpenAICompatibleLLM.complete",
        fake_complete,
    )
    provider = GroqLLM(
        api_key="not-sent",
        max_output_tokens=2048,
        tpm_limit=8000,
        estimated_image_tokens=2048,
        focused_vision_max_output_tokens=1024,
    )
    image = ImagePart(data=b"page-three-image", media_type="image/png")
    messages = [
        LLMMessage(
            text="Only the compact Page 3 evidence and citation identity.",
            images=[image],
        )
    ]

    response = provider.complete(
        messages,
        max_output_tokens=2048,
        requirements=LLMRequestRequirements(
            requires_images=True,
            operation="final_answer",
        ),
    )

    assert captured["messages"] is messages
    assert captured["messages"][0].images == [image]
    assert captured["max_output_tokens"] == 1024
    assert response.diagnostics["selected_visuals"] == 1
    assert response.diagnostics["estimated_total_tokens"] < 8000


def test_groq_retry_after_waits_once_when_within_bound(monkeypatch):
    from backend.utils import retry_call

    sleeps = []
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimitError("TPM", provider="groq", retry_after=10)
        return "ok"

    result = retry_call(
        operation,
        attempts=2,
        max_delay=20,
        skip_if_retry_after_exceeds_max=True,
        sleep=sleeps.append,
    )

    assert result == "ok"
    assert attempts == 2
    assert sleeps == [10]


def test_groq_retry_after_over_bound_falls_back_without_wait():
    from backend.utils import retry_call

    sleeps = []
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        raise RateLimitError("TPM", provider="groq", retry_after=30)

    with pytest.raises(RateLimitError):
        retry_call(
            operation,
            attempts=2,
            max_delay=20,
            skip_if_retry_after_exceeds_max=True,
            sleep=sleeps.append,
        )

    assert attempts == 1
    assert sleeps == []


def test_hard_gemini_quota_is_dead_for_only_the_affected_session():
    gemini = OutcomeProvider(
        "gemini",
        [
            RateLimitError(
                "daily free quota exhausted",
                provider="gemini",
                quota_exhausted=True,
                quota_scope="hard_quota",
            ),
            "new-session-primary",
        ],
    )
    groq = OutcomeProvider("groq", ["fallback-one", "fallback-two"])
    router = FallbackLLMProvider([gemini, groq])

    with llm_session("quota-session"):
        assert router.complete([LLMMessage(text="one")]).provider == "groq"
        assert router.complete([LLMMessage(text="two")]).provider == "groq"
    with llm_session("different-session"):
        assert router.complete([LLMMessage(text="three")]).provider == "gemini"

    assert gemini.calls == 2
    assert groq.calls == 2


def test_openrouter_zero_remaining_is_skipped_until_reported_reset():
    now = [0.0]
    openrouter = OutcomeProvider(
        "openrouter",
        [
            RateLimitError(
                "daily quota",
                provider="openrouter",
                quota_exhausted=True,
                quota_scope="daily_or_account",
                reset_at="100",
            ),
            "after-reset",
        ],
    )
    router = FallbackLLMProvider(
        [openrouter],
        clock=lambda: now[0],
        wall_clock=lambda: 0.0,
    )

    with llm_session("openrouter-session"):
        with pytest.raises(AllProvidersFailedError):
            router.complete([LLMMessage(text="one")])
        with pytest.raises(AllProvidersFailedError):
            router.complete([LLMMessage(text="two")])
        assert openrouter.calls == 1
        now[0] = 101
        assert router.complete([LLMMessage(text="three")]).text == "after-reset"

    assert openrouter.calls == 2
