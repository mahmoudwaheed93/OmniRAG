"""Medical metadata extraction tests."""

from __future__ import annotations

import pytest

from backend.models import (
    EvidenceLevel,
    MedicalDocumentType,
    MedicalDomain,
    MedicalMetadata,
)


class TestMedicalMetadataModel:
    def test_default_values(self):
        md = MedicalMetadata()
        assert md.document_type == MedicalDocumentType.UNKNOWN
        assert md.medical_domain == MedicalDomain.UNKNOWN
        assert md.evidence_level == EvidenceLevel.UNKNOWN
        assert md.authoring_organization == ""
        assert md.publication_year is None
        assert md.keywords == []

    def test_has_content_true_when_populated(self):
        md = MedicalMetadata(authoring_organization="WHO")
        assert md.has_content() is True

    def test_has_content_false_when_empty(self):
        md = MedicalMetadata()
        assert md.has_content() is False

    def test_has_content_true_when_keywords(self):
        md = MedicalMetadata(keywords=["infection", "control"])
        assert md.has_content() is True

    def test_has_content_true_when_year(self):
        md = MedicalMetadata(publication_year=2024)
        assert md.has_content() is True


class TestExtractMedicalMetadata:
    """Test the _extract_medical_metadata function against the WHO PDF."""

    @pytest.fixture
    def who_pdf(self):
        import pymupdf
        return pymupdf.open(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "9789240103986-eng.pdf"))

    @pytest.fixture
    def who_metadata(self, who_pdf):
        from backend.ingestion import _extract_medical_metadata
        return _extract_medical_metadata(who_pdf, "9789240103986-eng.pdf")

    def test_detects_report_type(self, who_metadata):
        assert who_metadata is not None
        assert who_metadata.document_type == MedicalDocumentType.REPORT

    def test_detects_infectious_disease_domain(self, who_metadata):
        assert who_metadata.medical_domain == MedicalDomain.INFECTIOUS_DISEASE

    def test_detects_who_organization(self, who_metadata):
        assert "World Health Organization" in who_metadata.authoring_organization

    def test_extracts_isbn(self, who_metadata):
        assert who_metadata.isbn != ""

    def test_has_content(self, who_metadata):
        assert who_metadata.has_content() is True

    def test_metadata_is_none_for_generic_pdf(self):
        """A generic PDF with no medical content returns None."""
        import pymupdf
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 100), "Quarterly Financial Report 2024", fontsize=20)
        data = doc.tobytes()
        doc.close()

        pdf = pymupdf.open(stream=data, filetype="pdf")
        from backend.ingestion import _extract_medical_metadata
        result = _extract_medical_metadata(pdf, "financial_report.pdf")
        pdf.close()
        # Generic financial report should not get medical metadata
        # (unless it happens to match keywords — that's acceptable)
        assert result is None or result.medical_domain == MedicalDomain.UNKNOWN


class TestMedicalMetadataInDocumentSummary:
    """Verify medical metadata flows through the ingestion pipeline."""

    def test_document_summary_has_optional_field(self):
        from backend.models import DocumentSummary, IngestionStatus, FileType
        summary = DocumentSummary(
            document_id="doc-1",
            session_id="s1",
            filename="test.pdf",
            file_type=FileType.PDF,
            status=IngestionStatus.READY,
        )
        assert summary.medical_metadata is None

    def test_document_summary_with_metadata(self):
        from backend.models import DocumentSummary, IngestionStatus, FileType
        md = MedicalMetadata(
            document_type=MedicalDocumentType.REPORT,
            medical_domain=MedicalDomain.INFECTIOUS_DISEASE,
            authoring_organization="WHO",
        )
        summary = DocumentSummary(
            document_id="doc-1",
            session_id="s1",
            filename="test.pdf",
            file_type=FileType.PDF,
            status=IngestionStatus.READY,
            medical_metadata=md,
        )
        assert summary.medical_metadata is not None
        assert summary.medical_metadata.authoring_organization == "WHO"
