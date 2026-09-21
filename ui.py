"""Unified Streamlit UI layer.

Combines state management, styles, components, message actions, source rendering,
the main chat panel, and the sidebar into a single module.
"""

from __future__ import annotations

import html
import json
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import streamlit as st

from backend.config import AppSettings, get_settings
from backend.ingestion import get_router
from backend.models import (
    BlockType,
    ChatMessage,
    Citation,
    DocumentSummary,
    IngestionResult,
    IngestionStatus,
    Language,
    PipelineStage,
    RetrievalResult,
    Role,
    SourceKind,
)
from backend.services import (
    ChatRequest,
    ChatService,
    IngestionService,
    OmniRAGEngine,
    UploadedFile,
    apply_regeneration,
    get_engine,
    plan_regeneration,
)
from backend.storage import FileStore, new_session_id
from backend.utils import (
    SERVICE_NOT_CONFIGURED,
    get_logger,
    public_error_text,
    public_generation_warning,
    public_processing_note,
    short_hash,
)

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  API MODE
# --------------------------------------------------------------------------- #
_API_MODE_CHECKED: bool = False
_API_MODE: bool = False


def _use_api() -> bool:
    """Check if Streamlit should route chat through FastAPI."""
    global _API_MODE_CHECKED, _API_MODE
    if _API_MODE_CHECKED:
        return _API_MODE
    import os
    if os.environ.get("OMNIRAG_API_MODE", "").lower() in ("1", "true", "yes"):
        try:
            from frontend.api_client import is_api_available
            _API_MODE = is_api_available()
        except Exception:
            _API_MODE = False
    _API_MODE_CHECKED = True
    if _API_MODE:
        logger.info("Chat routed through FastAPI backend")
    return _API_MODE


# --------------------------------------------------------------------------- #
#  STATE
# --------------------------------------------------------------------------- #

SESSION_KEY = "omnirag_session_id"
MESSAGES_KEY = "omnirag_messages"
SELECTED_KEY = "omnirag_selected_documents"
PROCESSED_KEY = "omnirag_processed_uploads"
PENDING_KEY = "omnirag_pending_prompt"
EDITING_KEY = "omnirag_editing_message"
ACTION_ERROR_KEY = "omnirag_message_action_error"
GENERATION_KEY = "omnirag_generation"


def _cached_engine() -> OmniRAGEngine:
    return get_engine(get_settings())


def engine() -> OmniRAGEngine:
    return _cached_engine()


def settings() -> AppSettings:
    return get_settings()


def ingestion_service() -> IngestionService:
    return IngestionService(engine())


class _APIChatService:
    """ChatService adapter that routes through FastAPI HTTP."""

    def answer(self, request: ChatRequest) -> ChatMessage:
        from frontend.api_client import chat as api_chat

        history_dicts = [
            {"role": str(m.role.value) if hasattr(m.role, "value") else str(m.role), "content": m.content}
            for m in (request.history or [])
        ]
        resp = api_chat(
            question=request.question,
            session_id=request.session_id or "",
            document_ids=request.document_ids,
            history=history_dicts,
        )
        citations = [
            Citation(
                index=c.get("index", 0),
                chunk_id=c.get("chunk_id", ""),
                document_id=c.get("document_id", ""),
                filename=c.get("filename", ""),
                page_number=c.get("page_number", 1),
                page_label=c.get("page_label", ""),
                block_type=BlockType(c.get("block_type", "text")),
                source_kind=SourceKind(c.get("source_kind", "digital")),
                snippet=c.get("snippet", ""),
                score=c.get("score", 0.0),
                visual_asset_id=c.get("visual_asset_id"),
                visual_media_type=c.get("visual_media_type"),
                uncertain=c.get("uncertain", False),
                confidence=c.get("confidence"),
            )
            for c in resp.get("citations", [])
        ]
        debug = resp.get("debug", {})
        debug["api_backend"] = True
        return ChatMessage(
            role=Role.ASSISTANT,
            content=resp.get("answer", ""),
            citations=citations,
            debug=debug,
            error=resp.get("answer", "") == "The AI service is not configured. Please contact the app administrator.",
        )

    def suggested_prompts(self, session_id: str) -> List[str]:
        from frontend.api_client import get_suggestions
        return get_suggestions(session_id)


def chat_service():
    if _use_api():
        return _APIChatService()
    return ChatService(engine())


def init_state() -> str:
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = new_session_id()
        logger.info("Started a new browser session namespace")
    st.session_state.setdefault(MESSAGES_KEY, [])
    st.session_state.setdefault(SELECTED_KEY, None)
    st.session_state.setdefault(PROCESSED_KEY, set())
    st.session_state.setdefault(PENDING_KEY, None)
    st.session_state.setdefault(EDITING_KEY, None)
    st.session_state.setdefault(ACTION_ERROR_KEY, None)
    st.session_state.setdefault(GENERATION_KEY, {"status": "idle"})
    return st.session_state[SESSION_KEY]


def session_id() -> str:
    return init_state()


def messages() -> List[ChatMessage]:
    return st.session_state.get(MESSAGES_KEY, [])


def add_message(message: ChatMessage) -> ChatMessage:
    st.session_state.setdefault(MESSAGES_KEY, []).append(message)
    stored = st.session_state[MESSAGES_KEY][-1]
    if get_settings().debug_generation:
        logger.info(
            "Generation lifecycle stage=session_state generation_id=%s message_id=%s "
            "stored_chars=%d",
            stored.debug.get("generation_id", "") if stored.debug else "",
            stored.message_id,
            len(stored.content),
        )
    return stored


def clear_messages() -> None:
    st.session_state[MESSAGES_KEY] = []


def replace_messages(msgs: List[ChatMessage]) -> None:
    st.session_state[MESSAGES_KEY] = list(msgs)


def editing_message_id() -> Optional[str]:
    return st.session_state.get(EDITING_KEY)


def set_editing_message(message_id: Optional[str]) -> None:
    st.session_state[EDITING_KEY] = message_id


def set_action_error(message: Optional[str]) -> None:
    st.session_state[ACTION_ERROR_KEY] = message


def take_action_error() -> Optional[str]:
    message = st.session_state.get(ACTION_ERROR_KEY)
    st.session_state[ACTION_ERROR_KEY] = None
    return message


def begin_generation(generation_id: str, user_message_id: str) -> None:
    st.session_state[GENERATION_KEY] = {
        "status": "generating",
        "generation_id": generation_id,
        "user_message_id": user_message_id,
    }


def complete_generation() -> None:
    current = dict(st.session_state.get(GENERATION_KEY) or {})
    current["status"] = "complete"
    st.session_state[GENERATION_KEY] = current


def recover_interrupted_generation() -> Optional[Dict[str, Any]]:
    current = dict(st.session_state.get(GENERATION_KEY) or {})
    if current.get("status") != "generating":
        if current.get("status") == "complete":
            st.session_state[GENERATION_KEY] = {"status": "idle"}
        return None
    current["status"] = "interrupted"
    st.session_state[GENERATION_KEY] = current
    return current


def documents() -> List[DocumentSummary]:
    return engine().registry.list(session_id())


def ready_documents() -> List[DocumentSummary]:
    return engine().registry.ready_documents(session_id())


def selected_document_ids() -> Optional[List[str]]:
    selected = st.session_state.get(SELECTED_KEY)
    available = {d.document_id for d in ready_documents()}
    if selected is None:
        return None
    kept = [d for d in selected if d in available]
    return kept or None


def set_selected_documents(document_ids: Optional[List[str]]) -> None:
    st.session_state[SELECTED_KEY] = document_ids


def already_processed(key: str) -> bool:
    return key in st.session_state.get(PROCESSED_KEY, set())


def mark_processed(key: str) -> None:
    st.session_state.setdefault(PROCESSED_KEY, set()).add(key)


def forget_processed() -> None:
    st.session_state[PROCESSED_KEY] = set()


def set_pending_prompt(prompt: Optional[str]) -> None:
    st.session_state[PENDING_KEY] = prompt


def take_pending_prompt() -> Optional[str]:
    prompt = st.session_state.get(PENDING_KEY)
    st.session_state[PENDING_KEY] = None
    return prompt


def new_chat() -> None:
    clear_messages()


def reset_session() -> None:
    current = st.session_state.get(SESSION_KEY)
    if current:
        try:
            engine().clear_session(current)
        except Exception as exc:
            logger.warning("Session cleanup failed: %s", exc)
    st.session_state[SESSION_KEY] = new_session_id()
    clear_messages()
    forget_processed()
    set_selected_documents(None)


# --------------------------------------------------------------------------- #
#  STYLES
# --------------------------------------------------------------------------- #

CSS = """
<style>
/* ---------- layout ---------- */
.block-container { padding-top: 2.2rem; padding-bottom: 5rem; max-width: 1080px; }
section[data-testid="stSidebar"] { width: 358px !important; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
#MainMenu, footer { visibility: hidden; }

/* ---------- brand ---------- */
.omni-brand { display:flex; align-items:center; gap:.65rem; margin-bottom:.15rem; }
.omni-brand-mark {
  width:36px; height:36px; border-radius:10px; flex:0 0 36px;
  background: linear-gradient(135deg,#6366f1 0%,#0ea5e9 55%,#06b6d4 100%);
  display:flex; align-items:center; justify-content:center;
  color:#fff; font-weight:700; font-size:1rem; letter-spacing:-.02em;
}
.omni-brand-text { display:flex; flex-direction:column; line-height:1.15; }
.omni-brand-title { font-size:1.16rem; font-weight:700; letter-spacing:-.02em; }
.omni-brand-sub { font-size:.72rem; opacity:.62; }

/* ---------- pills ---------- */
.omni-pill {
  display:inline-flex; align-items:center; gap:.32rem;
  padding:.13rem .55rem; border-radius:999px;
  font-size:.7rem; font-weight:600; line-height:1.5;
  border:1px solid transparent; white-space:nowrap;
}
.omni-pill-ok    { background:rgba(16,185,129,.13); color:#059669; border-color:rgba(16,185,129,.28); }
.omni-pill-warn  { background:rgba(245,158,11,.14); color:#b45309; border-color:rgba(245,158,11,.30); }
.omni-pill-err   { background:rgba(239,68,68,.13);  color:#dc2626; border-color:rgba(239,68,68,.28); }
.omni-pill-info  { background:rgba(99,102,241,.13); color:#4f46e5; border-color:rgba(99,102,241,.28); }
.omni-pill-muted { background:rgba(120,120,130,.13); color:#6b7280; border-color:rgba(120,120,130,.24); }

/* ---------- document card ---------- */
.omni-doc { padding:.15rem 0 .3rem 0; }
.omni-doc-name {
  font-size:.86rem; font-weight:600; letter-spacing:-.01em;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
.omni-doc-meta { font-size:.71rem; opacity:.6; margin-top:.1rem; }

/* ---------- source cards ---------- */
.omni-source {
  border:1px solid rgba(128,128,140,.22); border-left:3px solid #6366f1;
  border-radius:9px; padding:.6rem .75rem; margin-bottom:.5rem;
  background:rgba(128,128,140,.045);
}
.omni-source-head {
  display:flex; align-items:center; gap:.45rem;
  flex-wrap:wrap; margin-bottom:.35rem;
}
.omni-source-ref { font-size:.79rem; font-weight:650; letter-spacing:-.01em; }
.omni-source-body {
  font-size:.8rem; line-height:1.55; opacity:.86;
  white-space:pre-wrap; word-break:break-word;
}
.omni-source-uncited { border-left-color:rgba(128,128,140,.4); opacity:.72; }

/* ---------- welcome ---------- */
.omni-hero { padding:2.4rem 0 1.1rem 0; text-align:center; }
.omni-hero h1 { font-size:2.05rem; font-weight:700; letter-spacing:-.035em; margin:0 0 .5rem 0; }
.omni-hero p { font-size:.95rem; opacity:.66; margin:0 auto; max-width:33rem; line-height:1.6; }
.omni-feature {
  border:1px solid rgba(128,128,140,.2); border-radius:11px;
  padding:.85rem .9rem; height:100%;
}
.omni-feature-title { font-size:.83rem; font-weight:650; margin-bottom:.2rem; }
.omni-feature-body { font-size:.76rem; opacity:.65; line-height:1.5; }

/* ---------- Arabic / RTL ---------- */
.omni-rtl { direction:rtl; text-align:right; }
[data-testid="stChatMessageContent"] p:lang(ar) { direction:rtl; text-align:right; }
[data-testid="stChatMessageContent"] { unicode-bidi: plaintext; }

/* ---------- misc ---------- */
.omni-caption { font-size:.72rem; opacity:.55; }
.omni-divider { height:1px; background:rgba(128,128,140,.18); margin:.75rem 0; }
div[data-testid="stChatInput"] textarea { font-size:.92rem; }
</style>
"""


def inject_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  COMPONENTS
# --------------------------------------------------------------------------- #

STATUS_STYLE = {
    IngestionStatus.READY: ("ok", "Ready"),
    IngestionStatus.FAILED: ("err", "Failed"),
    IngestionStatus.DUPLICATE: ("muted", "Duplicate"),
    IngestionStatus.PENDING: ("info", "Queued"),
    IngestionStatus.PARSING: ("info", "Parsing"),
    IngestionStatus.ANALYZING: ("info", "Analysing"),
    IngestionStatus.CHUNKING: ("info", "Chunking"),
    IngestionStatus.EMBEDDING: ("info", "Embedding"),
    IngestionStatus.INDEXING: ("info", "Indexing"),
}

FILE_ICON = {
    "pdf": "📄",
    "docx": "📝",
    "pptx": "📊",
    "txt": "📃",
    "md": "📑",
    "image": "🖼️",
    "unknown": "📁",
}

BLOCK_LABEL = {
    BlockType.TEXT: "Text",
    BlockType.HEADING: "Heading",
    BlockType.OCR_TEXT: "Scanned text",
    BlockType.HANDWRITING: "Handwriting",
    BlockType.IMAGE: "Image",
    BlockType.TABLE: "Table",
    BlockType.CHART: "Chart",
    BlockType.DIAGRAM: "Diagram",
    BlockType.CAPTION: "Caption",
    BlockType.SPEAKER_NOTES: "Speaker notes",
    BlockType.PAGE_SNAPSHOT: "Page scan",
}

SOURCE_LABEL = {
    SourceKind.DIGITAL: "digital text",
    SourceKind.OCR: "scanned text",
    SourceKind.VISION: "visual analysis",
    SourceKind.STRUCTURED: "parsed structure",
    SourceKind.DERIVED: "derived",
}

LANGUAGE_LABEL = {
    Language.ARABIC: "العربية",
    Language.ENGLISH: "English",
    Language.MIXED: "AR/EN",
    Language.UNKNOWN: "",
}


def brand(title: str = "OmniRAG", subtitle: str = "Multimodal document intelligence") -> None:
    st.markdown(
        f"""
        <div class="omni-brand">
          <div class="omni-brand-mark">OR</div>
          <div class="omni-brand-text">
            <span class="omni-brand-title">{html.escape(title)}</span>
            <span class="omni-brand-sub">{html.escape(subtitle)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill(text: str, tone: str = "info") -> str:
    tone = tone if tone in ("ok", "warn", "err", "info", "muted") else "info"
    return f'<span class="omni-pill omni-pill-{tone}">{html.escape(str(text))}</span>'


def pills(items: Iterable[Tuple[str, str]]) -> None:
    markup = " ".join(pill(text, tone) for text, tone in items)
    if markup:
        st.markdown(markup, unsafe_allow_html=True)


def render_error(message: str, *, title: str = "") -> None:
    st.error(f"**{title}**\n\n{message}" if title else message)


def render_warnings(warnings: List[str], *, expanded: bool = False) -> None:
    if not warnings:
        return
    label = f"⚠️ {len(warnings)} note{'s' if len(warnings) > 1 else ''} about processing"
    with st.expander(label, expanded=expanded):
        for warning in warnings:
            st.markdown(f"- {warning}")


def status_pill(status: IngestionStatus) -> str:
    tone, label = STATUS_STYLE.get(status, ("info", str(status).title()))
    return pill(label, tone)


def file_icon(summary: DocumentSummary) -> str:
    return FILE_ICON.get(str(summary.file_type), FILE_ICON["unknown"])


def document_meta_line(summary: DocumentSummary) -> str:
    parts: List[str] = [str(summary.file_type).upper(), summary.size_label]
    if summary.page_count:
        unit = "slide" if str(summary.file_type) == "pptx" else "page"
        parts.append(f"{summary.page_count} {unit}{'s' if summary.page_count != 1 else ''}")
    if summary.chunk_count:
        parts.append(f"{summary.chunk_count} chunks")
    if summary.visual_block_count:
        parts.append(f"{summary.visual_block_count} visuals")
    if summary.table_count:
        parts.append(f"{summary.table_count} tables")
    language = LANGUAGE_LABEL.get(summary.language, "")
    if language:
        parts.append(language)
    return " · ".join(parts)


def rtl_markdown(text: str) -> None:
    st.markdown(text)


def caption(text: str) -> None:
    st.markdown(f'<div class="omni-caption">{html.escape(text)}</div>', unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<div class="omni-divider"></div>', unsafe_allow_html=True)


def empty_state(icon: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div style="text-align:center;padding:1.6rem 0;opacity:.72;">
          <div style="font-size:1.7rem;margin-bottom:.4rem;">{icon}</div>
          <div style="font-weight:620;font-size:.92rem;margin-bottom:.25rem;">{html.escape(title)}</div>
          <div style="font-size:.79rem;opacity:.72;">{html.escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def provider_badge(provider: str, model: str, *, fallback_used: bool = False) -> None:
    if not provider and not model:
        return
    tone = "warn" if fallback_used else "muted"
    label = f"{provider} · {model}" if provider and model else (provider or model)
    suffix = " (fallback)" if fallback_used else ""
    st.markdown(pill(f"{label}{suffix}", tone), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  MESSAGE ACTIONS
# --------------------------------------------------------------------------- #

_COPY_HTML = (
    '<button id="copy" type="button" title="Copy" '
    'aria-label="Copy message">Copy</button>'
    '<span id="status" role="status" aria-live="polite"></span>'
)
_COPY_CSS = """
button { border:0; background:transparent; color:#777; cursor:pointer;
  font:12px system-ui; padding:2px 6px; border-radius:6px; }
button:hover { background:rgba(127,127,127,.12); color:#333; }
#status { color:#56845f; font:11px system-ui; margin-left:4px; }
"""
_COPY_JS = """
export default function(component) {
  const { data, parentElement } = component;
  const button = parentElement.querySelector('#copy');
  const status = parentElement.querySelector('#status');
  const copy = async () => {
    const text = typeof data?.text === 'string' ? data.text : '';
    try {
      await navigator.clipboard.writeText(text);
      status.textContent = 'Copied';
    } catch (_) {
      const area = document.createElement('textarea');
      area.value = text; parentElement.appendChild(area); area.select();
      document.execCommand('copy'); area.remove(); status.textContent = 'Copied';
    }
    setTimeout(() => { status.textContent = ''; }, 1400);
  };
  button.addEventListener('click', copy);
  return () => button.removeEventListener('click', copy);
}
"""


def _register_copy_component():
    return st.components.v2.component(
        "omnirag_copy_button",
        html=_COPY_HTML,
        css=_COPY_CSS,
        js=_COPY_JS,
    )


_COPY_COMPONENT = _register_copy_component()


def copy_component_html(text: str, component_id: str = "copy_message") -> str:
    payload = json.dumps(text, ensure_ascii=False).replace("<", "\\u003c")
    safe_id = html.escape(component_id, quote=True)
    return f"""<!doctype html>
<html><body data-component-id="{safe_id}" style="margin:0;background:transparent">
<button id="copy" type="button" title="Copy" aria-label="Copy message">Copy</button>
<span id="status" role="status" aria-live="polite"></span>
<script>
const text = {payload};
const button = document.getElementById('copy');
const status = document.getElementById('status');
button.addEventListener('click', async () => {{
  try {{
    await navigator.clipboard.writeText(text);
    status.textContent = 'Copied';
  }} catch (_) {{
    const area = document.createElement('textarea');
    area.value = text; document.body.appendChild(area); area.select();
    document.execCommand('copy'); area.remove(); status.textContent = 'Copied';
  }}
  setTimeout(() => status.textContent = '', 1400);
}});
</script>
<style>
button {{ border:0; background:transparent; color:#777; cursor:pointer;
  font:12px system-ui; padding:2px 6px; border-radius:6px; }}
button:hover {{ background:rgba(127,127,127,.12); color:#333; }}
#status {{ color:#56845f; font:11px system-ui; margin-left:4px; }}
</style></body></html>"""


def render_copy_button(*, text: str, key: str) -> None:
    global _COPY_COMPONENT
    kwargs = {
        "key": key,
        "data": {"text": text},
        "width": "content",
        "height": "content",
    }
    try:
        _COPY_COMPONENT(**kwargs)
    except ValueError as exc:
        if "is not registered" not in str(exc):
            raise
        _COPY_COMPONENT = _register_copy_component()
        _COPY_COMPONENT(**kwargs)
    if get_settings().debug_generation:
        logger.info(
            "Generation lifecycle stage=clipboard component_id=%s clipboard_chars=%d",
            key,
            len(text),
        )


def action_key(action: str, message_id: str) -> str:
    return f"{action}_{html.escape(message_id, quote=True)}"


# --------------------------------------------------------------------------- #
#  SOURCES
# --------------------------------------------------------------------------- #

def render_sources(
    citations: Sequence[Citation],
    *,
    used_indices: Optional[set] = None,
    file_store: Optional[FileStore] = None,
    key_prefix: str = "",
    retrieval: Optional[RetrievalResult] = None,
    show_diagnostics: bool = False,
) -> None:
    if not citations:
        return

    cited = used_indices if used_indices is not None else {c.index for c in citations}
    cited_count = len([c for c in citations if c.index in cited])
    label = f"📎 Sources ({cited_count} cited of {len(citations)} retrieved)"

    with st.expander(label, expanded=False):
        if retrieval is not None and show_diagnostics:
            _render_retrieval_note(retrieval)

        for citation in citations:
            _render_source_card(
                citation,
                is_cited=citation.index in cited,
                file_store=file_store,
                key_prefix=key_prefix,
            )


def _render_retrieval_note(retrieval: RetrievalResult) -> None:
    bits: List[str] = [pill(f"{retrieval.strategy} search", "muted")]
    if retrieval.reranked:
        bits.append(pill("reranked", "muted"))
    if retrieval.expanded_queries:
        bits.append(pill(f"+{len(retrieval.expanded_queries)} query variants", "muted"))
    st.markdown(" ".join(bits), unsafe_allow_html=True)
    for note in retrieval.notes:
        caption(note)
    st.write("")


def _render_source_card(
    citation: Citation,
    *,
    is_cited: bool,
    file_store: Optional[FileStore],
    key_prefix: str,
) -> None:
    kind = BLOCK_LABEL.get(citation.block_type, "Text")
    origin = SOURCE_LABEL.get(citation.source_kind, "")

    badges = [pill(kind, "info")]
    if origin:
        badges.append(pill(origin, "muted"))
    if citation.uncertain:
        badges.append(pill("low confidence", "warn"))
    if citation.confidence is not None:
        badges.append(pill(f"{citation.confidence:.0%}", "muted"))
    if not is_cited:
        badges.append(pill("retrieved, not cited", "muted"))

    css_class = "omni-source" if is_cited else "omni-source omni-source-uncited"
    st.markdown(
        f"""
        <div class="{css_class}">
          <div class="omni-source-head">
            <span class="omni-source-ref">[{citation.index}] {html.escape(citation.filename)}
            — {html.escape(citation.page_label)}</span>
            {' '.join(badges)}
          </div>
          <div class="omni-source-body">{html.escape(citation.snippet)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if citation.visual_asset_id and file_store is not None:
        _render_visual(citation, file_store, key_prefix)


def _render_visual(citation: Citation, file_store: FileStore, key_prefix: str) -> None:
    key = f"{key_prefix}_visual_{citation.index}_{citation.chunk_id[:8]}"
    with st.expander("🖼️ View original visual", expanded=False):
        try:
            data = file_store.get(citation.visual_asset_id or "")
        except Exception as exc:
            logger.warning("Could not load visual asset: %s", exc)
            data = None

        if not data:
            st.caption(
                "The original image is no longer available in this session's "
                "temporary storage."
            )
            return
        st.image(
            data,
            caption=f"{citation.filename} — {citation.page_label}",
            width="stretch",
        )


def render_inline_references(citations: Sequence[Citation], used: set) -> None:
    referenced = [c for c in citations if c.index in used]
    if not referenced:
        return
    parts = [
        f"`[{c.index}]` {c.filename} — {c.page_label}" for c in referenced
    ]
    caption(" · ".join(parts))


def group_citations(citations: Sequence[Citation]) -> Dict[str, List[Citation]]:
    grouped: Dict[str, List[Citation]] = {}
    for citation in citations:
        grouped.setdefault(citation.filename, []).append(citation)
    return grouped


# --------------------------------------------------------------------------- #
#  CHAT
# --------------------------------------------------------------------------- #

_FEATURES = [
    ("📄", "Any document", "PDF, scans, Word, PowerPoint, images, Markdown and text."),
    ("👁️", "Sees the visuals", "Charts, diagrams, tables and handwriting — not just extracted text."),
    ("🌍", "Arabic & English", "Ask in one language, search documents written in the other."),
    ("🔎", "Cited answers", "Every claim links back to a file, page and passage."),
]

_FALLBACK_PROMPTS = [
    "Summarize these documents.",
    "What are the key findings?",
    "Explain the charts and diagrams.",
    "لخّص هذه المستندات بالعربية.",
]


def render_chat() -> None:
    interrupted = recover_interrupted_generation()
    history = messages()
    documents = ready_documents()
    action_error = take_action_error()
    if action_error:
        st.error(public_error_text(
            action_error, debug=settings().debug_generation
        ))
    if interrupted:
        st.warning(
            "The previous generation was interrupted by an app rerun. No partial "
            "assistant answer was saved; use Regenerate to retry the prompt."
        )

    if not history:
        _render_welcome(bool(documents))

    for index, message in enumerate(history):
        _render_message(message, index)

    _handle_input(bool(documents))


def _render_welcome(has_documents: bool) -> None:
    st.markdown(
        """
        <div class="omni-hero">
          <h1>Medical RAG Assistant</h1>
          <p>Upload medical documents, research papers, clinical guidelines or reports — then ask
          anything. OmniRAG reads the text <em>and</em> the visuals, and cites
          every answer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "\u26a0\ufe0f **Medical Disclaimer**: This system is for educational and informational "
        "purposes only. It does **not** provide medical advice, diagnosis, or treatment "
        "recommendations. Always consult a qualified healthcare professional for clinical decisions. "
        "For emergencies, call your local emergency number immediately.",
        icon="\u26a0\ufe0f",
    )

    if not has_documents:
        columns = st.columns(4, gap="small")
        for column, (icon, title, body) in zip(columns, _FEATURES):
            with column:
                st.markdown(
                    f"""
                    <div class="omni-feature">
                      <div class="omni-feature-title">{icon} {html.escape(title)}</div>
                      <div class="omni-feature-body">{html.escape(body)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.write("")
        st.info("Upload a document in the sidebar to get started.", icon="👈")
        return

    _render_example_prompts()


def _render_example_prompts() -> None:
    try:
        prompts = chat_service().suggested_prompts(session_id())
    except Exception as exc:
        logger.debug("Could not build suggested prompts: %s", exc)
        prompts = []
    prompts = prompts or _FALLBACK_PROMPTS

    st.markdown("###### Try asking")
    columns = st.columns(2, gap="small")
    for index, prompt in enumerate(prompts[:6]):
        with columns[index % 2]:
            if st.button(prompt, key=f"example_{index}", width="stretch"):
                set_pending_prompt(prompt)
                st.rerun()


def _render_message(message: ChatMessage, index: int) -> None:
    avatar = "🧑" if message.role == Role.USER else "🔷"
    with st.chat_message(message.role.value, avatar=avatar):
        if message.role == Role.USER and editing_message_id() == message.message_id:
            _render_editor(message)
            return
        if message.error:
            render_error(public_error_text(
                message.content,
                debug=settings().debug_generation,
            ))
        else:
            if settings().debug_generation:
                message.debug["rendered_chars"] = len(message.content)
                logger.info(
                    "Generation lifecycle stage=render generation_id=%s message_id=%s "
                    "render_chars=%d finish_reason=%s",
                    message.debug.get("generation_id", ""),
                    message.message_id,
                    len(message.content),
                    message.debug.get("finish_reason", "unspecified"),
                )
            rtl_markdown(message.content)

        if message.role == Role.USER:
            _render_user_actions(message)
            return

        if message.citations:
            render_sources(
                message.citations,
                used_indices=_used_indices(message),
                file_store=engine().file_store,
                key_prefix=f"m{index}",
                retrieval=message.retrieval,
                show_diagnostics=settings().debug_generation,
            )

        _render_message_footer(message)
        _render_assistant_actions(message)


def _render_user_actions(message: ChatMessage) -> None:
    columns = st.columns([0.34, 0.18, 0.22, 0.26], gap="small")
    with columns[0]:
        render_copy_button(
            text=message.content, key=action_key("copy_user", message.message_id)
        )
    with columns[1]:
        if st.button(
            "Edit",
            key=action_key("edit_user", message.message_id),
            help="Edit this prompt",
        ):
            set_editing_message(message.message_id)
            st.rerun()
    with columns[2]:
        if st.button(
            "Regenerate",
            key=action_key("regen_user", message.message_id),
            help="Resend this prompt and replace later turns",
        ):
            _regenerate(message.message_id)


def _render_assistant_actions(message: ChatMessage) -> None:
    columns = st.columns([0.34, 0.24, 0.42], gap="small")
    with columns[0]:
        render_copy_button(
            text=message.content,
            key=action_key("copy_assistant", message.message_id),
        )
    with columns[1]:
        if st.button(
            "Regenerate",
            key=action_key("regen_assistant", message.message_id),
            help="Regenerate this answer from its preceding prompt",
        ):
            _regenerate(message.message_id)


def _render_editor(message: ChatMessage) -> None:
    edited = st.text_area(
        "Edit message",
        value=message.content,
        key=action_key("edit_text", message.message_id),
        label_visibility="collapsed",
    )
    cancel, save, _ = st.columns([0.18, 0.32, 0.5], gap="small")
    with cancel:
        if st.button(
            "Cancel",
            key=action_key("cancel_edit", message.message_id),
            help="Cancel editing",
        ):
            set_editing_message(None)
            st.rerun()
    with save:
        if st.button(
            "Save & Regenerate",
            key=action_key("save_edit", message.message_id),
            help="Save this prompt and regenerate from this point",
        ):
            _regenerate(message.message_id, edited_text=edited)


def _regenerate(message_id: str, *, edited_text: Optional[str] = None) -> None:
    current = list(messages())
    try:
        plan = plan_regeneration(current, message_id, edited_text=edited_text)
        gen_id = uuid.uuid4().hex
        begin_generation(gen_id, plan.user_message_id)
        with st.spinner("Searching your documents again…"):
            answer = chat_service().answer(
                ChatRequest(
                    question=plan.prompt,
                    session_id=session_id(),
                    document_ids=selected_document_ids(),
                    history=plan.history,
                    user_message_id=plan.user_message_id,
                    generation_id=gen_id,
                )
            )
        if answer.error:
            set_action_error(answer.content)
        else:
            replace_messages(apply_regeneration(current, plan, answer))
            set_editing_message(None)
        complete_generation()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Message regeneration failed")
        set_action_error("Could not regenerate this message. Please try again.")
    st.rerun()


def _render_message_footer(message: ChatMessage) -> None:
    debug = message.debug or {}
    diagnostic_mode = settings().debug_generation
    warnings: List[str] = list(debug.get("warnings") or [])
    for warning in warnings:
        public_warning = public_generation_warning(warning, debug=diagnostic_mode)
        if public_warning:
            st.warning(public_warning, icon="⚠️")

    if not debug:
        return

    if diagnostic_mode:
        provider_badge(
            str(debug.get("provider", "")),
            str(debug.get("model", "")),
            fallback_used=_used_fallback(debug),
        )

    bits = []
    if debug.get("contexts"):
        bits.append(f"{debug['contexts']} sources")
    if debug.get("images_sent"):
        count = int(debug["images_sent"])
        bits.append(f"{count} image{'s' if count != 1 else ''} analysed")
    if debug.get("elapsed_s"):
        bits.append(f"{debug['elapsed_s']}s")
    if bits:
        caption(" · ".join(bits))

    if diagnostic_mode:
        with st.expander("Generation diagnostics", expanded=False):
            st.json(_generation_debug_payload(message), expanded=False)


def _generation_debug_payload(message: ChatMessage) -> dict:
    debug = message.debug or {}
    return {
        "Revision": getattr(engine(), "revision", "unknown"),
        "Generation ID": debug.get("generation_id", ""),
        "Message ID": message.message_id,
        "Provider": debug.get("provider", ""),
        "Model": debug.get("model", ""),
        "Query scope": debug.get("query_scope", "FOCUSED"),
        "Requested output tokens": debug.get("requested_output_tokens")
        or debug.get("requested_max_output_tokens"),
        "Finish reason": debug.get("finish_reason", ""),
        "Raw chars": debug.get("provider_raw_chars"),
        "Parsed chars": debug.get("parsed_chars"),
        "Stored chars": len(message.content),
        "Rendered chars": debug.get("rendered_chars", len(message.content)),
        "Continuation count": debug.get("continuation_count", 0),
    }


def _used_fallback(debug: dict) -> bool:
    attempts = debug.get("provider_attempts") or []
    return len(attempts) > 1 or bool(debug.get("openrouter_free_fallback"))


def _used_indices(message: ChatMessage) -> set:
    from backend.rag.generation import parse_markers

    valid = {c.index for c in message.citations}
    return {i for i in parse_markers(message.content) if i in valid}


def _handle_input(has_documents: bool) -> None:
    s = settings()
    placeholder = (
        "Ask anything about your documents…"
        if has_documents
        else "Upload a document first…"
    )

    typed = st.chat_input(placeholder, disabled=not has_documents)
    prompt = typed or take_pending_prompt()

    if not prompt:
        return
    if not has_documents:
        return
    if not s.llm.is_configured:
        st.error(SERVICE_NOT_CONFIGURED)
        return

    user_message = ChatMessage(role=Role.USER, content=prompt)
    add_message(user_message)
    gen_id = uuid.uuid4().hex
    begin_generation(gen_id, user_message.message_id)

    with st.chat_message("user", avatar="🧑"):
        rtl_markdown(prompt)

    with st.chat_message("assistant", avatar="🔷"):
        with st.spinner("Searching your documents…"):
            answer = _answer(prompt, user_message.message_id, gen_id)
    if settings().debug_generation:
        logger.info(
            "Generation lifecycle stage=pre_storage generation_id=%s message_id=%s "
            "chat_message_chars=%d finish_reason=%s",
            gen_id,
            answer.message_id,
            len(answer.content),
            answer.debug.get("finish_reason", "unspecified"),
        )
    add_message(answer)
    complete_generation()
    st.rerun()


def _answer(
    prompt: str,
    user_message_id: Optional[str] = None,
    generation_id: str = "",
) -> ChatMessage:
    service = chat_service()
    hist = [m for m in messages() if m.role in (Role.USER, Role.ASSISTANT)][:-1]
    try:
        return service.answer(
            ChatRequest(
                question=prompt,
                session_id=session_id(),
                document_ids=selected_document_ids(),
                history=hist,
                user_message_id=user_message_id,
                generation_id=generation_id,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error while answering")
        msg = "Something went wrong while answering. Please try again."
        return ChatMessage(role=Role.ASSISTANT, content=msg, error=msg)


# --------------------------------------------------------------------------- #
#  SIDEBAR
# --------------------------------------------------------------------------- #

_STAGE_ORDER = [
    PipelineStage.UPLOADING,
    PipelineStage.PARSING,
    PipelineStage.EXTRACTING_TEXT,
    PipelineStage.ANALYZING_VISUALS,
    PipelineStage.CHUNKING,
    PipelineStage.EMBEDDING,
    PipelineStage.INDEXING,
    PipelineStage.READY,
]


def render_sidebar() -> None:
    with st.sidebar:
        brand()
        divider()
        _render_uploader()
        _render_library()
        divider()
        _render_actions()
        _render_settings()
        _render_footer()


def _render_uploader() -> None:
    router = get_router()
    s = settings()

    st.markdown("##### Upload documents")
    uploads = st.file_uploader(
        "Drop files here",
        type=router.supported_extensions(),
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="omnirag_uploader",
        help=(
            f"Supported: {router.supported_label()} · "
            f"up to {s.upload.max_upload_mb:.0f} MB per file"
        ),
    )

    if not uploads:
        return

    pending = []
    for upload in uploads:
        try:
            data = upload.getvalue()
        except Exception as exc:
            logger.warning("Could not read an upload: %s", exc)
            continue
        key = f"{upload.name}:{len(data)}:{short_hash(data, 12)}"
        if already_processed(key):
            continue
        pending.append((key, UploadedFile(name=upload.name, data=data)))

    if not pending:
        return

    existing = len(documents())
    allowed = max(0, s.upload.max_files - existing)
    if allowed <= 0:
        st.warning(
            f"Document limit reached ({s.upload.max_files}). "
            "Remove a document before uploading more."
        )
        return
    if len(pending) > allowed:
        st.warning(f"Only the first {allowed} file(s) will be processed.")
        pending = pending[:allowed]

    _process_uploads(pending)


def _process_uploads(pending: List[Tuple[str, UploadedFile]]) -> None:
    svc = ingestion_service()
    sid = session_id()
    results: List[IngestionResult] = []

    with st.status(f"Processing {len(pending)} file(s)…", expanded=True) as status_box:
        progress_bar = st.progress(0.0)
        for position, (key, upload) in enumerate(pending):
            label = st.empty()
            label.markdown(f"**{html.escape(upload.name)}**")
            stage_text = st.empty()

            def on_progress(
                stage: PipelineStage,
                value: float,
                message: str = "",
                _pos: int = position,
            ) -> None:
                overall = (_pos + max(0.0, min(1.0, value))) / len(pending)
                progress_bar.progress(min(1.0, overall))
                suffix = f" — {message}" if message else ""
                stage_text.caption(f"{stage.value}{suffix}")

            result = svc.ingest(sid, upload, progress=on_progress)
            results.append(result)
            mark_processed(key)

            if result.status == IngestionStatus.READY:
                stage_text.caption(
                    f"✅ Ready — {result.page_count} page(s), {result.chunk_count} chunks"
                )
            elif result.status == IngestionStatus.DUPLICATE:
                stage_text.caption("↩︎ Already indexed in this session")
            else:
                stage_text.caption(
                    "❌ " + public_error_text(
                        result.error or "Processing failed",
                        debug=settings().debug_generation,
                    )
                )

        progress_bar.progress(1.0)
        failed = [r for r in results if r.status == IngestionStatus.FAILED]
        ready = [r for r in results if r.status == IngestionStatus.READY]

        if failed and not ready:
            status_box.update(label="Processing failed", state="error")
        elif failed:
            status_box.update(
                label=f"{len(ready)} indexed, {len(failed)} failed", state="error"
            )
        else:
            status_box.update(label=f"{len(results)} file(s) ready", state="complete")

    for result in results:
        if result.status == IngestionStatus.FAILED and result.error:
            st.error(public_error_text(
                result.error, debug=settings().debug_generation
            ))
        for warning in result.warnings[:3]:
            st.info(public_processing_note(
                warning, debug=settings().debug_generation
            ), icon="ℹ️")

    st.rerun()


def _render_library() -> None:
    docs = documents()
    st.markdown("##### Document library")

    if not docs:
        empty_state("📚", "No documents yet", "Upload a file to start asking questions.")
        return

    ready = [d for d in docs if d.status == IngestionStatus.READY]
    sel = selected_document_ids()
    active_ids = set(sel) if sel is not None else {d.document_id for d in ready}

    for summary in docs:
        _render_document_row(summary, is_active=summary.document_id in active_ids)

    if len(ready) > 1:
        divider()
        _render_selector(ready, sel)


def _render_document_row(summary: DocumentSummary, *, is_active: bool) -> None:
    columns = st.columns([0.84, 0.16], gap="small")
    with columns[0]:
        marker = "" if is_active or summary.status != IngestionStatus.READY else "  ·  muted"
        st.markdown(
            f"""
            <div class="omni-doc">
              <div class="omni-doc-name">{file_icon(summary)} {html.escape(summary.filename)}</div>
              <div class="omni-doc-meta">{html.escape(document_meta_line(summary))}{marker}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        badges = [status_pill(summary.status)]
        if summary.status == IngestionStatus.READY and not is_active:
            badges.append(pill("not in chat", "muted"))
        st.markdown(" ".join(badges), unsafe_allow_html=True)

    with columns[1]:
        if st.button(
            "✕",
            key=f"remove_{summary.document_id}",
            help="Remove this document",
            width="stretch",
        ):
            _remove_document(summary)

    if summary.error:
        st.caption("⚠️ " + public_error_text(
            summary.error, debug=settings().debug_generation
        ))
    elif summary.warnings:
        with st.expander(f"{len(summary.warnings)} processing note(s)", expanded=False):
            for warning in summary.warnings:
                st.caption("• " + public_processing_note(
                    warning, debug=settings().debug_generation
                ))
    st.write("")


def _remove_document(summary: DocumentSummary) -> None:
    svc = ingestion_service()
    if svc.remove_document(session_id(), summary.document_id):
        st.toast(f"Removed {summary.filename}")
    else:
        st.toast("Could not remove the document", icon="⚠️")
    st.rerun()


def _render_selector(ready: List[DocumentSummary], sel: Optional[List[str]]) -> None:
    by_name = {d.filename: d.document_id for d in ready}
    default = (
        list(by_name)
        if sel is None
        else [name for name, doc_id in by_name.items() if doc_id in set(sel)]
    )

    chosen = st.multiselect(
        "Documents in this chat",
        options=list(by_name),
        default=default,
        key="omnirag_doc_selector",
        help="Retrieval is restricted to the selected documents.",
    )

    chosen_ids = [by_name[name] for name in chosen]
    if not chosen_ids or len(chosen_ids) == len(by_name):
        set_selected_documents(None)
    else:
        set_selected_documents(chosen_ids)


def _render_actions() -> None:
    columns = st.columns(2, gap="small")
    with columns[0]:
        if st.button("💬 New chat", width="stretch", help="Clear the conversation, keep documents"):
            new_chat()
            st.rerun()
    with columns[1]:
        if st.button("🗑️ Clear all", width="stretch", help="Remove every document and message"):
            reset_session()
            st.rerun()

    if documents():
        if st.button(
            "🔄 Re-index documents",
            width="stretch",
            help="Rebuild the search index from the uploaded files",
        ):
            _reindex()


def _reindex() -> None:
    svc = ingestion_service()

    with st.status("Re-indexing…", expanded=True) as box:
        progress_bar = st.progress(0.0)
        label = st.empty()

        def on_progress(position: int, total: int, filename: str) -> None:
            progress_bar.progress(min(1.0, position / max(1, total)))
            if filename:
                label.caption(f"Rebuilding {filename} ({position + 1}/{total})")

        report = svc.reindex(session_id(), progress=on_progress)
        progress_bar.progress(1.0)
        label.empty()

        if report.ok and report.reindexed:
            box.update(label=f"Re-indexed {len(report.reindexed)} document(s)", state="complete")
        elif report.reindexed:
            box.update(
                label=f"Re-indexed {len(report.reindexed)}, {len(report.missing_source) + len(report.failed)} need attention",
                state="error",
            )
        else:
            box.update(label="Nothing could be re-indexed", state="error")

    if report.missing_source:
        st.warning(
            "The original file is no longer in this session's temporary storage "
            "for: " + ", ".join(report.missing_source) + ". Please upload them again.",
            icon="⚠️",
        )
        forget_processed()
    for filename, error in report.failed:
        safe_error = public_error_text(
            str(error), debug=settings().debug_generation
        )
        st.error(f"{filename}: {safe_error}")

    st.rerun()


def _render_settings() -> None:
    eng = engine()
    status = eng.status()

    diagnostic_mode = settings().debug_generation
    label = "⚙️ Diagnostics" if diagnostic_mode else "⚙️ System status"
    with st.expander(label, expanded=not status.ready):
        if not status.ready:
            if diagnostic_mode:
                for issue in status.issues:
                    st.error(issue, icon="🔑")
            else:
                st.error(SERVICE_NOT_CONFIGURED, icon="🔑")
        else:
            st.success("Ready")

        if diagnostic_mode:
            st.markdown("**Internal services**")
            st.markdown(
                "\n".join(
                    [
                        f"- **LLM chain:** `{status.llm_chain}`",
                        f"- **Embeddings:** `{status.embedding_provider}` / `{status.embedding_model}`",
                        f"- **Vector store:** `{status.vector_store}`",
                        f"- **Reranker:** `{status.reranker}`",
                        f"- **OCR:** `{status.ocr_provider}`",
                        f"- **Vision:** {'available' if status.vision_available else 'unavailable'}",
                    ]
                )
            )
            stats = eng.provider_stats()
            if stats.get("calls"):
                st.markdown("**Provider usage**")
                usage = ", ".join(
                    f"`{key}`: {value}" for key, value in stats["by_provider"].items()
                )
                st.caption(f"{stats['calls']} call(s) — {usage}")
                if stats.get("failovers"):
                    st.caption(f"↪︎ {stats['failovers']} failover(s)")
            for warning in status.warnings:
                st.warning(warning, icon="⚠️")
        else:
            st.caption(
                "Document search is available."
                if status.ready
                else "Document search is waiting for administrator configuration."
            )
            st.caption(
                "Visual analysis is available."
                if status.vision_available
                else "Visual analysis is currently unavailable."
            )
            s = settings()
            if s.embedding.provider == "hash":
                st.warning(
                    "Document search is running in reduced-quality mode.", icon="⚠️"
                )
            if not s.vector_store.use_qdrant:
                st.caption("The document index resets when the app restarts.")


def _render_footer() -> None:
    divider()
    caption(
        "Documents stay in this browser session only. The AI service receives "
        "only the content needed to answer your questions."
    )


# --------------------------------------------------------------------------- #
#  __all__
# --------------------------------------------------------------------------- #

__all__ = [
    # state
    "add_message",
    "already_processed",
    "begin_generation",
    "chat_service",
    "clear_messages",
    "complete_generation",
    "documents",
    "engine",
    "forget_processed",
    "ingestion_service",
    "init_state",
    "mark_processed",
    "messages",
    "new_chat",
    "ready_documents",
    "recover_interrupted_generation",
    "replace_messages",
    "reset_session",
    "selected_document_ids",
    "session_id",
    "set_pending_prompt",
    "editing_message_id",
    "set_editing_message",
    "set_action_error",
    "take_action_error",
    "set_selected_documents",
    "settings",
    "take_pending_prompt",
    # styles
    "CSS",
    "inject_styles",
    # components
    "BLOCK_LABEL",
    "SOURCE_LABEL",
    "brand",
    "caption",
    "divider",
    "document_meta_line",
    "empty_state",
    "file_icon",
    "pill",
    "pills",
    "provider_badge",
    "render_error",
    "render_warnings",
    "rtl_markdown",
    "status_pill",
    # message actions
    "action_key",
    "copy_component_html",
    "render_copy_button",
    # sources
    "group_citations",
    "render_inline_references",
    "render_sources",
    # chat
    "render_chat",
    # sidebar
    "render_sidebar",
]
