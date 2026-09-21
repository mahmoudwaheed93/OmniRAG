"""OmniRAG — Streamlit entry point.

Deliberately thin: it bootstraps configuration, sets up the page, and delegates
to the UI modules. All domain logic lives under ``backend/`` and knows nothing
about Streamlit.

Run locally:
    streamlit run main.py
"""

from __future__ import annotations

import streamlit as st

# Page config must be the first Streamlit call in the script.
st.set_page_config(
    page_title="OmniRAG — Document Intelligence",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "about": (
            "**OmniRAG** — Universal Multimodal Document Intelligence & RAG.\n\n"
            "Upload documents and ask questions. Answers are grounded in your "
            "files and cite the exact page they came from."
        )
    },
)

from backend.config import bootstrap_environment  # noqa: E402

# Copies .env and st.secrets into the environment BEFORE settings are read.
bootstrap_environment()

from frontend import ui  # noqa: E402
from backend.utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    ui.inject_styles()
    ui.init_state()

    try:
        ui.render_sidebar()
    except Exception as exc:  # noqa: BLE001 - a sidebar fault must not blank the app
        logger.exception("Sidebar rendering failed")
        st.sidebar.error(
            "The sidebar could not be displayed. Reload the page to try again."
        )

    try:
        ui.render_chat()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Main panel rendering failed")
        st.error(
            "Something went wrong while rendering the page. Reload the page to continue."
        )


if __name__ == "__main__":
    main()
