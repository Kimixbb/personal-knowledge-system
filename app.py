from __future__ import annotations

from pathlib import PurePosixPath

import streamlit as st

from personal_rag.chunking import MultilingualChunker
from personal_rag.config import ConfigurationError, Settings, ensure_vault
from personal_rag.embeddings import Qwen3Embeddings
from personal_rag.file_actions import (
    DuplicateFileError,
    UnsafePathError,
    UnsupportedUploadError,
    open_vault_file,
    save_upload,
)
from personal_rag.manifest import Manifest
from personal_rag.providers import available_providers, create_chat_model
from personal_rag.rag_service import RAGResult, RAGService
from personal_rag.retrieval import Retriever
from personal_rag.synchronizer import (
    IndexConfigurationMismatch,
    SynchronizationError,
    Synchronizer,
)
from personal_rag.vector_store import ChromaVectorStore


st.set_page_config(page_title="Personal Knowledge Vault", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Preparing the local retrieval index…")
def build_runtime(settings: Settings):  # type: ignore[no-untyped-def]
    embeddings = Qwen3Embeddings(
        settings.embedding_model,
        revision=settings.embedding_revision,
        dimensions=settings.embedding_dimensions,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    manifest = Manifest(settings.manifest_path)
    vector_store = ChromaVectorStore(
        settings.chroma_dir, embeddings, settings.collection_name
    )
    chunker = MultilingualChunker(
        settings.chunk_size,
        settings.chunk_overlap,
        model_name=settings.embedding_model,
        model_revision=settings.embedding_revision,
    )
    synchronizer = Synchronizer(
        settings, manifest, vector_store, chunker
    )
    retriever = Retriever(
        vector_store,
        top_k=settings.retrieval_top_k,
        minimum_score=settings.retrieval_min_score,
    )
    return manifest, synchronizer, RAGService(synchronizer, retriever), embeddings


def show_sync_result(result) -> None:  # type: ignore[no-untyped-def]
    if result.errors:
        st.error("Synchronization did not complete safely. Questions are blocked.")
        for error in result.errors:
            st.write(f"• {error}")
        return
    st.success(
        "Synchronization complete — "
        f"{result.added} added, {result.changed} changed, "
        f"{result.deleted} deleted, {result.skipped} unchanged, "
        f"{result.empty} empty."
    )


def render_result(result: RAGResult, settings: Settings) -> None:
    st.subheader("Answer")
    if result.hosted_error == "MissingCitation":
        st.error(
            "The hosted response was rejected because it did not contain a valid "
            "source citation such as [S1]. The original response is retained in "
            "Retrieval Debug."
        )
    elif result.hosted_error:
        st.error(f"The hosted request failed ({result.hosted_error}). Retrieval debug is retained below.")
    elif result.answer:
        st.markdown(result.answer)
        st.caption(f"{result.provider} · {result.model}")
    if result.invalid_citations:
        st.warning(
            "The hosted model returned invalid source IDs, which were removed: "
            + ", ".join(result.invalid_citations)
        )

    if result.passages:
        st.subheader("Sources")
        for passage in result.passages:
            filename = PurePosixPath(passage.relative_path).name
            page_label = f" · page {passage.page}" if passage.page is not None else ""
            with st.container(border=True):
                st.markdown(f"**[{passage.source_id}] {filename}**{page_label}")
                st.caption(passage.relative_path)
                st.write(passage.content)
                if st.button(
                    "Open File",
                    key=f"open-{passage.chunk_id}-{passage.rank}",
                ):
                    try:
                        open_vault_file(passage.relative_path, settings.vault_path)
                    except (OSError, UnsafePathError) as exc:
                        st.error(f"Could not open this source ({type(exc).__name__}).")

    if result.passages or result.exact_hosted_context:
        with st.expander("Retrieval Debug"):
            rows = [
                {
                    "rank": passage.rank,
                    "score": passage.score,
                    "chunk_id": passage.chunk_id,
                    "relative_path": passage.relative_path,
                    "page": passage.page,
                }
                for passage in result.passages
            ]
            if rows:
                st.dataframe(rows, hide_index=True, use_container_width=True)
                for passage in result.passages:
                    st.markdown(f"**[{passage.source_id}] Full retrieved chunk**")
                    st.code(passage.content, language=None)
            st.markdown("**Exact hosted context**")
            st.code(result.exact_hosted_context or "No hosted request was made.", language=None)
            st.markdown("**Hosted model response (before citation validation)**")
            st.code(
                result.hosted_response_text or "No hosted response was received.",
                language=None,
            )


try:
    settings = Settings.from_env()
    ensure_vault(settings)
except ConfigurationError as exc:
    st.title("Personal Knowledge Vault")
    st.error(str(exc))
    st.info("Copy .env.example to .env and set KNOWLEDGE_VAULT before starting the app.")
    st.stop()
except OSError as exc:
    st.error(f"The vault could not be prepared ({type(exc).__name__}).")
    st.stop()

try:
    manifest, synchronizer, rag_service, embeddings = build_runtime(settings)
except Exception as exc:
    st.error(f"The local index could not be opened ({type(exc).__name__}).")
    st.stop()


st.title("Personal Knowledge Vault")
st.caption("Local retrieval for Chinese and English PDF, TXT, and Markdown files")

availability = available_providers(settings)
configured_providers = [item.provider for item in availability if item.configured]
provider_labels = {
    "deepseek": "DeepSeek",
    "kimi": "Kimi / Moonshot",
    "openai": "OpenAI",
    "openai-compatible": "OpenAI-compatible",
}
model_defaults = {
    "deepseek": "deepseek-chat",
    "kimi": "kimi-k2.5",
    "openai": "gpt-4.1-mini",
    "openai-compatible": settings.default_llm_model,
}

with st.sidebar:
    st.header("Knowledge Vault")
    st.caption("Vault path")
    st.code(str(settings.vault_path), language=None)
    first, second = st.columns(2)
    first.metric("Documents", manifest.count_documents())
    second.metric("Chunks", manifest.count_chunks())
    st.caption(f"Last synchronization: {manifest.get_state('last_sync') or 'Never'}")
    st.caption(f"Embedding device: {embeddings.active_device.upper()}")

    st.header("Hosted Model")
    if configured_providers:
        default_index = 0
        if settings.default_llm_provider in configured_providers:
            default_index = configured_providers.index(settings.default_llm_provider)
        provider = st.selectbox(
            "Provider",
            configured_providers,
            index=default_index,
            format_func=lambda value: provider_labels[value],
        )
        default_model = (
            settings.default_llm_model
            if provider == settings.default_llm_provider
            else model_defaults[provider]
        )
        model = st.text_input(
            "Model", value=default_model, key=f"model-name-{provider}"
        )
    else:
        provider = ""
        model = ""
        st.warning("Configure at least one provider API key in .env.")

    with st.expander("Provider availability"):
        for item in availability:
            marker = "✓" if item.configured else "—"
            detail = "Configured" if item.configured else item.reason
            st.write(f"{marker} {provider_labels[item.provider]}: {detail}")

    st.header("Actions")
    if st.button("Refresh Index", use_container_width=True):
        with st.spinner("Synchronizing the vault…"):
            try:
                show_sync_result(synchronizer.sync_library())
            except IndexConfigurationMismatch as exc:
                st.error(str(exc))

    confirm_rebuild = st.checkbox("I understand rebuild removes and recreates all vectors")
    if st.button(
        "Rebuild Index",
        disabled=not confirm_rebuild,
        use_container_width=True,
    ):
        with st.spinner("Rebuilding the local index…"):
            show_sync_result(synchronizer.rebuild_index())


st.header("Import Files")
uploads = st.file_uploader(
    "Choose PDF, TXT, or Markdown files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

if "pending_uploads" not in st.session_state:
    st.session_state.pending_uploads = {}

if st.button("Import selected files", disabled=not uploads):
    imported: list[str] = []
    rejected: list[str] = []
    for upload in uploads or []:
        try:
            saved = save_upload(
                upload.getvalue(),
                upload.name,
                settings.inbox_dir,
                settings.temp_dir,
            )
            imported.append(saved.filename)
        except DuplicateFileError as exc:
            st.session_state.pending_uploads[exc.filename] = upload.getvalue()
        except (UnsupportedUploadError, UnsafePathError, OSError) as exc:
            rejected.append(f"{upload.name}: {type(exc).__name__}")
    if imported:
        with st.spinner("Indexing imported files…"):
            sync_result = synchronizer.sync_library()
        st.success("Imported: " + ", ".join(imported))
        show_sync_result(sync_result)
    for item in rejected:
        st.error(item)

for filename in list(st.session_state.pending_uploads):
    st.warning(f"{filename} already exists")
    replace_column, cancel_column, _ = st.columns([1, 1, 3])
    if replace_column.button("Replace existing file", key=f"replace-{filename}"):
        try:
            save_upload(
                st.session_state.pending_uploads[filename],
                filename,
                settings.inbox_dir,
                settings.temp_dir,
                replace=True,
            )
            with st.spinner("Replacing and re-indexing the document…"):
                sync_result = synchronizer.sync_library()
            del st.session_state.pending_uploads[filename]
            if sync_result.successful:
                st.success(f"Replaced and synchronized {filename}.")
            else:
                st.error(
                    f"{filename} was replaced, but indexing failed. Questions are blocked."
                )
            show_sync_result(sync_result)
        except Exception as exc:
            st.error(f"Replacement failed ({type(exc).__name__}).")
    if cancel_column.button("Cancel", key=f"cancel-{filename}"):
        del st.session_state.pending_uploads[filename]
        st.rerun()


st.header("Ask a Question")
with st.form("question-form"):
    question = st.text_area(
        "Question",
        placeholder="Ask independently in Chinese or English…",
        height=100,
    )
    ask_clicked = st.form_submit_button(
        "Ask",
        disabled=not configured_providers,
        use_container_width=True,
    )

if ask_clicked and question.strip():
    try:
        with st.spinner("Synchronizing the vault before retrieval…"):
            llm = create_chat_model(provider, model, settings)
            result = rag_service.ask(
                question,
                llm,
                provider=provider,
                model=model,
            )
        st.session_state.last_result = result
    except SynchronizationError as exc:
        st.error("The vault could not be synchronized, so no hosted request was made.")
        show_sync_result(exc.result)
    except IndexConfigurationMismatch as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"The question could not be processed ({type(exc).__name__}).")

if "last_result" in st.session_state:
    render_result(st.session_state.last_result, settings)
