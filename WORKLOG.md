# Worklog

## 2026-08-01 — Personal RAG MVP implementation

- Task/phase: Build the local-first Windows personal knowledge RAG MVP from an empty repository.
- Progress: Added pinned Python project configuration; vault validation and scanning; PDF/TXT/Markdown extraction; multilingual token-aware chunking; lazy normalized Qwen3 document/query embeddings with CUDA fallback; persistent Chroma operations; SQLite manifest; add/change/delete synchronization; incompatible-index protection and rebuild; provider factory; strict grounded prompting; stateless RAG service; secure upload/replace/open-file actions; Streamlit interface; unit tests; documentation; and a 30-question bilingual evaluation fixture.
- Decisions: Keep model/tokenizer loading lazy; embed new chunks before deleting stale vectors; block questions after any synchronization error; store only safe error types; remove invalid model-generated source IDs; locally refuse when retrieval has no useful passages; and keep all provider-specific options inside the factory.
- Blockers: The bundled environment is Python 3.12 rather than the required Python 3.11, and network package installation timed out, so dependency-backed tests and live Qwen/hosted-provider checks still need verification in a Python 3.11 environment.
- Next steps: Resolve/install the pinned dependencies, run the complete unit suite, fix any compatibility issues, then perform a live CUDA/CPU Qwen-Chroma smoke test and provider-key acceptance checks.

## 2026-08-01 — Verification follow-up

- Task/phase: Verify the completed MVP against installed integrations and the actual Streamlit runtime.
- Progress: Installed the lightweight pinned runtime/test integrations; reached 41 passing tests; verified persistent Chroma add, idempotent retry, replace, cosine retrieval, delete, and rebuild; constructed real DeepSeek, Moonshot, OpenAI, and OpenAI-compatible LangChain chat models without network invocation; checked the installed environment with `pip check`; and executed the Streamlit application with no rendered errors against a disposable external vault.
- Decisions: Pin the Qwen model to immutable revision `d43997c8a1046d1734f8d519effbb424a832a0a2`; match the model card's exact `Instruct`/`Query` format; refuse to display hosted answers that contain no valid supplied source citation; and make Chroma replacement retries idempotent after partial failures.
- Blockers: The available verification interpreter is Python 3.12, while the shipped project correctly requires Python 3.11. PyTorch, Transformers, Sentence Transformers, the Qwen model weights, and live hosted API calls were not exercised in this environment.
- Next steps: On the target Windows machine, install the complete pinned project under Python 3.11, verify CUDA selection (and CPU fallback), index the synthetic evaluation vault, run Recall@5 checks, and perform user-key provider acceptance tests.

## 2026-08-01 — Python 3.13 compatibility

- Task/phase: Expand the supported interpreter range so the application can use the machine's current Python 3.13 installation.
- Progress: Added a project-metadata compatibility test; changed `requires-python` from `>=3.11,<3.12` to `>=3.11,<3.14`; updated setup/run/test commands; created `.venv313`; installed the entire pinned dependency graph under Python 3.13.0; passed all 42 tests; passed `pip check`; and ran the Streamlit application without rendered errors against a disposable external vault.
- Decisions: Continue supporting Python 3.11 and 3.12 while adding 3.13, with 3.14 intentionally excluded until separately verified. Ignore all `.venv*` directories so version-specific local environments cannot enter Git.
- Blockers: The installed PyTorch 2.13.0 wheel is CPU-only (`torch.cuda.is_available()` is false). Qwen model-weight loading and hosted calls still require their first live run and user-owned API keys.
- Next steps: Run the app from `.venv313`, configure `.env`, index the synthetic or personal vault, and perform live retrieval/provider acceptance checks. Install an appropriate CUDA-enabled PyTorch build later if GPU acceleration is desired.

## 2026-08-02 — Hosted-response retrieval debug

- Task/phase: Make citation-validation failures diagnosable without weakening grounded-answer enforcement.
- Progress: Preserved the hosted model's normalized text before citation cleanup; exposed that response in Retrieval Debug for accepted and rejected answers; replaced the misleading `MissingCitation` API-failure message with a validation-specific explanation; updated the README; and added regression coverage for successful cleanup, hosted failures, and uncited responses. The complete Python 3.13 suite passes with 42 tests.
- Decisions: Continue suppressing uncited answers from the main answer panel, retain provider text only in the in-memory latest result, and display it only inside the existing debug expander. Do not write hosted responses or personal text to logs.
- Blockers: The running Streamlit process uses `server.fileWatcherType=none`, so it must be restarted before the UI change appears.
- Next steps: Restart Streamlit, repeat the question that produced `MissingCitation`, and inspect **Hosted model response (before citation validation)** to determine whether the model omitted citations, used a different format, or returned an evidence-insufficiency refusal.
