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

## 2026-08-02 — Stale Streamlit result compatibility

- Task/phase: Prevent a live Streamlit session from crashing when it retains a pre-change `RAGResult` object without the new hosted-response field.
- Progress: Made Retrieval Debug tolerate legacy session-state results and show a restart-and-retry explanation instead of raising `AttributeError`.
- Decisions: Preserve the user's most recent visible result across ordinary reruns while treating newly added diagnostic fields as optional during an in-process code upgrade.
- Blockers: A complete process restart remains necessary to reload the updated `RAGService` module and capture hosted response text for new questions.
- Next steps: Stop the existing Streamlit server with `Ctrl+C`, start it again, and repeat the question.

## 2026-08-02 — Personal-vault chunking evaluation

- Task/phase: Reduce dense-retrieval chunk dilution and preserve the reported Charlie query as a manual personal-vault regression rather than a unit test.
- Progress: Changed the default and active chunk configuration from 700/100 to 300/50 tokens; documented the new rebuild requirement; added a separate `personal_vault_evaluation_set.jsonl`; rebuilt the CUDA-backed Chroma collection cleanly; and verified 4 indexed documents, 4,636 manifest chunks, 4,636 vectors, valid JSONL, and 42 passing unit tests.
- Decisions: Keep the synthetic 30-case evaluation fixture unchanged; store user-owned corpus regressions separately; use the smallest values inside the agreed 300–400/50–75 range after 350/60 left a 248-word multi-topic chunk; and report retrieval measurements without promoting the example into the unit suite.
- Blockers: The exact question still does not retrieve the page-70 Charlie passage in the top 200, so smaller chunks alone do not resolve this semantic mismatch.
- Next steps: Treat `pv01` as a red manual evaluation and, when authorized, test the next recall strategy such as query expansion, adjacent-chunk retrieval, or hybrid lexical-plus-dense search.

## 2026-08-02 — Hybrid BM25 retrieval

- Task/phase: Combine the existing Qwen/Chroma similarity search with keyword/BM25 retrieval, without adding a reranking stage.
- Progress: Added a lazy in-memory Okapi BM25 index over the current Chroma chunks, English possessive and CJK-aware tokenization, automatic keyword-cache invalidation after vector mutations, equal normalized dense/keyword score fusion, five-times candidate collection from each channel, separate component scores in Retrieval Debug, documentation, and generic unit coverage. All 46 tests pass. On the personal-vault Charlie evaluation, the target passage ranks 32nd with dense retrieval, 9th with BM25, and 7th with the configured hybrid retrieval, so it is included in the app's 20-passage answer context.
- Decisions: Keep hybrid retrieval transparent and deterministic: normalize BM25 against the strongest keyword candidate, blend dense and keyword scores equally, and do not add a reranker. Keep the Charlie wording only in the manual personal-vault evaluation set.
- Correction: The previous entry's statement that the Charlie passage was absent from the dense top 200 was caused by a Windows console curly-apostrophe detection error. ASCII-safe chunk-ID validation confirms the dense rank is 32.
- Blockers: None.
- Next steps: Observe answer quality and broader manual evaluation behavior before deciding whether another retrieval strategy is warranted.

## 2026-08-02 — Live question-processing status

- Task/phase: Replace the question form's single synchronization spinner with accurate live feedback for the complete RAG pipeline.
- Progress: Added typed stage notifications at the actual RAG service boundaries; connected them to a compact Streamlit status display; identified the selected provider and model while waiting for the hosted answer; and added specific completion states for grounded answers, insufficient retrieval, provider failures, citation-validation failures, synchronization failures, and index-configuration failures.
- Decisions: Keep stage reporting optional and side-effect free outside the GUI, and report only real operation boundaries rather than simulated percentages.
- Blockers: None.
- Next steps: Restart or refresh the running Streamlit app and ask a question to confirm that each live stage is visible with the configured provider.

## 2026-08-02 — Precise import and synchronization status

- Task/phase: Replace generic import, replacement, refresh, and rebuild spinners with accurate live synchronization feedback.
- Progress: Added typed synchronization events for index verification, vault scanning, per-file checks, hashing, CPU extraction, CPU tokenization/chunking, embedding and vector writes, manifest recording, deleted-document cleanup, rebuild reset, and finalization. The Streamlit status now includes the current file number and name, real chunk counts, the active CPU/CUDA device, and whether the lazy embedding model must first be loaded. Added focused regression coverage and verified all 55 tests pass.
- Decisions: Report only real synchronous boundaries rather than simulated percentages; keep unchanged-file scans concise by omitting stages that do not run; and preserve the separately committed question-processing status behavior.
- Blockers: A running Streamlit process would require one restart to load the new synchronization callback API. No app was listening during final validation.
- Next steps: Start Streamlit and import a document to visually confirm the live labels against CPU, disk, and GPU activity.

## 2026-08-02 - Kimi model-family compatibility

- Task/phase: Diagnose Kimi K2.5/K2.6 failures and make the hosted-model factory compatible with Kimi K3.
- Progress: Confirmed live calls to K2.5, K2.6, and K3 all fail authentication with the configured key; updated Kimi request construction so K2.5/K2.6 use non-thinking mode without overriding API-managed temperature while K3 receives no K2-only parameters; changed the Kimi UI default to `kimi-k3`; documented the model-family behavior; and added regression coverage against both fake and installed `ChatMoonshot` integrations. All 58 tests pass.
- Decisions: Treat authentication and model request schemas as separate concerns. Follow Moonshot's model matrix by omitting explicit temperature for all current Kimi models and omitting `thinking` for always-reasoning K3. Keep retrieval, prompting, and indexing unchanged.
- Blockers: The configured Moonshot key receives HTTP 401 `Invalid Authentication` from the global endpoint. The China endpoint could not be reached from the verification environment, so a regional-key mismatch could not be ruled out.
- Next steps: Replace or reissue the Moonshot API Platform key, or confirm whether it belongs to the China platform and configure the matching endpoint before repeating live provider acceptance checks.

## 2026-08-02 - DeepSeek default model

- Task/phase: Switch the DeepSeek default model to `deepseek-v4-flash`.
- Progress: Updated the Settings fallback, Streamlit DeepSeek model fallback, and `.env.example`; added regression coverage for the configuration default. The full test suite passes with 59 tests.
- Decisions: Leave the user’s existing `.env` unchanged because it already specifies `deepseek-v4-flash`.
- Blockers: None.
- Next steps: None.
