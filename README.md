# Personal RAG

Personal RAG is a local-first Windows application for searching a private PDF, TXT, and Markdown vault. Files, extracted text, embeddings, the SQLite manifest, and Chroma stay on the PC. Only the current question, fixed grounding instructions, and the top retrieved passages are sent to the selected hosted chat model.

The MVP is deliberately stateless: every question synchronizes the vault, retrieves from the current index, and creates a new two-message hosted request. Earlier questions and answers are never included.

## Requirements

- Windows 10 or 11
- Python 3.11, 3.12, or 3.13 (64-bit)
- About 2–4 GB of free space for packages, model files, and the index
- Optional NVIDIA GPU with a working CUDA-compatible PyTorch installation
- At least one DeepSeek, Moonshot/Kimi, OpenAI, or OpenAI-compatible API key

OCR is not included. Image-only/scanned PDFs are recorded as empty and are retried only after the file changes.

## Install

Open PowerShell in this repository:

```powershell
python --version
python -m venv .venv313
.\.venv313\Scripts\python.exe -m pip install --upgrade pip
.\.venv313\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Edit `.env` locally. At minimum, set an external vault and one hosted-provider key:

```dotenv
KNOWLEDGE_VAULT=C:\PersonalKnowledgeVault
DEEPSEEK_API_KEY=your-key-here
```

Do not commit `.env`. API keys are marked secret in the configuration object and are never placed in prompts, logs, chunks, or the Streamlit debug panel.

Start the application:

```powershell
.\.venv313\Scripts\python.exe -m streamlit run app.py
```

The first document indexing operation downloads `Qwen/Qwen3-Embedding-0.6B`. The application requests CUDA and automatically falls back to CPU when CUDA is unavailable. Confirm the active device in the sidebar. If the pinned PyTorch wheel does not match your NVIDIA setup, install the matching Windows CUDA wheel from the official PyTorch selector, then rerun the application.

The default embedding revision is an immutable Hugging Face commit rather than the moving `main` branch. Changing that revision is treated as incompatible and requires a full rebuild, preventing vectors from different model snapshots from being mixed.

## Vault contract

The application creates this structure outside the repository:

```text
C:\PersonalKnowledgeVault\
├── library\
│   └── inbox\
└── .rag\
    ├── chroma\
    ├── manifest.sqlite3
    ├── temp\
    └── logs\
```

Put supported documents anywhere under `library`; File Explorer additions are detected recursively before the next question. Browser uploads go to `library\inbox`. `.rag` is application-managed.

Symlinks are not followed, hidden files are ignored, extension matching is case-insensitive, and only `.pdf`, `.txt`, and `.md` are indexed. PDF pages are extracted and split independently so one-based page citations remain precise.

## Using the app

- **Refresh Index** scans for added, edited, and deleted files.
- **Rebuild Index** is required when the embedding model/revision, dimensions, collection, or chunking configuration changes.
- Browser imports are staged and verified in `.rag\temp`. Existing names require an explicit **Replace existing file** or **Cancel** choice; files are never automatically renamed.
- Every non-empty question runs synchronization before retrieval. A changed-document failure blocks the hosted request so stale evidence is not returned unknowingly.
- **Retrieval Debug** shows ranks, cosine relevance scores, deterministic chunk IDs, full passages, relative paths, pages, and the exact two message bodies sent to the provider.
- **Open File** validates the citation path inside `library` before asking Windows to open it with the default application.

No raw document text is written to application logs. The SQLite manifest stores fingerprints, status, counts, safe error types, and timestamps—not extracted passages.

## Providers

Only fully configured providers appear in the selection list:

| Provider | LangChain model | Required configuration |
|---|---|---|
| DeepSeek | `ChatDeepSeek` | `DEEPSEEK_API_KEY` |
| Kimi/Moonshot | `ChatMoonshot` | `MOONSHOT_API_KEY` |
| OpenAI | `ChatOpenAI` | `OPENAI_API_KEY` |
| OpenAI-compatible | `ChatOpenAI` | `OPENAI_COMPATIBLE_API_KEY` and `OPENAI_COMPATIBLE_BASE_URL` |

Kimi-specific thinking and temperature constraints remain inside the provider factory. Retrieval and prompting use only the common `llm.invoke(messages)` interface.

## Tests

```powershell
.\.venv313\Scripts\python.exe -m pytest
```

The unit suite uses fake embeddings, vector storage, and chat models, so it does not need model weights or API keys. It covers loaders, multilingual chunking, the SQLite manifest, new/change/delete synchronization, the August 1 → August 7 stale-data regression, provider switching, upload safety, retrieval filtering, grounded context, invalid citations, hosted failure debug, and question statelessness.

Live Qwen/Chroma and hosted-provider checks require the normal application dependencies, model download, suitable hardware, and user-owned keys. They are intentionally not invoked by the unit test suite.

## Evaluation

[`evaluation/evaluation_set.jsonl`](evaluation/evaluation_set.jsonl) contains 30 manually checked bilingual scenarios against the included synthetic fixture vault. Copy `evaluation/fixture_vault/library` into a disposable vault, index it, then ask the questions independently. Record whether an expected path appears in the top five (`Recall@5`) and assess answer faithfulness, citation correctness, response language, refusal behavior, conflicts, edits, deletions, and stale-data removal.

## MVP boundaries

There is no OCR, media transcription, EPUB support, automatic categorization, file movement, agent workflow, LangGraph, conversation memory, hybrid search, reranking, document history, cloud sync, or multi-user permission system.
