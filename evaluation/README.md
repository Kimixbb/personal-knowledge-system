# Automated fixture benchmark

Run all 30 synthetic-fixture cases from the repository root:

```powershell
.\.venv313\Scripts\python.exe -m personal_rag.evaluation
```

The command uses the model, revision, device, dimensions, chunking, and minimum score from `.env`, while forcing `top_k=5` for the benchmark. It creates a disposable vault outside the repository, copies `fixture_vault/library`, rebuilds the Chroma index, synchronizes before every question, and runs the current hybrid dense-plus-BM25 retriever. It does not use a provider API key or send fixture text to a hosted model.

The q29 `replace_text` and q30 `delete` setup actions are declared in `evaluation_set.jsonl` and applied automatically. Their forbidden text and path checks catch stale chunks. Every result includes ranks, paths, scores, chunk IDs, and retrieved content in a timestamped JSON report under the git-ignored `evaluation/results` directory.

The command exits `0` when all automated checks pass and `1` when any fails. Recall@5 is calculated over 26 applicable cases: absent-information cases q25/q26 and conflicting-source cases q27/q28 are excluded. The conflict cases still require both paths in the top five. q25/q26 appear as `REVIEW` because refusal quality requires an answer model.

Use `--output <path>` to choose the report location, `--top-k <number>` for exploratory retrieval runs, or `--work-dir <new-path>` to retain the generated benchmark vault. A supplied work directory must not already exist.

Hosted answer faithfulness, citation correctness, same-language behavior, refusal behavior, and conflict resolution remain manual checks in the Streamlit app. The expected facts and retrieved passages in the JSON report support that review.

## Personal-vault regressions

`personal_vault_evaluation_set.jsonl` records retrieval failures found against the user-owned vault. Run these separately from the synthetic fixture benchmark because their source documents are not committed to this repository. For each case, use the configured personal vault, rebuild after any chunking change, ask the question independently, and record whether the expected path and fact appear in the retrieved passages and grounded answer.

For hybrid retrieval cases, record the final rank plus the dense and normalized keyword component scores shown in Retrieval Debug. These measurements evaluate first-stage score blending only; no reranker is present.
