# Evaluation procedure

1. Use a disposable external vault and copy `fixture_vault/library` into it.
2. Start Personal RAG and rebuild the index.
3. Ask each question in `evaluation_set.jsonl` as an independent question.
4. Mark a Recall@5 hit when every required `expected_paths` entry appears among the five retrieved paths. For absent-information cases, verify that the app refuses rather than calling unsupported facts an answer.
5. Record answer faithfulness, citation correctness, same-language behavior, and the expected fact manually.
6. For q29, edit the birthday in `english-notes.md`, save, and ask without refreshing manually. Confirm the retrieved chunks contain only the new date.
7. For q30, delete `old-plan.txt`, ask without refreshing manually, and confirm neither its path nor its old fact appears.

Recall@5 is `questions with the required passage in the top five / applicable questions`. Keep conflicts as separate qualitative checks because both old and current passages are intentionally relevant.
