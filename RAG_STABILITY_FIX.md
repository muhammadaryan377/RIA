# RAG Groq stability fix

This branch addresses two live issues observed in the Insight PDF-RAG logs:

1. Groq structured outputs may return `null` for optional array fields such as `target_pages`. The runtime now accepts that provider representation and normalizes it to an empty list before ARIA's route validation.
2. Groq TPM rate-limit responses can request a retry delay longer than five seconds. The runtime now respects the server retry hint (with a small safety margin), falls back to bounded exponential waiting when no hint is exposed, and still fails closed after a bounded number of attempts.

The semantic routing policy, retrieval stack, grounding checks, citation validation, and selected model are unchanged.

Regression coverage is in `tests/test_rag_groq_stability.py`.
