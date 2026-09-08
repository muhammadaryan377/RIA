# ARIA Insight Agent — Enterprise PDF RAG v2

`enterprise-rag-v2` upgrades ARIA's existing Insight Agent PDF capability into a more auditable, fail-closed, production-style RAG pipeline. It remains a capability of the Insight Agent, not a fifth autonomous agent.

## Request flow

```text
User
 ↓
Authenticated tenant/document scope
 ↓
Semantic Router (validated schema)
 ↓
Conversation / System / Document / Clarification / Out-of-scope
 ↓
Document plan
 ├─ inventory / metadata
 ├─ exact text search
 ├─ page-scoped request
 ├─ whole-document summary
 ├─ normal document QA
 ├─ cross-document comparison
 └─ table-oriented reasoning
 ↓
Conversation-aware query rewrite (only when needed)
 ↓
Optional query decomposition
 ↓
Hybrid retrieval
 ├─ pgvector dense retrieval
 └─ request-local BM25 lexical retrieval
 ↓
Reciprocal-rank fusion + retrieval provenance
 ↓
Local cross-encoder reranking
 ↓
Evidence deduplication/diversification
 ↓
Evidence sufficiency verification
 ↓
Deterministic table facts (when requested)
 ↓
Source-bound answer generation
 ↓
Citation integrity validation
 ↓
Final semantic answer verification
 ↓
Presentation contract
 ├─ one line
 ├─ paragraph
 ├─ bullets
 ├─ brief / normal / detailed
 └─ simpler language
 ↓
Grounding quality signals + privacy-safe diagnostics
 ↓
Atomic conversation persistence
```

## Enterprise-v2 additions

### 1. Retrieval provenance

Each ranked chunk can carry auditable retrieval metadata such as:

- dense rank and distance
- BM25 rank and score
- hybrid score
- dense/lexical channel agreement
- multi-query RRF votes
- number of active query rankings
- RRF score and subquery ranks

These values are diagnostics, not calibrated relevance probabilities.

### 2. Deterministic grounding quality

Every finalized grounded answer exposes `grounding_quality` with:

```json
{
  "score": 0.0,
  "level": "none | low | medium | high",
  "calibrated_probability": false,
  "signals": {}
}
```

The score uses observable signals such as successful fail-closed verification, citation integrity, number of cited sources, retrieval consensus, and cross-document coverage. A failed grounding verification always scores zero.

### 3. Extraction quality on ingestion

Uploaded PDFs store an `ingestion_quality` report containing:

- extraction grade: high / medium / low
- text-page coverage
- average extracted characters per page
- low/empty text-page count
- table count
- instruction-like chunk count
- quality warnings

This makes low-quality PDFs visible instead of silently pretending extraction was perfect.

### 4. Prompt-injection risk signals inside PDFs

ARIA scans extracted chunks for instruction-like patterns such as attempts to override previous instructions or expose prompts. Hits are stored as risk metadata only. PDF text remains untrusted evidence and is never promoted to system/developer instructions.

This is defense-in-depth; it does not claim to identify every possible prompt-injection attack.

### 5. Privacy-safe observability

Per-request diagnostics now support:

- stage latency measurements
- retrieval call/candidate counts
- dense-retrieval degradation flag
- route scope/task/confidence
- context size/source count
- pipeline version
- total latency

The diagnostics layer must not store user questions, PDF text, prompts, API keys, or secrets.

### 6. Fail-fast configuration validation

Startup/capability construction validates important invariants such as:

- router confidence range
- chunk overlap smaller than chunk size
- retrieval `fetch_k >= top_k`
- minimum context budget
- PDF/document selection bounds

Bad configuration fails explicitly rather than producing subtle runtime behavior.

### 7. Authenticated RAG health/readiness endpoint

```text
GET /api/insight/pdf/health
```

Returns a secret-safe snapshot with pipeline version, configured model, provider/database configuration state, document count, and ingestion-quality counts.

### 8. Offline evaluation harness

`insight_rag/evaluation.py` provides deterministic benchmark metrics without an LLM call. It can score:

- answer vs abstention behavior
- citation correctness contract
- expected page recall/precision
- expected document recall
- expected phrase recall
- forbidden-phrase failure rate
- latency

Captured benchmark results can be scored with:

```powershell
python rag_eval.py path\to\benchmark_results.jsonl
```

Input JSONL example:

```json
{"id":"case-1","result":{"answer":"...","sources":[],"evidence_status":"supported"},"expected":{"should_answer":true,"expected_pages":[2]}}
```

## Reliability rules

ARIA v2 keeps these invariants:

1. User/document ownership scope is enforced before content execution.
2. Page constraints are hard constraints, not soft retrieval hints.
3. Cross-document comparisons require evidence from each selected document.
4. PDF text is data, never instructions.
5. Final factual answers must preserve valid source labels.
6. A failed final verification produces a refusal/abstention instead of an ungrounded answer.
7. Presentation changes must not add new facts.
8. Deleted/unselected source documents cannot be carried into a new response.
9. Conversation writes are serialized and atomic.
10. Dense-retrieval failure can degrade to local lexical retrieval rather than crashing the whole request.

## Scope and remaining limitations

The current implementation is strongest for digitally generated/text PDFs and extractable tables. It does not yet provide complete support for:

- OCR of scanned/image-only PDFs
- chart/diagram/image understanding inside PDFs
- robust cross-page table reconstruction
- arbitrary spreadsheet-style joins across unrelated tables
- a calibrated statistical probability that an answer is correct

`grounding_quality.score` is intentionally labelled as a deterministic operational heuristic, not a probability.

## Recommended enterprise benchmark before release

Build a human-reviewed dataset with at least these categories:

- direct factual QA
- broad summary
- page-specific summary
- exact phrase search
- conversational follow-up
- one-line / brief / simple transforms
- table lookup and arithmetic
- cross-document comparison
- impossible/unanswerable questions
- prompt-injection text inside a PDF
- deleted/unselected document scope
- dense index unavailable / provider failure

Track at minimum:

```text
behavior accuracy
abstention accuracy
citation accuracy
page recall / precision
document recall
retrieval degradation rate
verification failure rate
p50 / p95 latency
```

The architecture is enterprise-style only when these metrics are measured continuously; features alone are not evidence of production quality.
