# ARIA PDF RAG: reliability and retrieval upgrade

This update builds on `muhammadaryan377/RIA`, branch `insight-rag-pdf-v1`,
commit `488e0837444f2ee9311456cb308e32a06f40d698`. The uploaded `ARIA-fyp (2).zip`
was an older schema-agent snapshot with an empty Insight Agent package; it was
reviewed for context, not used as the implementation base.

The public entry point remains `from insight_rag import InsightPDFRAG`.
FastAPI continues to run with `python app.py`; PDF chat remains `/pdf-chat`.
The Schema, Goal, prediction and existing Industry Insight Agent interfaces
remain connected through the original application.

## What the audit found and what changed

| Finding in the original code | Resulting behavior |
| --- | --- |
| Page numbers only affected sorting; other pages could still enter context. | Page and document constraints are hard filters. Missing pages return insufficient evidence. |
| Broad summaries operated on a small semantic retrieval set. | Summary evidence is sampled from the complete persisted chunk corpus, spread across the beginning, middle and end, and balanced across PDFs. The answer includes a coverage note when only some indexed pages are represented. |
| Lexical search used token overlap and substring boosts. | BM25 adds term frequency, inverse document frequency and length normalization, without another search dependency. |
| Dense retrieval failure blocked local lexical search. | Owned local chunks can still supply BM25 evidence; diagnostics identify the fallback. Verification remains mandatory. |
| RRF discarded candidates before the cross-encoder; repeated subquery hits consumed slots. | A larger candidate pool reaches reranking. Subquery lists are fused and chunk identities deduplicated first. |
| Identical passages in different PDFs could collapse to one source. | Deduplication preserves document/page provenance, allowing both PDFs to be cited. |
| High word overlap, any page scope, and verifier outages could bypass the evidence check. | QA requires an exact positive semantic verdict. Outages abstain with a distinct status. Summaries also receive a post-generation answer check. |
| Removing a fabricated citation left its factual claim in the answer. | Unknown or missing citation markers fail closed. A second semantic check tests every claim against its cited evidence and deterministic table facts. |
| Final citation cleanup happened after saving the answer; previous-answer paths bypassed cleanup. | Every route is finalized once at the public boundary, then that exact answer and source list are saved as one atomic turn. |
| Simplification sometimes became clarification. | More explicit semantic routing guidance plus one bounded previous-answer repair attempt. No phrase-list intent classifier was added. |
| Transformations could add numbers/claims or carry sources despite failure. | Transformations get a fidelity check. Failures clear carried sources; deleted or unselected source PDFs cannot be silently reused. |
| A failed follow-up rewrite fell back to an ambiguous question. | Failed rewrites request the full question instead of searching a lost subject/date context. |
| Large first chunks could exceed the context budget. | Text is bounded and marked as truncated; table chunks are never cut into malformed rows. |
| Reranking could discard large-table siblings before arithmetic. | All chunks of each cited table are restored for calculation after evidence selection. New uploads record table part counts and positions. |
| `drop_duplicates()` removed legitimate repeated table rows. | Only repeated chunk identities are removed. Repeated business rows count separately. |
| Partial/unparseable values could produce deceptively complete totals. | Incomplete chunk sets, missing numeric cells, malformed headers/rows, ambiguous comma formats, mixed currencies, mixed percentage units and explicit subtotal/total aggregation cases are withheld. Percentage means are labeled unweighted. |
| Metadata read/modify/write and two-message persistence could race. | Manifest updates and full turns use file locks and atomic replacement. Same-conversation requests serialize. Full transcripts are retained; working context is bounded. |
| Failed ingestion could leave vector or local artifacts behind. | Failed writes attempt vector rollback and remove temporary local document/chunk artifacts. Rollback failure is logged for index repair. |
| Corrupt JSON looked like an empty store. | Corruption produces a controlled error instead of silently overwriting stored state. |

## Run on your Windows computer

From your existing repository, with your own local work committed or saved:

```powershell
git fetch origin
git switch --track origin/improve-rag-reliability
python -m pip install -r requirements.txt
python app.py
```

If the local branch already exists, use `git switch improve-rag-reliability`.
Open the URL printed by the application, then visit `/pdf-chat`.

Keep your existing `.env`, database, uploaded PDFs and account configuration.
For a fresh checkout, copy `.env.example` to `.env` and fill in your own values:

```dotenv
GROQ_API_KEY=your-own-key
ARIA_RAG_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/aria_rag
```

PostgreSQL must have pgvector available. `filelock>=3.13` is the only newly
added package requirement. Embedding/reranker models still need their first
download unless already cached. This update does not run a database migration.

Existing indexed PDFs remain readable. Table chunk completeness metadata is
available for new uploads; to refresh an older document, keep the original PDF,
delete its indexed copy through the app, and upload the original again. No
existing user PDFs or conversations were changed during this development work.

## Practical acceptance checks

Use a text PDF with a small known table and, for comparison, a second PDF.

| Ask / action | Expected behavior |
| --- | --- |
| `hi` / `who are you?` | Conversation / runtime-profile route; no PDF retrieval. |
| `How many PDFs do I have?` | Authoritative metadata count. |
| `Summarize my PDF` | Broad page sampling; coverage note when applicable. |
| `Summarize page 4` | Only page 4 sources, or an insufficient-evidence response. |
| `What was revenue in 2025?` then `and 2024?` | Standalone follow-up search with subject/year resolved. |
| `Explain that simply` / `make it shorter` | Previous-answer transformation with fidelity and citation checks. |
| `Where did you find that?` | Sources used by the saved previous answer; no new retrieval. |
| `Find "data science lifecycle"` | Literal normalized-text lookup. |
| `Compare these PDFs` | Evidence from every requested PDF, or an explicit incomplete-comparison response. |
| `Sum Sales` for rows A=100, A=100 | Repeated rows contribute 200, not 100. |
| Remove a cited PDF, then `repeat that` | Scope-change response rather than reusing removed evidence. |
| Stop pgvector after indexing, then ask a matching term | BM25 fallback can retrieve local evidence; cloud verification is still required. |
| Ask a fact absent from the document | Abstain when evidence/claim verification fails. |

## Validation performed

- Original RAG test baseline: **35 passed**.
- Updated full repository suite: **241 passed**; includes **78 RAG tests**
  (43 newly added behavioral regressions).
- Real application wiring and an authenticated PDF-chat endpoint path tested.
- `python app.py` subprocess smoke: `/api/health` and `/pdf-chat` respond HTTP 200.
- Actual uploaded proposal extraction: **17 pages, 31 text chunks, 13 table
  chunks, 44 total chunks**. Extraction output is not a manual table-accuracy audit.
- Compilation and whitespace/diff checks performed.

Reproduce the automated checks after installing requirements:

```powershell
python -m pip install pytest
python -m pytest -q
```

Tests use model/vector doubles where external services would be required.
They establish execution, scope, persistence and failure-handling contracts;
they do **not** establish live Groq answer accuracy, real embedding/reranker
quality, PostgreSQL integration performance, or production load capacity.
No configured live Groq + pgvector end-to-end run was performed here.
The existing FastAPI deprecation and pandas date-parsing warnings are unrelated
to this RAG update.

## Scope and tradeoffs

This is a tested improvement to the existing PDF RAG, not a claim that all
advanced RAG features are complete or that hallucinations are impossible.

- LLM evidence/claim verification is fallible and normally adds two short
  verification calls for factual QA. Conservative checks can refuse otherwise
  answerable questions. Broad summaries use one post-generation check.
- Summaries use bounded page excerpts, not exhaustive hierarchical map/reduce
  synthesis of every passage. Broad comparisons covering more than the evidence
  budget can abstain and ask for fewer PDFs.
- OCR/scanned PDFs, image/chart understanding, cross-page table reconstruction,
  exact decimal financial arithmetic, arbitrary grouping/joins and weighted
  percentage aggregation remain outside this version.
- Calculations describe the extracted rows of one table, not an entire PDF or
  business dataset. Locale-specific numeric formats beyond supported formats
  need explicit preprocessing. Table extraction still needs human QA for complex
  layouts. Existing table header/column matching remains conservative.
- BM25 scans the selected local corpus per request. This is appropriate for a
  modest FYP corpus; a persisted search index and measured retrieval benchmark
  are needed before claiming large-corpus scalability.
- File locks protect processes sharing the same filesystem. This is not a
  distributed transaction across PostgreSQL and multiple application hosts.
  If vector rollback also fails, operational cleanup is still necessary.
- Full transcript JSON is retained on disk. Context passed to routing/rewrite
  is bounded; there is no semantic long-term memory summarizer in this change.

Responses expose `request_id`, `latency_ms`, `grounding`, `diagnostics`, and
summary `coverage` where applicable. These fields are for inspection/evaluation;
user-facing prose stays focused on the PDF answer.
