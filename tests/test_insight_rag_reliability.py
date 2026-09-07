"""Behavioral regressions: real orchestration/storage, deterministic model doubles.

These tests do not require a cloud key, model download, or PostgreSQL. They test
failure handling and source contracts, not live model answer accuracy.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from insight_rag import InsightPDFRAG
from insight_rag.context_engine import ContextEngineer, ContextPlan
from insight_rag.friendly import InsightPDFRAG as GroundedRAG
from insight_rag.grounding import citation_integrity, verify_answer
from insight_rag.retrieval import bm25_search
from insight_rag.semantic_router import SemanticRouter
from insight_rag.storage import ConversationStore, RAGMetadataStore, UserPGVectorStore
from insight_rag.table_reasoner import build_table_facts, markdown_to_frame

DID = "document-a"


def document(text="Revenue was 100 in 2025.", *, page=1, did=DID, cid=None, **meta):
    return Document(page_content=text, metadata={
        "document_id": did, "filename": f"{did}.pdf", "page": page,
        "chunk_id": cid or f"{did}-p{page}", "content_type": "text", **meta,
    })


def route(**overrides):
    return dict(scope="DOCUMENT", task="DOCUMENT_QA", confidence=0.99,
                document_keys=["D1"], **overrides)


class Model:
    provider = "cloud"

    def __init__(self, routes=None, answer="Revenue was 100 in 2025 [S1].", verify="SUPPORTED"):
        self.models = {}
        self.routes = list(routes or [route()])
        self.answer = answer
        self.verify = verify
        self.calls = []

    def chat(self, role, messages, **kwargs):
        self.calls.append((role, messages))
        if role == "rag_scope":
            return json.dumps(self.routes.pop(0))
        if role == "rag_verify":
            if isinstance(self.verify, Exception):
                raise self.verify
            return self.verify
        if role == "rag_rewrite":
            return "What was revenue in 2024?"
        return self.answer


@pytest.fixture
def make_rag(tmp_path, monkeypatch):
    import insight_rag.config as config
    monkeypatch.setattr(config, "RAG_USERS_DIR", tmp_path)

    def factory(model=None, docs=None, user="test-user"):
        model = model or Model()
        rag = InsightPDFRAG(insight_agent=SimpleNamespace(llm=model), user_id=user)
        docs = docs if docs is not None else [document()]
        for did in dict.fromkeys(doc.metadata["document_id"] for doc in docs):
            scoped = [doc for doc in docs if doc.metadata["document_id"] == did]
            rag.metadata.put_document({"document_id": did, "filename": f"{did}.pdf",
                                       "pages": max(d.metadata["page"] for d in scoped)})
            rag.metadata.save_chunks(did, scoped)
        # QA retrieval is stubbed at the vector boundary; page/summary paths are real.
        monkeypatch.setattr(rag, "_retrieve", lambda q, ids, **kw: [d for d in docs if d.metadata["document_id"] in ids])
        monkeypatch.setattr("insight_rag.reranker.LocalCrossEncoderReranker._get_model", lambda: None)
        return rag
    return factory


def test_page_scope_excludes_strong_off_page_match():
    selected = ContextEngineer.select_evidence("revenue", [document(page=1), document(page=4)],
        selected_document_ids=[DID], page_numbers=[4])
    assert [doc.metadata["page"] for doc in selected] == [4]


def test_scope_selection_excludes_other_owner_document():
    assert ContextEngineer.select_evidence("revenue", [document(did="other-user")], selected_document_ids=[DID]) == []


def test_missing_page_does_not_fall_back_to_another_page(make_rag):
    rag = make_rag(Model([route(target_pages=[99])]))
    result = rag.chat("summarize page 99")
    assert result["evidence_status"] == "insufficient"
    assert not result["sources"]
    assert not any(role == "rag" for role, _ in rag.llm.calls)


def test_page_summary_does_not_need_vector_store(make_rag, monkeypatch):
    rag = make_rag(Model([dict(scope="DOCUMENT", task="DOCUMENT_SUMMARY", confidence=.99,
                               document_keys=["D1"], target_pages=[4])]), docs=[document(page=4)])
    monkeypatch.setattr(rag, "_retrieve", lambda *a, **kw: pytest.fail("page scope used vectors"))
    result = rag.chat("summarize page 4")
    assert result["retrieval"] == "local_corpus"
    assert {s["page"] for s in result["sources"]} == {4}


def test_summary_samples_beginning_middle_and_end():
    docs = [document(f"Page content {i}", page=i) for i in range(1, 101)]
    selected = ContextEngineer.select_evidence("summary", docs, selected_document_ids=[DID], broad_query=True)
    pages = {d.metadata["page"] for d in selected}
    assert len(pages) == 8 and 1 in pages and 100 in pages
    assert any(40 < p < 60 for p in pages)


def test_identical_text_keeps_independent_cross_document_provenance():
    selected = ContextEngineer.select_evidence("compare", [document(), document(did="document-b")],
        selected_document_ids=[DID, "document-b"], cross_document=True)
    assert len(selected) == 2


def test_summary_comparison_balances_documents():
    docs = [document(f"Page {i}", page=i, did=did) for did in [DID, "document-b"] for i in range(1, 21)]
    selected = ContextEngineer.select_evidence("summaries", docs, selected_document_ids=[DID, "document-b"],
        broad_query=True, cross_document=True)
    for did in [DID, "document-b"]:
        pages = {d.metadata["page"] for d in selected if d.metadata["document_id"] == did}
        assert len(pages) == 4 and 1 in pages and 20 in pages


def test_bm25_matches_identifiers_and_prefers_concise_relevant_chunk():
    docs = [document("CRISP-DM " + "filler " * 300, cid="long"), document("CRISP-DM methodology", cid="short")]
    ranked = bm25_search("CRISP-DM", docs, limit=5)
    assert ranked[0].metadata["chunk_id"] == "short"
    assert bm25_search("unmatched", docs, limit=5) == []


def test_dense_outage_preserves_lexical_retrieval(make_rag, monkeypatch):
    from insight_rag.service import InsightPDFRAG as Base
    rag = make_rag()
    monkeypatch.setattr("insight_rag.service.UserPGVectorStore", lambda *a: (_ for _ in ()).throw(RuntimeError("offline")))
    found = Base._retrieve(rag, "revenue", [DID])
    assert found and found[0].metadata["document_id"] == DID


def test_empty_explicit_vector_scope_does_not_query_store():
    store = UserPGVectorStore.__new__(UserPGVectorStore)
    assert store.search("anything", k=5, document_ids=[]) == []


def test_evidence_keyword_overlap_is_not_automatic_support(make_rag):
    rag = make_rag(Model(verify="UNSUPPORTED"))
    result = rag.chat("What was revenue in 2025?")
    assert result["evidence_status"] == "insufficient"
    assert not any(role == "rag" for role, _ in rag.llm.calls)


@pytest.mark.parametrize("verdict", ["SUPPORTED maybe", "UNSUPPORTED", RuntimeError("offline")])
def test_verifier_fails_closed(verdict):
    assert verify_answer(Model(verify=verdict), question="q", answer="answer [S1]", context="[S1] evidence") != "verified"


@pytest.mark.parametrize("answer", ["Profit doubled [S99].", "Profit doubled."])
def test_invalid_answer_is_not_saved_as_supported(make_rag, answer):
    rag = make_rag(Model(answer=answer))
    result = rag.chat("What was revenue?")
    saved = rag.conversations.load(result["conversation_id"])["messages"]
    assert result["evidence_status"] == "verification_failed"
    assert result["sources"] == []
    assert saved[-1]["content"] == result["answer"] and saved[-1]["sources"] == []
    assert len(saved) == 2


def test_answer_entailment_failure_blocks_real_but_wrong_citation(make_rag):
    model = Model(answer="Revenue was 999 [S1].")
    original = model.chat
    count = 0
    def chat(role, messages, **kwargs):
        nonlocal count
        if role == "rag_verify":
            count += 1
            return "SUPPORTED" if count == 1 else "UNSUPPORTED"
        return original(role, messages, **kwargs)
    model.chat = chat
    result = make_rag(model).chat("What was revenue?")
    assert result["evidence_status"] == "verification_failed"
    assert "999" not in result["answer"]


def test_previous_answer_sources_are_final_and_do_not_retrieve(make_rag, monkeypatch):
    model = Model(routes=[route(), dict(scope="CONVERSATION", task="PREVIOUS_ANSWER", confidence=.99,
                                       previous_action="SOURCES")])
    rag = make_rag(model)
    first = rag.chat("What was revenue?")
    monkeypatch.setattr(rag, "_retrieve", lambda *a, **kw: pytest.fail("previous answer retrieved"))
    second = rag.chat("Where did you find that?", conversation_id=first["conversation_id"])
    assert second["sources"] == first["sources"]
    assert second["citation_status"] == "cited"
    assert second["retrieval"] == "not_used"
    assert len(rag.conversations.load(first["conversation_id"])["messages"]) == 4


def test_deleted_pdf_cannot_be_reused_by_repeat(make_rag):
    model = Model(routes=[route(), dict(scope="CONVERSATION", task="PREVIOUS_ANSWER", confidence=.99,
                                       previous_action="REPEAT")])
    rag = make_rag(model)
    first = rag.chat("What was revenue?")
    rag.metadata.remove_document(DID)
    second = rag.chat("repeat that", conversation_id=first["conversation_id"])
    assert second["evidence_status"] == "scope_changed"
    assert not second["sources"]


def test_invalid_selected_pdf_does_not_expand_scope(make_rag):
    with pytest.raises(ValueError, match="unavailable"):
        make_rag().chat("revenue", document_ids=["missing-pdf"])


def test_semantic_repair_recovers_transform_without_keywords():
    model = Model(routes=[dict(scope="CLARIFICATION", task="CLARIFY", confidence=.9),
                          dict(scope="CONVERSATION", task="PREVIOUS_ANSWER", confidence=.99,
                               previous_action="TRANSFORM", transform_instruction="Simplify")])
    result = SemanticRouter(model).decide("Explain that simply", history=[{"role": "assistant", "content": "Earlier answer"}],
        documents=[], selected_document_ids=None)
    assert result.previous_action == "TRANSFORM"
    assert not result.requires_retrieval


def test_context_budget_holds_even_for_first_oversized_chunk(make_rag, monkeypatch):
    monkeypatch.setattr("insight_rag.service.MAX_CONTEXT_CHARS", 300)
    context, sources = make_rag()._build_context([document("x" * 2000)])
    assert len(context) <= 300
    assert sources[0]["context_truncated"] is True


def table(text, **meta):
    return document(text, content_type="table", table_index=1, **meta)


def test_table_preserves_repeated_business_rows_but_deduplicates_chunks():
    doc = table("| Product | Sales |\n| --- | --- |\n| A | 100 |\n| A | 100 |")
    facts = build_table_facts("sales", [doc, doc], operations=["SUM", "COUNT"])
    assert "Sales sum=200" in facts and "Sales count=2" in facts


def test_table_rejects_incomplete_chunk_set():
    doc = table("| Product | Sales |\n| --- | --- |\n| A | 100 |", table_chunk_count=2, table_chunk_index=0)
    facts = build_table_facts("sales", [doc], operations=["SUM"])
    assert "incomplete" in facts and "sum=" not in facts


def test_table_siblings_are_restored_after_evidence_selection(make_rag):
    docs = [table(f"| Product | Sales |\n| --- | --- |\n| A{i} | 10 |", cid=f"table-part-{i}",
                  table_chunk_index=i, table_chunk_count=10) for i in range(10)]
    rag = make_rag(docs=docs)
    restored = rag._expand_table_siblings(docs[:1])
    facts = build_table_facts("sales", restored, operations=["SUM"])
    assert "Sales sum=100" in facts


@pytest.mark.parametrize("values", [("100", "n/a"), ("10%", "20"), ("$10", "€20"), ("nan", "100")])
def test_ambiguous_table_cells_do_not_produce_aggregate(values):
    doc = table(f"| Product | Sales |\n| --- | --- |\n| A | {values[0]} |\n| B | {values[1]} |")
    assert "sum=" not in build_table_facts("sales", [doc], operations=["SUM"])


def test_duplicate_headers_are_rejected_instead_of_crashing():
    assert markdown_to_frame("| Sales | Sales |\n| --- | --- |\n| 1 | 2 |") is None


def test_concurrent_turn_writes_preserve_transcript_and_bound_context(make_rag):
    store = make_rag().conversations
    def append(i):
        store.append_turn("conversation-test", question=f"q{i}", answer=f"a{i}", sources=[],
                          user_metadata={}, assistant_metadata={})
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(append, range(50)))
    assert len(store.load("conversation-test")["messages"]) == 100
    messages = store.load_recent("conversation-test")["messages"]
    assert len(messages) == 80
    for i in range(0, 80, 2):
        assert messages[i]["content"][1:] == messages[i + 1]["content"][1:]


def test_corrupt_manifest_is_not_silently_overwritten(make_rag):
    rag = make_rag()
    rag.metadata.manifest_path.write_text("broken")
    with pytest.raises(RuntimeError, match="Stored RAG"):
        rag.metadata.put_document({"document_id": "new-doc-1"})
    assert rag.metadata.manifest_path.read_text() == "broken"


def test_users_cannot_read_each_others_conversations(make_rag):
    first = make_rag(user="alice")
    second = make_rag(user="bob")
    result = first.chat("What was revenue?", conversation_id="shared-id")
    assert result["sources"]
    assert second.conversations.load("shared-id")["messages"] == []


def test_failed_ingestion_rolls_back_vector_and_local_artifacts(make_rag, tmp_path, monkeypatch):
    rag = make_rag(docs=[])
    pdf = tmp_path / "new.pdf"
    pdf.write_bytes(b"%PDF-1.4\nexample")
    deleted = []
    def extract(path, *, document_id, filename):
        doc = document(did=document_id)
        return [doc], dict(pages=1, tables=0, text_chunks=1, table_chunks=0, total_chunks=1)
    monkeypatch.setattr("insight_rag.service.extract_pdf_documents", extract)
    monkeypatch.setattr("insight_rag.service.UserPGVectorStore", lambda *a: SimpleNamespace(
        add_documents=lambda docs: None, delete_chunks=lambda ids: deleted.extend(ids)))
    monkeypatch.setattr(rag.metadata, "save_chunks", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        rag.ingest_pdf(pdf, original_filename="new.pdf")
    assert deleted and rag.list_documents() == []
    assert list((rag.metadata.root / "documents").glob("*.pdf")) == []


def test_verified_transform_preserves_sources_and_saved_answer(make_rag, monkeypatch):
    model = Model(routes=[route(), dict(scope="CONVERSATION", task="PREVIOUS_ANSWER", confidence=.99,
        previous_action="TRANSFORM", transform_instruction="Use simpler words")])
    rag = make_rag(model)
    first = rag.chat("What was revenue in 2025?")
    model.answer = "In 2025, revenue was 100 [S1]."
    monkeypatch.setattr(rag, "_retrieve", lambda *a, **kw: pytest.fail("transform retrieved"))
    second = rag.chat("Explain that simply", conversation_id=first["conversation_id"])
    assert second["answer"] == model.answer
    assert second["citation_status"] == "cited"
    assert rag.conversations.load(first["conversation_id"])["messages"][-1]["content"] == second["answer"]


def test_unverified_transform_clears_carried_sources(make_rag):
    model = Model(routes=[route(), dict(scope="CONVERSATION", task="PREVIOUS_ANSWER", confidence=.99,
        previous_action="TRANSFORM")])
    rag = make_rag(model)
    first = rag.chat("What was revenue?")
    model.answer = "Revenue was 999 [S1]."
    model.verify = "UNSUPPORTED"
    second = rag.chat("Make it shorter", conversation_id=first["conversation_id"])
    assert second["evidence_status"] == "verification_failed"
    assert not second["sources"] and "999" not in second["answer"]


def test_failed_rewrite_does_not_search_ambiguous_followup(make_rag, monkeypatch):
    model = Model(routes=[route(), route(needs_rewrite=True)])
    rag = make_rag(model)
    first = rag.chat("What was revenue in 2025?")
    original = model.chat
    def chat(role, messages, **kw):
        if role == "rag_rewrite":
            raise RuntimeError("offline")
        return original(role, messages, **kw)
    model.chat = chat
    monkeypatch.setattr(rag, "_retrieve", lambda *a, **kw: pytest.fail("ambiguous query retrieved"))
    second = rag.chat("And 2024?", conversation_id=first["conversation_id"])
    assert second["evidence_status"] == "rewrite_unavailable"


def test_verifier_outage_has_distinct_status(make_rag):
    result = make_rag(Model(verify=RuntimeError("offline"))).chat("What was revenue?")
    assert result["evidence_status"] == "verification_unavailable"


def test_totals_and_locale_ambiguous_numbers_are_not_summed():
    subtotal = table("| Product | Sales |\n| --- | --- |\n| A | 100 |\n| Total | 100 |")
    assert "sum=" not in build_table_facts("sales", [subtotal], operations=["SUM"])
    ambiguous = table("| Product | Sales |\n| --- | --- |\n| A | 1,2 |\n| B | 100 |")
    assert "sum=" not in build_table_facts("sales", [ambiguous], operations=["SUM"])


def test_cross_document_request_abstains_if_one_pdf_has_no_evidence(make_rag, monkeypatch):
    model = Model(routes=[dict(scope="DOCUMENT", task="DOCUMENT_COMPARE", confidence=.99,
                               document_keys=["D1", "D2"])])
    rag = make_rag(model, docs=[document(), document(did="document-b")])
    monkeypatch.setattr(rag, "_retrieve", lambda q, ids, **kw: [document()] if DID in ids else [])
    result = rag.chat("Compare these PDFs")
    assert result["evidence_status"] == "incomplete_comparison"
    assert not any(role == "rag" for role, _ in model.calls)


def test_http_chat_endpoint_uses_authenticated_user_and_complete_result(make_rag, monkeypatch):
    from app import app
    from api import insight_rag_routes
    from core.deps import require_writable
    from fastapi.testclient import TestClient
    rag = make_rag()
    captured = []
    def capability(user_id):
        captured.append(user_id)
        return rag
    monkeypatch.setattr(insight_rag_routes, "_capability", capability)
    app.dependency_overrides[require_writable] = lambda: {"user_id": "authenticated-user"}
    try:
        response = TestClient(app).post("/api/insight/pdf/chat", json={"question": "What was revenue?"})
    finally:
        app.dependency_overrides.pop(require_writable, None)
    assert response.status_code == 200
    assert captured == ["authenticated-user"]
    assert response.json()["grounding"]["used_pdf_evidence"] is True


def test_concurrent_manifest_updates_keep_every_document(make_rag):
    metadata = make_rag(docs=[]).metadata
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: metadata.put_document({"document_id": f"document-{i}"}), range(30)))
    assert len(metadata.list_documents()) == 30


def test_multiquery_fusion_promotes_shared_hit_without_duplicates():
    from insight_rag.retrieval import fuse_ranked_lists
    a, b, c = [document(cid=key) for key in ["a", "b", "c"]]
    ranked = fuse_ranked_lists([[a, b, b], [c, b]])
    assert ranked[0] is b and len(ranked) == 3
