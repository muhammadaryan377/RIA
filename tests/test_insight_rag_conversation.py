"""Unit checks for user-friendly Insight Agent PDF-RAG conversation routing."""

from insight_rag.intent import (
    conversational_reply,
    is_broad_document_query,
    lexical_evidence_score,
    route_message,
)
from insight_rag.system_layer import detect_system_question


def test_greetings_do_not_route_to_document_retrieval():
    assert route_message("hello") == "smalltalk"
    assert route_message("Good morning!") == "smalltalk"
    assert route_message("thank you") == "smalltalk"
    assert route_message("how are you?") == "smalltalk"


def test_capability_questions_are_conversational():
    assert route_message("what can you do?") == "capability"
    assert route_message("what you can do i mean what you can find") == "capability"
    reply = conversational_reply("what can you do?", "capability")
    assert "text-based PDFs" in reply
    assert "won't guess" in reply


def test_assistant_identity_questions_skip_pdf_retrieval():
    assert detect_system_question("well which model you are") == "model"
    assert detect_system_question("i mean what agent you are") == "agent"
    assert detect_system_question("well what kind of agent uou are ]") == "agent"
    assert detect_system_question("which llm are u using?") == "model"
    assert detect_system_question("what embedding model do you use?") == "model"
    assert detect_system_question("what version are you?") == "version"


def test_document_model_question_is_not_mistaken_for_assistant_identity():
    assert detect_system_question("what model is mentioned in the PDF?") is None
    assert detect_system_question("which model does this document describe?") is None


def test_capability_phrase_with_pdf_target_still_routes_to_document_query():
    assert route_message("what can you find in this pdf about revenue?") == "document_query"


def test_greeting_plus_real_question_still_routes_to_rag():
    assert route_message("hello what was revenue in 2025?") == "document_query"
    assert route_message("hi, which product had the highest profit?") == "document_query"


def test_normal_document_questions_route_to_rag():
    assert route_message("What was total revenue in 2025?") == "document_query"
    assert route_message("Which product had the highest profit?") == "document_query"


def test_broad_document_questions_are_recognised():
    assert is_broad_document_query("Summarize this document")
    assert is_broad_document_query("What is this PDF about?")
    assert is_broad_document_query("What topics are in this PDF?")


def test_lexical_evidence_score_prefers_matching_evidence():
    question = "What was revenue in 2025?"
    matching = ["Annual results: revenue in 2025 was 14.2 million dollars."]
    unrelated = ["The company describes its employee onboarding policy and office locations."]

    assert lexical_evidence_score(question, matching) > lexical_evidence_score(question, unrelated)
    assert lexical_evidence_score(question, unrelated) == 0.0
