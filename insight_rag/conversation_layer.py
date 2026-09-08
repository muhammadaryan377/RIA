"""Previous-answer operations for ARIA Insight PDF-RAG.

The semantic router decides whether a user turn refers to the previous answer.
This layer executes that validated action. Presentation-only transformations are
source-stable: ARIA preserves prior source IDs itself and uses the provider-stable
binary verifier instead of relying on exact free-text verdict formatting.
"""

from __future__ import annotations

import json
import re
import uuid

from .config import RAG_LLM_MODEL
from .friendly import InsightPDFRAG as GroundedInsightPDFRAG
from .grounding import citation_integrity, verify_transformation
from .presentation import infer_presentation


_CITATION_RE = re.compile(r"\[(S\d+)\]", re.IGNORECASE)


class InsightPDFRAG(GroundedInsightPDFRAG):
    """Grounded RAG plus execution helpers for previous-answer operations."""

    @staticmethod
    def _last_assistant_message(history: list[dict]) -> dict | None:
        return next(
            (item for item in reversed(history or []) if item.get("role") == "assistant"),
            None,
        )

    @staticmethod
    def _source_summary(sources: list[dict]) -> str:
        if not sources:
            return "My previous answer did not use PDF evidence, so there are no PDF citations to show."
        lines = ["I used these PDF sources for my previous answer:"]
        seen = set()
        for source in sources:
            key = (
                source.get("document_id"),
                source.get("filename"),
                source.get("page"),
                source.get("table_index"),
            )
            if key in seen:
                continue
            seen.add(key)
            detail = f"- {source.get('filename') or 'PDF'}, page {source.get('page')}"
            if source.get("content_type") == "table" and source.get("table_index"):
                detail += f", table {source.get('table_index')}"
            if source.get("source_id"):
                detail += f" [{source['source_id']}]"
            lines.append(detail)
            if len(lines) >= 6:
                break
        return "\n".join(lines)

    @staticmethod
    def _ordered_source_labels(previous: dict) -> list[str]:
        labels: list[str] = []
        for source in previous.get("sources") or []:
            label = str(source.get("source_id") or "").upper().strip()
            if label and label not in labels:
                labels.append(label)
        # Conversation persistence should already carry sources, but retain cited
        # labels from the visible answer as a compatibility fallback.
        for label in _CITATION_RE.findall(str(previous.get("content") or "")):
            label = label.upper()
            if label not in labels:
                labels.append(label)
        return labels

    @staticmethod
    def _remove_citations(text: str) -> str:
        return _CITATION_RE.sub("", text or "").strip()

    @staticmethod
    def _one_line(text: str) -> str:
        text = re.sub(r"(?m)^\s*[-*•]+\s*", "", text or "")
        return " ".join(text.split())

    def _generate_transformation(self, instruction: str, previous_text: str, *, directive) -> str:
        """Generate only wording/layout; source markers are added by ARIA later."""
        clean_previous = self._remove_citations(previous_text)
        structured = getattr(self.llm, "chat_structured", None)
        instructions = [
            "Transform only the supplied previous answer exactly as requested.",
            "Do not add facts, examples, names, numbers, claims, or outside knowledge.",
            "Do not output citation markers; ARIA preserves source attribution separately.",
        ]
        if directive.shape == "ONE_LINE":
            instructions.append("Return one concise sentence on one physical line.")
        elif directive.shape == "PARAGRAPH":
            instructions.append("Return one coherent paragraph without bullets.")
        elif directive.shape == "BULLETS":
            instructions.append("Return clear bullet points.")
        if directive.detail == "BRIEF":
            instructions.append("Keep only the central information and make it materially shorter.")
        elif directive.detail == "DETAILED":
            instructions.append("Preserve the available detail without adding anything new.")
        if directive.simple:
            instructions.append("Use simpler, easier wording while preserving meaning.")

        if callable(structured):
            schema = {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            }
            raw = structured(
                "rag",
                [
                    {"role": "system", "content": " ".join(instructions)},
                    {
                        "role": "user",
                        "content": (
                            f"Transformation instruction:\n{instruction}\n\n"
                            f"Previous answer:\n{clean_previous}"
                        ),
                    },
                ],
                json_schema=schema,
                schema_name="aria_previous_answer_transform",
                temperature=0.0,
                num_predict=350,
                timeout=15,
                reasoning_effort="low",
            )
            payload = json.loads(raw)
            transformed = str(payload.get("text") or "").strip()
        else:
            transformed = self.llm.chat(
                "rag",
                [
                    {"role": "system", "content": " ".join(instructions) + " Return only the transformed answer."},
                    {
                        "role": "user",
                        "content": (
                            f"Transformation instruction:\n{instruction}\n\n"
                            f"Previous answer:\n{clean_previous}"
                        ),
                    },
                ],
                temperature=0.0,
                num_predict=350,
                timeout=20,
            ).strip()

        transformed = self._remove_citations(transformed)
        if directive.shape == "ONE_LINE":
            transformed = self._one_line(transformed)
        elif directive.shape == "PARAGRAPH":
            transformed = " ".join(transformed.split())
        if not transformed:
            raise ValueError("Transformation was empty")
        return transformed

    def _transform_previous_answer(self, instruction: str, previous: dict) -> str:
        previous_text = str(previous.get("content") or "").strip()
        if not previous_text:
            return "I don't have a previous answer to transform yet."

        directive = infer_presentation(self.llm, instruction)
        labels = self._ordered_source_labels(previous)
        suffix = " ".join(f"[{label}]" for label in labels)

        # One retry is enough for stochastic wording/verification variance. Each
        # attempt must independently pass the source/citation and semantic checks.
        for _ in range(2):
            try:
                text = self._generate_transformation(
                    instruction,
                    previous_text,
                    directive=directive,
                )
                transformed = (f"{text} {suffix}" if suffix else text).strip()
                if previous.get("sources") and citation_integrity(transformed, previous["sources"]) != "valid":
                    continue
                verification = verify_transformation(
                    self.llm,
                    original=previous_text,
                    transformed=transformed,
                    instruction=instruction,
                )
                if verification == "verified":
                    return transformed
            except Exception:
                continue

        # A one-line fallback can be made deterministically without changing a
        # single factual token: only bullets/newlines/extra whitespace are removed.
        # This is safer than refusing a purely presentational request.
        if directive.shape == "ONE_LINE":
            text = self._one_line(self._remove_citations(previous_text))
            return (f"{text} {suffix}" if suffix else text).strip()

        raise ValueError("Transformation could not be verified")

    def handle_previous_answer(
        self,
        question: str,
        *,
        conversation_id: str | None,
        history: list[dict],
        action: str,
        transform_instruction: str | None = None,
    ) -> dict:
        """Execute a router-validated SOURCES, REPEAT or TRANSFORM action."""
        conversation_id = conversation_id or uuid.uuid4().hex
        previous = self._last_assistant_message(history) or {}
        previous_sources = list(previous.get("sources") or [])
        previous_doc_ids: list[str] = []
        for source in previous_sources:
            document_id = str(source.get("document_id") or "")
            if document_id and document_id not in previous_doc_ids:
                previous_doc_ids.append(document_id)

        failed = False
        normalized_action = str(action or "NONE").upper()
        if not previous:
            answer = "I don't have a previous answer in this conversation yet."
            model = None
            normalized_action = "NONE"
        elif normalized_action == "SOURCES":
            answer = self._source_summary(previous_sources)
            model = None
        elif normalized_action == "REPEAT":
            answer = str(previous.get("content") or "I don't have a previous answer to repeat yet.")
            model = None
        elif normalized_action == "TRANSFORM":
            instruction = (transform_instruction or question).strip()
            try:
                answer = self._transform_previous_answer(instruction, previous)
                model = RAG_LLM_MODEL
            except Exception:
                failed = True
                previous_sources, previous_doc_ids = [], []
                answer = (
                    "I'm having trouble safely reshaping the previous answer right now. "
                    "The original grounded answer is still available just above."
                )
                model = RAG_LLM_MODEL
        else:
            answer = "Could you clarify what you want me to do with my previous answer?"
            model = None

        intent = f"conversation_{normalized_action.lower()}"
        return self._base_result(
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            intent=intent,
            document_ids=previous_doc_ids,
            sources=previous_sources,
            retrieval="not_used",
            evidence_status=("verification_failed" if failed else "carried_forward" if previous_sources else "not_applicable"),
            model=model,
            context_plan={"conversation_action": normalized_action},
        )

    def chat(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> dict:
        return super().chat(
            question,
            conversation_id=conversation_id,
            document_ids=document_ids,
        )
