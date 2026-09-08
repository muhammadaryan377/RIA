"""Presentation planning and safe answer reshaping for ARIA PDF-RAG.

Factual routing and presentation are separate concerns. This module interprets
only how the user wants an answer presented (for example one line, brief,
simple, bullets, paragraph, or detailed) and never decides document scope or
factual content. Explicit presentation requests therefore cannot accidentally
change retrieval or grounding.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


_CITATION_RE = re.compile(r"\[(S\d+)\]", re.IGNORECASE)


@dataclass(frozen=True)
class PresentationDirective:
    shape: str = "AUTO"          # AUTO | ONE_LINE | PARAGRAPH | BULLETS
    detail: str = "NORMAL"       # BRIEF | NORMAL | DETAILED
    simple: bool = False

    @property
    def explicit(self) -> bool:
        return self.shape != "AUTO" or self.detail != "NORMAL" or self.simple

    @property
    def max_items(self) -> int:
        if self.shape in {"ONE_LINE", "PARAGRAPH"}:
            return 1
        if self.detail == "BRIEF":
            return 3
        return 8 if self.detail == "DETAILED" else 6


_PRESENTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "shape": {
            "type": "string",
            "enum": ["AUTO", "ONE_LINE", "PARAGRAPH", "BULLETS"],
        },
        "detail": {
            "type": "string",
            "enum": ["BRIEF", "NORMAL", "DETAILED"],
        },
        "simple": {"type": "boolean"},
    },
    "required": ["shape", "detail", "simple"],
    "additionalProperties": False,
}


def infer_presentation(llm, request: str) -> PresentationDirective:
    """Semantically infer only explicit output-format preferences.

    Failure is intentionally harmless: the factual answer path continues with
    the default presentation instead of refusing the user's question.
    """
    structured = getattr(llm, "chat_structured", None)
    if not callable(structured):
        return PresentationDirective()

    messages = [
        {
            "role": "system",
            "content": (
                "Classify only the user's requested answer presentation. Do not classify topic, intent, "
                "document scope, or factual content. Use ONE_LINE only when the user explicitly wants a "
                "single line/sentence. Use PARAGRAPH only when a paragraph is explicitly requested. Use "
                "BULLETS only when bullets/points/list are explicitly requested. AUTO means no explicit "
                "shape. BRIEF means explicitly short/concise; DETAILED means explicitly detailed/in-depth; "
                "otherwise NORMAL. simple=true only when simpler/easier language is explicitly requested."
            ),
        },
        {"role": "user", "content": str(request or "")[:2000]},
    ]
    try:
        raw = structured(
            "rag_plan",
            messages,
            json_schema=_PRESENTATION_SCHEMA,
            schema_name="aria_answer_presentation",
            temperature=0.0,
            num_predict=100,
            timeout=10,
            reasoning_effort="low",
        )
        payload = json.loads(raw)
        shape = str(payload.get("shape") or "AUTO").upper()
        detail = str(payload.get("detail") or "NORMAL").upper()
        simple = bool(payload.get("simple"))
        if shape not in {"AUTO", "ONE_LINE", "PARAGRAPH", "BULLETS"}:
            shape = "AUTO"
        if detail not in {"BRIEF", "NORMAL", "DETAILED"}:
            detail = "NORMAL"
        return PresentationDirective(shape=shape, detail=detail, simple=simple)
    except Exception:
        return PresentationDirective()


def cited_labels(text: str) -> list[str]:
    labels: list[str] = []
    for value in _CITATION_RE.findall(text or ""):
        label = value.upper()
        if label not in labels:
            labels.append(label)
    return labels


def _strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text or "").strip()


def _normalise_shape(text: str, directive: PresentationDirective) -> str:
    text = (text or "").strip()
    if directive.shape == "ONE_LINE":
        text = re.sub(r"(?m)^\s*[-*•]+\s*", "", text)
        return " ".join(text.split())
    if directive.shape == "PARAGRAPH":
        return " ".join(text.split())
    return text


def reformat_grounded_answer(
    llm,
    *,
    request: str,
    answer: str,
    directive: PresentationDirective | None = None,
) -> str:
    """Apply an explicit presentation request without changing source identity.

    Existing source labels are preserved deterministically by ARIA rather than
    asking the model to recreate citation syntax. The caller's final grounding
    verifier still audits the rewritten factual claims against PDF evidence.
    """
    directive = directive or infer_presentation(llm, request)
    if not directive.explicit:
        return answer

    labels = cited_labels(answer)
    original = _strip_citations(answer)
    if not original:
        return answer

    structured = getattr(llm, "chat_structured", None)
    if not callable(structured):
        if directive.shape == "ONE_LINE":
            shaped = _normalise_shape(original, directive)
            suffix = " ".join(f"[{label}]" for label in labels)
            return (f"{shaped} {suffix}" if suffix else shaped).strip()
        return answer

    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    instructions = [
        "Rewrite only the supplied answer; do not add facts, examples, numbers, names, or outside knowledge.",
        "Do not output citation markers; ARIA will preserve the existing citations itself.",
    ]
    if directive.shape == "ONE_LINE":
        instructions.append("Return exactly one concise sentence on one physical line.")
    elif directive.shape == "PARAGRAPH":
        instructions.append("Return one coherent paragraph with no bullets.")
    elif directive.shape == "BULLETS":
        instructions.append("Return clear bullet-style points.")
    if directive.detail == "BRIEF":
        instructions.append("Make it materially shorter and keep only the central information.")
    elif directive.detail == "DETAILED":
        instructions.append("Keep the available detail, but do not introduce anything new.")
    if directive.simple:
        instructions.append("Use simpler, easier language while preserving the meaning.")

    try:
        raw = structured(
            "rag",
            [
                {"role": "system", "content": " ".join(instructions)},
                {
                    "role": "user",
                    "content": f"User presentation request:\n{request}\n\nGrounded answer to reshape:\n{original}",
                },
            ],
            json_schema=schema,
            schema_name="aria_answer_reshape",
            temperature=0.0,
            num_predict=350,
            timeout=15,
            reasoning_effort="low",
        )
        payload = json.loads(raw)
        text = _strip_citations(str(payload.get("text") or ""))
        if not text:
            return answer
        text = _normalise_shape(text, directive)
        suffix = " ".join(f"[{label}]" for label in labels)
        return (f"{text} {suffix}" if suffix else text).strip()
    except Exception:
        if directive.shape == "ONE_LINE":
            shaped = _normalise_shape(original, directive)
            suffix = " ".join(f"[{label}]" for label in labels)
            return (f"{shaped} {suffix}" if suffix else shaped).strip()
        return answer
