"""Self-Reflective Critic and Verifier Agent for Corrective RAG (CRAG)."""

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from rag_pdf.models import RetrievedSource


@dataclass
class VerificationResult:
    is_verified: bool
    verified_citations: List[str] = field(default_factory=list)
    unverified_citations: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    critique_notes: str = ""
    requires_re_retrieval: bool = False


class CriticAgent:
    """Evaluates answer grounding, validates citations, and detects hallucinations."""

    def verify(
        self,
        question: str,
        answer: str,
        sources: List[RetrievedSource]
    ) -> VerificationResult:
        """
        Verify that generated answer is strictly grounded in retrieved sources:
        1. Checks citation validity (no invented [S#]).
        2. Checks citation coverage (answers must cite sources unless refusing).
        3. Checks refusal compliance when information was missing.
        """
        if not answer.strip():
            return VerificationResult(
                is_verified=False,
                confidence_score=0.0,
                critique_notes="Generated answer is empty."
            )

        valid_ids = {src.citation_id for src in sources}
        found_cits = re.findall(r"\[S(\d+)\]", answer)
        
        verified: List[str] = []
        unverified: List[str] = []

        for num in found_cits:
            cid = f"[S{num}]"
            if cid in valid_ids:
                if cid not in verified:
                    verified.append(cid)
            else:
                if cid not in unverified:
                    unverified.append(cid)

        # Refusal check
        is_refusal = (
            "couldn't find that information" in answer.lower()
            or "cannot find" in answer.lower()
            or "no information" in answer.lower()
        )

        critique = []
        confidence = 1.0
        re_retrieve = False

        if unverified:
            confidence -= 0.35 * len(unverified)
            critique.append(f"Model cited non-existent source(s): {', '.join(unverified)}")

        if not verified and not is_refusal:
            if sources:
                confidence -= 0.3
                critique.append("Answer lacks explicit [S#] citations for factual assertions.")
            else:
                confidence -= 0.4
                critique.append("Answer generated despite zero supporting passages found.")
                re_retrieve = True

        if is_refusal:
            confidence = 1.0
            critique.append("Model correctly identified missing information and issued standard refusal.")

        confidence = max(0.0, min(1.0, confidence))
        is_fully_verified = len(unverified) == 0 and (len(verified) > 0 or is_refusal)

        if not is_fully_verified and not critique:
            critique.append("Answer verified with valid citation coverage.")

        return VerificationResult(
            is_verified=is_fully_verified,
            verified_citations=verified,
            unverified_citations=unverified,
            confidence_score=round(confidence, 2),
            critique_notes="; ".join(critique),
            requires_re_retrieval=re_retrieve
        )
