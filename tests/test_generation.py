"""Tests for grounded generation prompting, citation validation, and LLM client."""

import pytest
from unittest.mock import MagicMock, patch

from rag_pdf.models import RetrievedSource
from rag_pdf.generation import (
    build_grounded_prompt,
    extract_and_validate_citations,
    check_llama_health,
    LLMClient
)


@pytest.fixture
def sample_sources():
    return [
        RetrievedSource(
            citation_id="[S1]",
            chunk_id="doc1:p0001:c001",
            doc_id="doc1",
            filename="guide.pdf",
            page_number=1,
            page_index=0,
            section_title="Engine",
            text="Stop the car safely and allow engine to cool down.",
            score=0.85,
            block_bboxes=[[10, 10, 100, 50]]
        ),
        RetrievedSource(
            citation_id="[S2]",
            chunk_id="doc1:p0002:c001",
            doc_id="doc1",
            filename="guide.pdf",
            page_number=2,
            page_index=1,
            section_title="Tires",
            text="Check tire pressure when tires are cold.",
            score=0.72,
            block_bboxes=[[10, 10, 100, 50]]
        )
    ]


def test_build_grounded_prompt(sample_sources):
    prompt_text, budgeted = build_grounded_prompt(
        question="Why is engine overheating?",
        sources=sample_sources
    )

    assert "Question: Why is engine overheating?" in prompt_text
    assert "[S1] guide.pdf | Page 1 | Section: Engine" in prompt_text
    assert "[S2] guide.pdf | Page 2 | Section: Tires" in prompt_text
    assert "Stop the car safely" in prompt_text
    assert len(budgeted) == 2


def test_context_budget_pruning(sample_sources):
    # Set a tight budget that only accommodates one source
    prompt_text, budgeted = build_grounded_prompt(
        question="Why is engine overheating?",
        sources=sample_sources,
        max_budget_tokens=250
    )

    assert len(budgeted) == 1
    assert budgeted[0].citation_id == "[S1]"
    assert "[S2]" not in prompt_text


def test_citation_validation(sample_sources):
    # Response citing valid [S1] and [S2]
    text1 = "The manual instructs to pull over [S1] and check cold tires [S2]."
    verified, unverified = extract_and_validate_citations(text1, sample_sources)
    assert verified == ["[S1]", "[S2]"]
    assert unverified == []

    # Response citing valid [S1] and hallucinated [S5]
    text2 = "Pull over immediately [S1]. Then consult your dealer [S5]."
    verified2, unverified2 = extract_and_validate_citations(text2, sample_sources)
    assert verified2 == ["[S1]"]
    assert unverified2 == ["[S5]"]


def test_mock_streaming_generation(sample_sources):
    """Test LLMClient streaming with a mocked OpenAI stream response."""
    client = LLMClient()

    # Create mock chunks
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock(delta=MagicMock(content="Stop "))]
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock(delta=MagicMock(content="the vehicle [S1]."))]

    mock_stream = [chunk1, chunk2]

    with patch.object(client.client.chat.completions, "create", return_value=mock_stream):
        tokens = []
        final_metrics = None
        for tok, metrics in client.stream_answer("What should I do?", sample_sources):
            if tok:
                tokens.append(tok)
            if metrics:
                final_metrics = metrics

        full_output = "".join(tokens)
        assert full_output == "Stop the vehicle [S1]."
        assert final_metrics is not None
        assert final_metrics.output_tokens == 2
        assert final_metrics.total_generation_ms >= 0


def test_health_check_offline():
    # An unused localhost port should report offline cleanly without crash
    connected, msg = check_llama_health("http://127.0.0.1:59999/health")
    assert connected is False
    assert "offline" in msg or "unreachable" in msg
