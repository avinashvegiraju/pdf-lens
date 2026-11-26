"""Tests for Agentic RAG components (Planner, Calculator, Critic, and Orchestrator)."""

import pytest
from unittest.mock import MagicMock, patch

from rag_pdf.models import RetrievedSource
from rag_pdf.agents.planner import plan_query
from rag_pdf.agents.tools import CalculatorTool, SearchTool
from rag_pdf.agents.critic import CriticAgent
from rag_pdf.agents.orchestrator import AgenticRAGOrchestrator


def test_query_planner_single_topic():
    plan = plan_query("What should I do if the engine temperature turns red?")
    assert plan.is_complex is False
    assert len(plan.sub_queries) == 1
    assert plan.requires_calculation is False


def test_query_planner_comparative_decomposition():
    plan = plan_query("Compare coolant overheating alerts with tire maintenance guidelines")
    assert plan.is_complex is True
    assert len(plan.sub_queries) == 2
    assert "coolant overheating alerts" in plan.sub_queries[0]
    assert "tire maintenance guidelines" in plan.sub_queries[1]


def test_query_planner_unit_conversion():
    plan = plan_query("What is the cold tire pressure and convert 33 psi to kpa?")
    assert plan.requires_calculation is True
    assert "33 psi to kpa" in plan.calculation_expression.lower()


def test_calculator_tool_math_and_conversions():
    calc = CalculatorTool()

    # Basic arithmetic
    res_math = calc.execute("(33 * 2) + 10")
    assert res_math.success is True
    assert res_math.data == 76

    # PSI to kPa
    res_kpa = calc.execute("33 psi to kpa")
    assert res_kpa.success is True
    assert res_kpa.data["output_kpa"] == pytest.approx(227.53, abs=0.1)

    # PSI to bar
    res_bar = calc.execute("33 psi to bar")
    assert res_bar.success is True
    assert res_bar.data["output_bar"] == pytest.approx(2.28, abs=0.05)


def test_calculator_tool_security_sandbox():
    calc = CalculatorTool()

    # Reject dangerous keywords
    res_bad1 = calc.execute("import os; os.system('echo 1')")
    assert res_bad1.success is False
    assert "Security error" in res_bad1.output

    res_bad2 = calc.execute("open('/etc/passwd')")
    assert res_bad2.success is False
    assert "Security error" in res_bad2.output


def test_critic_agent_verification():
    critic = CriticAgent()
    sources = [
        RetrievedSource("[S1]", "c1", "d1", "manual.pdf", 1, 0, "Engine", "Overheating instructions", 0.9, []),
        RetrievedSource("[S2]", "c2", "d1", "manual.pdf", 2, 1, "Tires", "Tire pressure 33 psi", 0.8, [])
    ]

    # Valid answer citing S1
    ans_good = "When the engine light turns red, pull over safely [S1]."
    res_good = critic.verify("engine light", ans_good, sources)
    assert res_good.is_verified is True
    assert res_good.verified_citations == ["[S1]"]
    assert res_good.unverified_citations == []
    assert res_good.confidence_score >= 0.9

    # Hallucinated citation S99
    ans_hallucinated = "Check tire pressure [S1] and also consult your dealer [S99]."
    res_bad = critic.verify("tire pressure", ans_hallucinated, sources)
    assert res_bad.is_verified is False
    assert res_bad.unverified_citations == ["[S99]"]
    assert res_bad.confidence_score < 0.8

    # Standard refusal
    ans_refusal = "I couldn't find that information in the selected PDF pages."
    res_refusal = critic.verify("stock price", ans_refusal, sources)
    assert res_refusal.is_verified is True
    assert res_refusal.confidence_score == 1.0


def test_agentic_orchestrator_mock_workflow():
    mock_vector_index = MagicMock()
    mock_llm_client = MagicMock()

    # Mock search tool returning sample sources
    sample_src = RetrievedSource("[S1]", "c1", "d1", "doc.pdf", 1, 0, "Title", "Text", 0.88, [])
    mock_vector_index.search.return_value = [sample_src]

    # Mock streaming response
    mock_stream = [("Stop ", None), ("the car [S1].", None)]
    mock_llm_client.stream_answer.return_value = iter(mock_stream)

    orchestrator = AgenticRAGOrchestrator(
        vector_index=mock_vector_index,
        llm_client=mock_llm_client
    )

    steps_seen = []
    tokens_seen = []
    final_res = None

    for step, token, res in orchestrator.run_agentic_workflow(
        query="What should I do if the engine overheats?"
    ):
        if step:
            steps_seen.append(step.agent_name)
        if token:
            tokens_seen.append(token)
        if res:
            final_res = res

    assert "PLANNER" in steps_seen
    assert "RETRIEVAL" in steps_seen
    assert "GENERATOR" in steps_seen
    assert "CRITIC" in steps_seen

    assert "".join(tokens_seen) == "Stop the car [S1]."
    assert final_res is not None
    assert final_res.verification.is_verified is True
