"""Agentic RAG Orchestrator coordinating Planning, Search, Calculation, and Verification."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

from rag_pdf.models import RetrievedSource, GenerationMetrics, ChatMessage
from rag_pdf.retrieval import LocalVectorIndex
from rag_pdf.generation import LLMClient
from .planner import plan_query, QueryPlan
from .tools import SearchTool, CalculatorTool, ToolResult
from .critic import CriticAgent, VerificationResult


@dataclass
class AgentTraceStep:
    agent_name: str          # "PLANNER", "RETRIEVAL", "CALCULATOR", "GENERATOR", "CRITIC"
    title: str
    detail: str
    status: str = "success"  # "success", "running", "warning"


@dataclass
class AgenticResponse:
    answer: str
    sources: List[RetrievedSource]
    metrics: GenerationMetrics
    plan: QueryPlan
    trace_steps: List[AgentTraceStep]
    verification: VerificationResult


class AgenticRAGOrchestrator:
    """
    Coordinates the multi-agent reasoning loop:
    1. Planner Agent decomposes queries and identifies math/unit conversions.
    2. Search Agent executes hybrid retrieval for each sub-query.
    3. Python Sandbox Tool performs calculations if requested.
    4. Generation Engine streams the synthesized answer.
    5. Critic Agent verifies citations and factuality.
    """

    def __init__(
        self,
        vector_index: LocalVectorIndex,
        llm_client: Optional[LLMClient] = None
    ):
        self.vector_index = vector_index
        self.llm_client = llm_client or LLMClient()
        self.search_tool = SearchTool(vector_index)
        self.calc_tool = CalculatorTool()
        self.critic = CriticAgent()

    def run_agentic_workflow(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = 4,
        embedding_model: Optional[Any] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        search_mode: str = "hybrid"
    ) -> Generator[Tuple[Optional[AgentTraceStep], str, Optional[AgenticResponse]], None, None]:
        """
        Executes the agentic workflow as a generator:
        Yields (trace_step, token_delta, final_response).
        """
        trace_steps: List[AgentTraceStep] = []
        t_start = time.perf_counter()

        # STEP 1: QUERY PLANNING AGENT
        plan = plan_query(query)
        step_plan = AgentTraceStep(
            agent_name="PLANNER",
            title="Query Planning & Decomposition",
            detail=f"Plan: {plan.reasoning}"
        )
        trace_steps.append(step_plan)
        yield step_plan, "", None

        # STEP 2: RETRIEVAL AGENT (Execute Sub-Queries)
        aggregated_sources: List[RetrievedSource] = []
        seen_chunk_ids = set()

        t_ret_start = time.perf_counter()
        for idx, sub_q in enumerate(plan.sub_queries):
            step_search = AgentTraceStep(
                agent_name="RETRIEVAL",
                title=f"Sub-Query {idx+1}/{len(plan.sub_queries)}",
                detail=f"Searching ({search_mode}): '{sub_q}'"
            )
            trace_steps.append(step_search)
            yield step_search, "", None

            sub_k = max(2, top_k // len(plan.sub_queries)) if plan.is_complex else top_k
            result = self.search_tool.execute(
                query=sub_q,
                doc_ids=doc_ids,
                top_k=sub_k,
                mode=search_mode,
                embedding_model=embedding_model
            )
            if result.success and result.data:
                for src in result.data:
                    if src.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(src.chunk_id)
                        aggregated_sources.append(src)

        t_ret_end = time.perf_counter()
        retrieval_duration_ms = (t_ret_end - t_ret_start) * 1000.0

        # Renumber citation IDs deterministically: [S1] .. [SK]
        for i, src in enumerate(aggregated_sources):
            src.citation_id = f"[S{i+1}]"

        step_ret_done = AgentTraceStep(
            agent_name="RETRIEVAL",
            title="Evidence Aggregation",
            detail=f"Aggregated {len(aggregated_sources)} unique source chunk(s) across sub-queries in {retrieval_duration_ms:.1f}ms."
        )
        trace_steps.append(step_ret_done)
        yield step_ret_done, "", None

        # STEP 3: PYTHON CALCULATION SANDBOX TOOL (If requested)
        calc_context = ""
        if plan.requires_calculation and plan.calculation_expression:
            step_calc = AgentTraceStep(
                agent_name="CALCULATOR",
                title="Python Sandbox Execution",
                detail=f"Evaluating expression: '{plan.calculation_expression}'"
            )
            trace_steps.append(step_calc)
            yield step_calc, "", None

            calc_res = self.calc_tool.execute(plan.calculation_expression)
            if calc_res.success:
                calc_context = f"\n[Calculated Data Reference: {calc_res.output}]"
                step_calc.detail = f"Output: {calc_res.output}"
            else:
                step_calc.status = "warning"
                step_calc.detail = f"Warning: {calc_res.output}"

        # Augment question with calculation reference if present
        augmented_query = query + (f" (Note: {calc_context})" if calc_context else "")

        # STEP 4: GENERATOR AGENT (Stream LLM Response)
        full_answer = ""
        gen_metrics: Optional[GenerationMetrics] = None

        step_gen = AgentTraceStep(
            agent_name="GENERATOR",
            title="Grounded Answer Synthesis",
            detail="Streaming response from local LFM3-1.7B with citations..."
        )
        trace_steps.append(step_gen)
        yield step_gen, "", None

        try:
            stream_gen = self.llm_client.stream_answer(
                question=augmented_query,
                sources=aggregated_sources,
                chat_history=chat_history
            )
            for token, metrics in stream_gen:
                if token:
                    full_answer += token
                    yield None, token, None
                if metrics:
                    gen_metrics = metrics
        except Exception as e:
            full_answer = f"Error during generation: {str(e)}"
            yield None, full_answer, None

        if gen_metrics:
            gen_metrics.retrieval_ms = round(retrieval_duration_ms, 1)
            gen_metrics.total_response_ms = round(gen_metrics.retrieval_ms + gen_metrics.total_generation_ms, 1)
        else:
            total_time_ms = (time.perf_counter() - t_start) * 1000.0
            gen_metrics = GenerationMetrics(
                retrieval_ms=round(retrieval_duration_ms, 1),
                total_generation_ms=round(total_time_ms - retrieval_duration_ms, 1),
                total_response_ms=round(total_time_ms, 1)
            )

        # STEP 5: CRITIC & VERIFIER AGENT (CRAG)
        verification = self.critic.verify(query, full_answer, aggregated_sources)
        step_critic = AgentTraceStep(
            agent_name="CRITIC",
            title="Citation & Fact Verification",
            detail=f"Confidence: {int(verification.confidence_score * 100)}% | {verification.critique_notes}",
            status="success" if verification.is_verified else "warning"
        )
        trace_steps.append(step_critic)
        yield step_critic, "", None

        final_response = AgenticResponse(
            answer=full_answer,
            sources=aggregated_sources,
            metrics=gen_metrics,
            plan=plan,
            trace_steps=trace_steps,
            verification=verification
        )

        yield None, "", final_response
