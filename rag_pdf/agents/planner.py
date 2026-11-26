"""Query Planner and Sub-Query Decomposer for PDF Lens v2."""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QueryPlan:
    original_query: str
    is_complex: bool
    sub_queries: List[str] = field(default_factory=list)
    requires_calculation: bool = False
    calculation_expression: Optional[str] = None
    reasoning: str = ""


def plan_query(query: str) -> QueryPlan:
    """
    Analyze the user prompt and generate an execution plan:
    - Determines if multi-hop decomposition is needed.
    - Identifies mathematical calculation or unit conversion requests.
    """
    clean_q = query.strip()
    sub_queries: List[str] = []
    is_complex = False
    requires_calc = False
    calc_expr = None
    reasoning_parts = []

    # 1. Calculation & Unit Conversion Detection
    unit_patterns = [
        r"(\d+(?:\.\d+)?\s*psi\s*(?:to|in)\s*kpa)",
        r"(\d+(?:\.\d+)?\s*psi\s*(?:to|in)\s*bar)",
        r"(?:convert|calculate|how much is)\s+([\d\.\s\+\-\*\/\(\)]+)",
        r"([\d\.]+\s*[\+\-\*\/]\s*[\d\.]+)"
    ]
    for pat in unit_patterns:
        match = re.search(pat, clean_q, re.I)
        if match:
            requires_calc = True
            calc_expr = match.group(1).strip()
            reasoning_parts.append(f"Detected numerical calculation or unit conversion: '{calc_expr}'")
            break

    # 2. Multi-Hop / Comparative Decomposition
    # Checks for conjunctions and comparison intents:
    # e.g., "compare A and B", "difference between X and Y", "what is X and how does Y"
    compare_match = re.search(r"(?:compare|difference between|versus|vs\.?)\s+(.+?)\s+(?:and|with|to)\s+(.+)", clean_q, re.I)
    multi_part_match = re.search(r"^(.+?)\s+(?:and also|as well as|additionally|and what about)\s+(.+)", clean_q, re.I)

    if compare_match:
        is_complex = True
        sub_a = compare_match.group(1).strip()
        sub_b = compare_match.group(2).strip()
        sub_queries = [sub_a, sub_b]
        reasoning_parts.append(f"Decomposed comparative query into 2 sub-queries: [1] '{sub_a}', [2] '{sub_b}'")
    elif multi_part_match:
        is_complex = True
        sub_a = multi_part_match.group(1).strip()
        sub_b = multi_part_match.group(2).strip()
        sub_queries = [sub_a, sub_b]
        reasoning_parts.append(f"Decomposed multi-part query into 2 sub-queries: [1] '{sub_a}', [2] '{sub_b}'")
    elif " and " in clean_q and len(clean_q.split()) > 7:
        # Check if query has two distinct clauses
        parts = clean_q.split(" and ")
        if len(parts) == 2 and len(parts[0].split()) >= 3 and len(parts[1].split()) >= 3:
            is_complex = True
            sub_queries = [parts[0].strip(), parts[1].strip()]
            reasoning_parts.append(f"Split dual-topic query into: [1] '{sub_queries[0]}', [2] '{sub_queries[1]}'")
        else:
            sub_queries = [clean_q]
            reasoning_parts.append("Single-topic query; executing direct hybrid search.")
    else:
        sub_queries = [clean_q]
        reasoning_parts.append("Single-topic query; executing direct hybrid search.")

    reasoning_text = "; ".join(reasoning_parts)

    return QueryPlan(
        original_query=clean_q,
        is_complex=is_complex,
        sub_queries=sub_queries if sub_queries else [clean_q],
        requires_calculation=requires_calc,
        calculation_expression=calc_expr,
        reasoning=reasoning_text
    )
