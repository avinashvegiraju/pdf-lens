"""Agent Tool Registry for PDF Lens v2."""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rag_pdf.models import RetrievedSource
from rag_pdf.retrieval import LocalVectorIndex


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str
    data: Any = None
    error: Optional[str] = None


class SearchTool:
    """Tool for searching document corpus using dense, sparse (BM25), or hybrid retrieval."""

    def __init__(self, vector_index: LocalVectorIndex):
        self.vector_index = vector_index

    def execute(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = 4,
        mode: str = "hybrid",
        embedding_model: Optional[Any] = None
    ) -> ToolResult:
        try:
            sources = self.vector_index.search(
                query=query,
                doc_ids=doc_ids,
                top_k=top_k,
                embedding_model=embedding_model,
                mode=mode
            )
            summary = f"Retrieved {len(sources)} source chunk(s) using {mode} retrieval."
            return ToolResult(
                tool_name="search",
                success=True,
                output=summary,
                data=sources
            )
        except Exception as e:
            return ToolResult(
                tool_name="search",
                success=False,
                output=f"Search failed: {str(e)}",
                error=str(e)
            )


class CalculatorTool:
    """Safe local Python execution sandbox for math, unit conversions, and numerical calculations."""

    ALLOWED_GLOBALS = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "math": math,
        "pow": pow,
    }

    def execute(self, expression: str) -> ToolResult:
        """
        Safely evaluate a mathematical expression or unit conversion.
        Rejects arbitrary code, os/sys imports, or filesystem access.
        """
        clean_expr = expression.strip()
        
        # Security sanitization
        forbidden = ["import", "exec", "eval", "open", "os", "sys", "subprocess", "__", ";"]
        for word in forbidden:
            if re.search(r"\b" + re.escape(word) + r"\b", clean_expr):
                return ToolResult(
                    tool_name="calculator",
                    success=False,
                    output="Security error: Forbidden keywords detected in expression.",
                    error="ForbiddenKeyword"
                )

        try:
            # Allow common unit conversions directly
            # e.g. "33 psi to kpa"
            unit_match = re.match(r"^([\d\.\s\+\-\*\/\(\)]+)\s*psi\s*(?:to|in)\s*kpa$", clean_expr, re.I)
            if unit_match:
                val = eval(unit_match.group(1), self.ALLOWED_GLOBALS, {})
                kpa = val * 6.89475729
                return ToolResult(
                    tool_name="calculator",
                    success=True,
                    output=f"{val} psi = {kpa:.2f} kPa",
                    data={"input_psi": val, "output_kpa": round(kpa, 2)}
                )

            unit_bar = re.match(r"^([\d\.\s\+\-\*\/\(\)]+)\s*psi\s*(?:to|in)\s*bar$", clean_expr, re.I)
            if unit_bar:
                val = eval(unit_bar.group(1), self.ALLOWED_GLOBALS, {})
                bar = val * 0.0689475729
                return ToolResult(
                    tool_name="calculator",
                    success=True,
                    output=f"{val} psi = {bar:.2f} bar",
                    data={"input_psi": val, "output_bar": round(bar, 2)}
                )

            # Standard math evaluation
            result = eval(clean_expr, self.ALLOWED_GLOBALS, {})
            return ToolResult(
                tool_name="calculator",
                success=True,
                output=f"Calculation result: {result}",
                data=result
            )
        except Exception as e:
            return ToolResult(
                tool_name="calculator",
                success=False,
                output=f"Calculation error: {str(e)}",
                error=str(e)
            )
