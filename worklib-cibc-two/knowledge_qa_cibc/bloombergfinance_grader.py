"""Bloomberg Financial News grader for evaluating AI-generated financial analysis.

This module evaluates responses based on journalistic quality, factual accuracy,
and usefulness for financial decision-making.
"""

import logging
from enum import Enum
from typing import Any

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation.graders._utils import run_structured_parse_call
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.evaluation.types import Evaluation
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class NewsQuality(str, Enum):
    """Representating Quality of a news."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class BloombergNewsResult(BaseModel):
    """Evaluation result for financial news responses."""

    accuracy: float = Field(0.0, description="Factual correctness (0-1)")
    relevance: float = Field(0.0, description="Relevance to the question (0-1)")
    insight: float = Field(0.0, description="Depth of financial insight (0-1)")
    clarity: float = Field(0.0, description="Clarity and readability (0-1)")

    overall_score: float = Field(0.0, description="Average score (0-1)")
    quality: NewsQuality = Field(default=NewsQuality.POOR)

    explanation: str = Field(default="", description="Grader explanation")

    def to_evaluations(self) -> list[Evaluation]:
        comment = (
            f"Accuracy: {self.accuracy:.2f}\n"
            f"Relevance: {self.relevance:.2f}\n"
            f"Insight: {self.insight:.2f}\n"
            f"Clarity: {self.clarity:.2f}\n"
            f"Overall: {self.overall_score:.2f}\n\n"
            f"Explanation: {self.explanation}"
        )

        return [
            Evaluation(name="Quality", value=self.quality.value, comment=self.explanation),
            Evaluation(name="Overall", value=self.overall_score, comment=comment),
            Evaluation(name="Accuracy", value=self.accuracy, comment=comment),
            Evaluation(name="Insight", value=self.insight, comment=comment),
        ]

    @staticmethod
    def error_evaluations(error_msg: str) -> list[Evaluation]:
        comment = f"Evaluation error: {error_msg}"
        return [
            Evaluation(name="Quality", value="poor", comment=comment),
            Evaluation(name="Overall", value=0.0, comment=comment),
            Evaluation(name="Accuracy", value=0.0, comment=comment),
            Evaluation(name="Insight", value=0.0, comment=comment),
        ]


class BloombergGraderResponse(BaseModel):
    """Structured grader response."""

    evaluation: dict[str, Any] = Field(
        alias="Evaluation",
        description="Contains scores and explanation"
    )


BLOOMBERG_GRADER_PROMPT = """\
You are a financial news editor at Bloomberg.

Your task is to evaluate the quality of an AI-generated financial news response.

Focus on:

1. Accuracy (0-1)
- Are the financial facts correct?
- Are claims plausible and not misleading?

2. Relevance (0-1)
- Does the response directly answer the question?

3. Insight (0-1)
- Does it provide meaningful market insight or analysis?
- Does it explain implications (investors, economy, markets)?

4. Clarity (0-1)
- Is it well-written and easy to understand?

5. Overall Score (0-1)
- Average of the above

6. Quality Label
- excellent (>=0.85)
- good (>=0.7)
- fair (>=0.5)
- poor (<0.5)

Return JSON format:

{
  "Evaluation": {
    "Accuracy": float,
    "Relevance": float,
    "Insight": float,
    "Clarity": float,
    "Overall": float,
    "Quality": "excellent|good|fair|poor",
    "Explanation": "..."
  }
}

User Prompt:
{prompt}

AI Response:
{response}
"""


def _parse_bloomberg_result(grader_result: dict[str, Any]) -> BloombergNewsResult:
    accuracy = grader_result.get("Accuracy", 0.0)
    relevance = grader_result.get("Relevance", 0.0)
    insight = grader_result.get("Insight", 0.0)
    clarity = grader_result.get("Clarity", 0.0)
    overall = grader_result.get("Overall", 0.0)

    quality = grader_result.get("Quality", "poor")
    explanation = grader_result.get("Explanation", "")

    return BloombergNewsResult(
        accuracy=accuracy,
        relevance=relevance,
        insight=insight,
        clarity=clarity,
        overall_score=overall,
        quality=NewsQuality(quality),
        explanation=explanation,
    )


async def evaluate_bloomberg_async(
    *,
    question: str,
    answer: str,
    model_config: LLMRequestConfig | None = None,
) -> BloombergNewsResult:
    """Evaluate a response as financial news content."""

    config = model_config or LLMRequestConfig()
    client_manager = AsyncClientManager.get_instance()

    user_prompt = BLOOMBERG_GRADER_PROMPT.format(
        prompt=question,
        response=answer,
    )

    try:
        completion = await run_structured_parse_call(
            openai_client=client_manager.openai_client,
            default_model=client_manager.configs.default_evaluator_model,
            model_config=config,
            system_prompt="",
            user_prompt=user_prompt,
            response_format=BloombergGraderResponse,
        )

        parsed = completion.choices[0].message.parsed

        if parsed is None:
            raise ValueError("Null grader response")

        return _parse_bloomberg_result(parsed.evaluation)

    except Exception as e:
        logger.warning(f"Bloomberg evaluation failed: {e}")
        return BloombergNewsResult(
            accuracy=0.0,
            relevance=0.0,
            insight=0.0,
            clarity=0.0,
            overall_score=0.0,
            quality=NewsQuality.POOR,
            explanation=f"Grader error: {e}",
        )