"""
Generation Evaluation Tests
============================
Unit tests for the generation (LLM answer) component of the RAG pipeline.

Metrics evaluated:
- AnswerRelevancyMetric: Is the generated answer relevant to the query?
- FaithfulnessMetric: Is the answer grounded in the retrieval context (no hallucinations)?
- GEval (custom): Domain-specific quality (code accuracy, helpfulness)

These run as pytest tests via `deepeval test run eval/test_generation.py`
"""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)

from src.config import get_config


# --- Metric Initialization ---

config = get_config()
thresholds = config.evaluation.thresholds

answer_relevancy = AnswerRelevancyMetric(
    threshold=thresholds.answer_relevancy,
    include_reason=True,
)

faithfulness = FaithfulnessMetric(
    threshold=thresholds.faithfulness,
    include_reason=True,
)

# Custom metric: Code Answer Quality
code_quality = GEval(
    name="Code Answer Quality",
    criteria=(
        "Evaluate if the answer correctly explains the code behavior, references "
        "specific functions/classes by name, and provides actionable information. "
        "The answer should be technically accurate based on the context provided."
    ),
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.7,
)


# --- Test Cases ---

GENERATION_TEST_CASES = [
    {
        "input": "How does error handling work in the API layer?",
        "actual_output": "The API uses a centralized error handler middleware that catches all exceptions, logs them with traceback info, and returns structured JSON error responses with appropriate HTTP status codes.",
        "retrieval_context": [
            "@app.errorhandler(Exception)\ndef handle_exception(e):\n    logger.error(f'Unhandled exception: {e}', exc_info=True)\n    status_code = getattr(e, 'code', 500)\n    return jsonify({'error': str(e), 'status': status_code}), status_code",
            "class ValidationError(Exception):\n    code = 400\n\nclass NotFoundError(Exception):\n    code = 404",
        ],
    },
    {
        "input": "What caching strategy is used?",
        "actual_output": "The application uses Redis-based caching with a TTL of 300 seconds for API responses. Cache keys are generated from the request path and query parameters.",
        "retrieval_context": [
            "cache = Redis(host='localhost', port=6379, db=0)\nCACHE_TTL = 300\n\ndef cache_key(request):\n    return f'{request.path}:{request.query_string.decode()}'",
            "@cache_decorator(ttl=CACHE_TTL)\ndef get_user_profile(user_id):\n    return db.query(User).filter_by(id=user_id).first()",
        ],
    },
]


@pytest.mark.parametrize("test_data", GENERATION_TEST_CASES, ids=[t["input"][:40] for t in GENERATION_TEST_CASES])
def test_answer_relevancy(test_data: dict):
    """Test that generated answers are relevant to the user's question."""
    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=test_data["actual_output"],
        retrieval_context=test_data["retrieval_context"],
    )
    assert_test(test_case, [answer_relevancy])


@pytest.mark.parametrize("test_data", GENERATION_TEST_CASES, ids=[t["input"][:40] for t in GENERATION_TEST_CASES])
def test_faithfulness(test_data: dict):
    """Test that generated answers don't hallucinate beyond the context."""
    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=test_data["actual_output"],
        retrieval_context=test_data["retrieval_context"],
    )
    assert_test(test_case, [faithfulness])


@pytest.mark.parametrize("test_data", GENERATION_TEST_CASES, ids=[t["input"][:40] for t in GENERATION_TEST_CASES])
def test_code_quality(test_data: dict):
    """Test that answers are technically accurate and reference specific code elements."""
    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=test_data["actual_output"],
        retrieval_context=test_data["retrieval_context"],
    )
    assert_test(test_case, [code_quality])
