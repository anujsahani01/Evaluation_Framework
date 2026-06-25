"""
Retrieval Evaluation Tests
==========================
Unit tests for the retrieval component of the RAG pipeline.

Metrics evaluated:
- ContextualPrecisionMetric: Are relevant chunks ranked higher?
- ContextualRecallMetric: Are all relevant chunks retrieved?
- ContextualRelevancyMetric: Is retrieved context relevant (no noise)?

These run as pytest tests via `deepeval test run eval/test_retrieval.py`
"""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)

from src.config import get_config


# --- Metric Initialization ---

config = get_config()
thresholds = config.evaluation.thresholds

contextual_precision = ContextualPrecisionMetric(
    threshold=thresholds.context_precision,
    include_reason=True,
)

contextual_recall = ContextualRecallMetric(
    threshold=thresholds.context_recall,
    include_reason=True,
)

contextual_relevancy = ContextualRelevancyMetric(
    threshold=thresholds.context_relevancy,
    include_reason=True,
)


# --- Test Cases ---
# In production, these would be loaded from golden_dataset.json
# For now, we provide a structure that demonstrates the pattern.

RETRIEVAL_TEST_CASES = [
    {
        "input": "How does the authentication middleware work?",
        "actual_output": "The authentication middleware validates JWT tokens by checking the Authorization header, extracting the token, and verifying it against the secret key.",
        "expected_output": "The auth middleware extracts JWT from the Authorization header and validates it using the configured secret key.",
        "retrieval_context": [
            "def auth_middleware(request):\n    token = request.headers.get('Authorization', '').replace('Bearer ', '')\n    payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])\n    request.user = payload\n    return request",
            "# config.py\nSECRET_KEY = os.getenv('JWT_SECRET')\nALGORITHM = 'HS256'\nTOKEN_EXPIRY = 3600",
        ],
    },
    {
        "input": "What database models are defined in the user module?",
        "actual_output": "The user module defines a User model with fields: id, email, password_hash, created_at, and is_active.",
        "expected_output": "The User model has fields for id, email, hashed password, creation timestamp, and active status.",
        "retrieval_context": [
            "class User(Base):\n    __tablename__ = 'users'\n    id = Column(Integer, primary_key=True)\n    email = Column(String, unique=True, nullable=False)\n    password_hash = Column(String, nullable=False)\n    created_at = Column(DateTime, default=datetime.utcnow)\n    is_active = Column(Boolean, default=True)",
        ],
    },
]


@pytest.mark.parametrize("test_data", RETRIEVAL_TEST_CASES, ids=[t["input"][:40] for t in RETRIEVAL_TEST_CASES])
def test_contextual_precision(test_data: dict):
    """Test that relevant context chunks are ranked higher in retrieval results."""
    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=test_data["actual_output"],
        expected_output=test_data["expected_output"],
        retrieval_context=test_data["retrieval_context"],
    )
    assert_test(test_case, [contextual_precision])


@pytest.mark.parametrize("test_data", RETRIEVAL_TEST_CASES, ids=[t["input"][:40] for t in RETRIEVAL_TEST_CASES])
def test_contextual_recall(test_data: dict):
    """Test that retrieval captures all relevant information."""
    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=test_data["actual_output"],
        expected_output=test_data["expected_output"],
        retrieval_context=test_data["retrieval_context"],
    )
    assert_test(test_case, [contextual_recall])


@pytest.mark.parametrize("test_data", RETRIEVAL_TEST_CASES, ids=[t["input"][:40] for t in RETRIEVAL_TEST_CASES])
def test_contextual_relevancy(test_data: dict):
    """Test that retrieved context is relevant with minimal noise."""
    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=test_data["actual_output"],
        expected_output=test_data["expected_output"],
        retrieval_context=test_data["retrieval_context"],
    )
    assert_test(test_case, [contextual_relevancy])
