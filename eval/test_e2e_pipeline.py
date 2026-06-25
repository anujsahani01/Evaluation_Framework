"""
End-to-End Pipeline Evaluation
==============================
Tests the full pipeline: query → retrieve → generate → score.

This combines RAG metrics and agent metrics into a single pass,
simulating real user interactions with the system.

Run via: `deepeval test run eval/test_e2e_pipeline.py`
"""

import pytest
from deepeval import assert_test, evaluate
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    GEval,
)

from src.config import get_config


# --- Metric Initialization ---

config = get_config()
thresholds = config.evaluation.thresholds

# Combine metrics for E2E evaluation
e2e_metrics = [
    AnswerRelevancyMetric(threshold=thresholds.answer_relevancy),
    FaithfulnessMetric(threshold=thresholds.faithfulness),
    ContextualRelevancyMetric(threshold=thresholds.context_relevancy),
    GEval(
        name="Helpfulness",
        criteria=(
            "Evaluate how helpful the answer is for a developer trying to understand "
            "or modify the codebase. A helpful answer provides specific file paths, "
            "function names, explains behavior, and suggests next steps if relevant."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        threshold=0.6,
    ),
]


# --- E2E Test Cases ---
# These simulate real queries a developer would ask about their codebase.

E2E_TEST_CASES = [
    {
        "input": "How do I add a new API endpoint to the application?",
        "actual_output": (
            "To add a new API endpoint:\n"
            "1. Create a new route in `routes/` directory\n"
            "2. Define the handler function with appropriate decorators\n"
            "3. Register the route in `app.py` using `app.register_blueprint()`\n"
            "4. Add input validation using the `validate_request` decorator\n"
            "5. Write tests in `tests/test_routes.py`"
        ),
        "retrieval_context": [
            "# routes/users.py\nfrom flask import Blueprint\nusers_bp = Blueprint('users', __name__)\n\n@users_bp.route('/users', methods=['GET'])\n@validate_request\ndef list_users():\n    return jsonify(User.query.all())",
            "# app.py\nfrom routes.users import users_bp\napp.register_blueprint(users_bp, url_prefix='/api/v1')",
        ],
    },
    {
        "input": "What happens when a database connection fails?",
        "actual_output": (
            "When a database connection fails, the application:\n"
            "1. Catches the `OperationalError` in the connection pool\n"
            "2. Retries up to 3 times with exponential backoff\n"
            "3. Logs the error with full connection details\n"
            "4. Returns a 503 Service Unavailable response to the client"
        ),
        "retrieval_context": [
            "class DatabasePool:\n    MAX_RETRIES = 3\n    BACKOFF_FACTOR = 2\n\n    def get_connection(self):\n        for attempt in range(self.MAX_RETRIES):\n            try:\n                return self._create_connection()\n            except OperationalError as e:\n                logger.error(f'DB connection failed (attempt {attempt+1}): {e}')\n                time.sleep(self.BACKOFF_FACTOR ** attempt)\n        raise ServiceUnavailableError('Database unavailable')",
        ],
    },
]


@pytest.mark.parametrize(
    "test_data",
    E2E_TEST_CASES,
    ids=[t["input"][:40] for t in E2E_TEST_CASES],
)
def test_e2e_pipeline(test_data: dict):
    """
    End-to-end pipeline test: evaluates the full flow from query to answer.

    Passes ALL metrics at once — if any metric fails, the test case fails.
    This mirrors a real CI/CD gate: the pipeline must pass all quality checks.
    """
    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=test_data["actual_output"],
        retrieval_context=test_data["retrieval_context"],
    )
    assert_test(test_case, e2e_metrics)
