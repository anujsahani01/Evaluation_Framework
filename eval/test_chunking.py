"""
Chunking Evaluation Tests
=========================
Unit tests for the transform/chunking component.

These evaluate chunking QUALITY — not just "did it chunk" but:
- Are chunks semantically coherent?
- Do chunks preserve function/class boundaries?
- Is information loss minimized between chunks?

Uses GEval custom metrics since chunking quality is domain-specific.

Run via: `deepeval test run eval/test_chunking.py`
"""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval

from src.config import get_config


# --- Custom Metrics for Chunking ---

config = get_config()

# Metric: Semantic Coherence
# Does each chunk represent a complete, understandable unit?
chunk_coherence = GEval(
    name="Chunk Semantic Coherence",
    criteria=(
        "Evaluate if the code chunk is semantically coherent - it should represent a complete, "
        "understandable unit of code. A good chunk contains a full function, class, or logical "
        "block rather than cutting mid-statement or mid-function. Score higher if the chunk "
        "can be understood in isolation without needing surrounding chunks."
    ),
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
    threshold=0.6,
)

# Metric: Boundary Preservation
# Did chunking respect function/class boundaries?
boundary_preservation = GEval(
    name="Boundary Preservation",
    criteria=(
        "Evaluate if the chunking preserves code boundaries. Check if functions start and end "
        "within the same chunk, if class definitions aren't split arbitrarily, and if import "
        "blocks are kept together. A perfect score means no logical unit is split across chunks."
    ),
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    threshold=0.7,
)

# Metric: Information Density
# Is the chunk information-dense (not just boilerplate/comments)?
info_density = GEval(
    name="Information Density",
    criteria=(
        "Evaluate the information density of this code chunk. A good chunk should contain "
        "meaningful logic, not just empty lines, repetitive boilerplate, or only comments. "
        "Score based on how much useful, queryable information is packed into the chunk."
    ),
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
    threshold=0.5,
)


# --- Test Cases ---

CHUNKING_TEST_CASES = [
    {
        "id": "complete_function",
        "actual_output": (
            "def authenticate_user(email: str, password: str) -> dict:\n"
            "    \"\"\"Authenticate a user and return a JWT token.\"\"\"\n"
            "    user = db.query(User).filter_by(email=email).first()\n"
            "    if not user or not verify_password(password, user.password_hash):\n"
            "        raise AuthenticationError('Invalid credentials')\n"
            "    token = create_jwt_token(user.id)\n"
            "    return {'token': token, 'user_id': user.id}\n"
        ),
        "expected_output": "A complete function that handles user authentication with error handling.",
    },
    {
        "id": "complete_class",
        "actual_output": (
            "class UserRepository:\n"
            "    \"\"\"Data access layer for user operations.\"\"\"\n\n"
            "    def __init__(self, db_session):\n"
            "        self.db = db_session\n\n"
            "    def get_by_id(self, user_id: int) -> User | None:\n"
            "        return self.db.query(User).get(user_id)\n\n"
            "    def create(self, email: str, password_hash: str) -> User:\n"
            "        user = User(email=email, password_hash=password_hash)\n"
            "        self.db.add(user)\n"
            "        self.db.commit()\n"
            "        return user\n"
        ),
        "expected_output": "A complete repository class with multiple methods for user data access.",
    },
    {
        "id": "bad_chunk_split_mid_function",
        "actual_output": (
            "def process_payment(amount, currency):\n"
            "    validated = validate_amount(amount)\n"
            "    if not validated:\n"
            "        raise ValueError('Invalid amount')\n"
            # Chunk cut off mid-function!
        ),
        "expected_output": "A complete payment processing function with validation, API call, and response handling.",
    },
]


@pytest.mark.parametrize(
    "test_data",
    CHUNKING_TEST_CASES,
    ids=[t["id"] for t in CHUNKING_TEST_CASES],
)
def test_chunk_coherence(test_data: dict):
    """Test that each chunk is semantically coherent and self-contained."""
    test_case = LLMTestCase(
        input="Evaluate this code chunk for semantic coherence.",
        actual_output=test_data["actual_output"],
    )
    assert_test(test_case, [chunk_coherence])


@pytest.mark.parametrize(
    "test_data",
    CHUNKING_TEST_CASES,
    ids=[t["id"] for t in CHUNKING_TEST_CASES],
)
def test_boundary_preservation(test_data: dict):
    """Test that chunks respect code structure boundaries."""
    test_case = LLMTestCase(
        input="Evaluate if this chunk preserves code boundaries.",
        actual_output=test_data["actual_output"],
        expected_output=test_data["expected_output"],
    )
    assert_test(test_case, [boundary_preservation])


@pytest.mark.parametrize(
    "test_data",
    CHUNKING_TEST_CASES,
    ids=[t["id"] for t in CHUNKING_TEST_CASES],
)
def test_info_density(test_data: dict):
    """Test that chunks are information-dense, not mostly boilerplate."""
    test_case = LLMTestCase(
        input="Evaluate the information density of this code chunk.",
        actual_output=test_data["actual_output"],
    )
    assert_test(test_case, [info_density])
