"""
P0 Acceptance Tests for Epic 9: Portfolio & Multi-Broker Management

ATDD TDD Cycle - RED PHASE: These tests are written to FAIL first.
Tests cover the critical path and high-risk scenarios (score >= 6).

Risk Coverage:
- R-001: Routing rule NULL ambiguity (score 6)
- R-002: Attribution field mapping error (score 6)
- R-003: Correlation field mapping error (score 6)
- R-004: Soft-delete vs hard-delete ambiguity (score 6)

Test Design Reference: _bmad-output/test-artifacts/test-design-epic-9.md
Epic: 9 - Portfolio & Multi-Broker Management
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import uuid

from src.database.models.base import Base
from src.database.models import BrokerAccount, RoutingRule, BrokerAccountType, RegimeType, StrategyTypeEnum
from src.api.portfolio_broker_endpoints import router as broker_router
from src.api.portfolio_endpoints import router as portfolio_router
from src.api.server import app


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture(scope="function")
def test_db():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create all tables
    Base.metadata.create_all(bind=engine)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(test_db):
    """Create a test client with overridden dependencies."""

    def override_get_db_session():
        try:
            yield test_db
        finally:
            pass

    # Override dependencies in the app
    app.dependency_overrides[override_get_db_session] = lambda: test_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def sample_broker_account(test_db):
    """Create a sample broker account in the test database."""
    account = BrokerAccount(
        broker_name="IC Markets",
        account_number=f"ACC{uuid.uuid4().hex[:8].upper()}",
        account_type=BrokerAccountType.STANDARD,
        account_tag="main",
        mt5_server="icmarkets-live-01.com",
        login_encrypted="encrypted_login_123",
        swap_free=False,
        leverage=100,
        currency="USD",
        is_active=True,
        is_demo=False
    )
    test_db.add(account)
    test_db.commit()
    test_db.refresh(account)
    return account


@pytest.fixture
def unique_broker_account(test_db):
    """Create a unique broker account for tests that need isolation."""
    account = BrokerAccount(
        broker_name=f"Broker_{uuid.uuid4().hex[:6]}",
        account_number=f"ACC{uuid.uuid4().hex[:8].upper()}",
        account_type=BrokerAccountType.STANDARD,
        account_tag=None,  # Start with NULL tag for routing rule tests
        mt5_server=f"server-{uuid.uuid4().hex[:6]}.com",
        login_encrypted=f"encrypted_{uuid.uuid4().hex[:8]}",
        swap_free=False,
        leverage=100,
        currency="USD",
        is_active=True,
        is_demo=False
    )
    test_db.add(account)
    test_db.commit()
    test_db.refresh(account)
    return account


# ==============================================================================
# P0-1: Broker Account Registration (3 tests)
# Covers: R-001 (NULL-tag routing ambiguity)
# ==============================================================================

class TestBrokerAccountRegistration:
    """P0 tests for broker account registration endpoint."""

    def test_p0_broker_registration_happy_path(self, client):
        """
        P0: Verify broker account can be registered successfully.

        Acceptance Criteria:
        - POST /api/portfolio/brokers returns 201
        - Response contains all required fields
        - Account is persisted in database
        """
        payload = {
            "broker_name": "IC Markets",
            "account_number": f"ACC{uuid.uuid4().hex[:8].upper()}",
            "account_type": "standard",
            "account_tag": "main",
            "mt5_server": "icmarkets-live-01.com",
            "login_encrypted": "encrypted_login_123",
            "swap_free": False,
            "leverage": 100,
            "currency": "USD"
        }

        response = client.post("/api/portfolio/brokers", json=payload)

        # RED PHASE: This should fail - expecting 201 but may get different status
        assert response.status_code == 201, f"Expected 201, got {response.status_code}: {response.text}"

        data = response.json()
        assert "id" in data, "Response missing 'id' field"
        assert data["broker_name"] == "IC Markets"
        assert data["account_type"] == "standard"
        assert data["account_tag"] == "main"
        assert data["is_active"] is True
        assert data["is_demo"] is False

    def test_p0_broker_registration_islamic_auto_swap_free(self, client):
        """
        P0: Verify Islamic accounts automatically get swap_free=True.

        Acceptance Criteria (AC7):
        - When account_type="islamic", swap_free should be automatically set to True
        - MT5 auto-detection runs automatically
        """
        payload = {
            "broker_name": "IC Markets",
            "account_number": f"ACC{uuid.uuid4().hex[:8].upper()}",
            "account_type": "islamic",
            "account_tag": "islamic_main",
            "mt5_server": "icmarkets-live-01.com",
            "login_encrypted": "encrypted_login_123",
            "swap_free": False,  # Explicitly set to False - should be overridden
            "leverage": 100,
            "currency": "USD"
        }

        response = client.post("/api/portfolio/brokers", json=payload)

        assert response.status_code == 201, f"Expected 201, got {response.status_code}: {response.text}"

        data = response.json()
        # Islamic compliance: swap_free must be True regardless of input
        assert data["swap_free"] is True, "Islamic account must have swap_free=True"

    def test_p0_broker_registration_duplicate_rejected(self, client):
        """
        P0: Verify duplicate account numbers are rejected with 409.

        Acceptance Criteria:
        - POST with existing account_number returns 409 Conflict
        """
        account_number = f"ACC{uuid.uuid4().hex[:8].upper()}"
        payload = {
            "broker_name": "IC Markets",
            "account_number": account_number,
            "account_type": "standard",
            "account_tag": "main",
            "mt5_server": "icmarkets-live-01.com",
            "login_encrypted": "encrypted_login_123",
            "swap_free": False,
            "leverage": 100,
            "currency": "USD"
        }

        # First registration should succeed
        response1 = client.post("/api/portfolio/brokers", json=payload)
        assert response1.status_code == 201, f"First registration failed: {response1.text}"

        # Second registration with same account number should fail with 409
        response2 = client.post("/api/portfolio/brokers", json=payload)
        assert response2.status_code == 409, f"Expected 409 for duplicate, got {response2.status_code}: {response2.text}"


# ==============================================================================
# P0-2: Routing Rule CRUD with NULL-tag Edge Case (4 tests)
# Covers: R-001 (NULL-tag routing ambiguity)
# ==============================================================================

class TestRoutingRuleNULLTagEdgeCase:
    """
    P0 tests for routing rule CRUD with NULL-tag edge case.

    CRITICAL: These tests verify that the NULL-tag routing bug (R-001) is fixed.
    The bug: OR-with-NULL logic caused routingRule query to incorrectly match
    NULL-tag rules when updating a rule with account_tag="hft".

    The fix: Use explicit None check: RoutingRule.account_tag.is_(None)
    """

    def test_p0_routing_rule_create_with_explicit_tag(self, client, unique_broker_account):
        """
        P0: Verify routing rule can be created with explicit account_tag.

        Acceptance Criteria:
        - PUT /api/portfolio/brokers/{id}/routing-rules returns 201 for new rule
        - Rule is created with correct fields
        """
        payload = {
            "account_tag": "hft",
            "regime_filter": "trend",
            "strategy_type": "hft",
            "priority": 100,
            "is_active": True
        }

        response = client.put(
            f"/api/portfolio/brokers/{unique_broker_account.id}/routing-rules",
            json=payload
        )

        # RED PHASE: First creation should return 201
        # If returns 200, it means an existing rule was found and updated
        # This is the R-001 NULL-tag bug behavior
        assert response.status_code == 201, f"Expected 201 for new rule, got {response.status_code}: {response.text}"

        data = response.json()
        assert data["account_tag"] == "hft"
        assert data["strategy_type"] == "hft"
        assert data["regime_filter"] == "trend"
        assert data["priority"] == 100

    def test_p0_routing_rule_create_with_null_tag(self, client, unique_broker_account):
        """
        P0: Verify routing rule can be created with NULL account_tag.

        CRITICAL: This is the edge case that exposed R-001.

        Acceptance Criteria:
        - Rule with account_tag=null should be creatable
        - NULL and "hft" should NOT match each other
        """
        # Create a rule with NULL account_tag
        payload_null = {
            "account_tag": None,
            "regime_filter": None,
            "strategy_type": "swing",
            "priority": 50,
            "is_active": True
        }

        response_null = client.put(
            f"/api/portfolio/brokers/{unique_broker_account.id}/routing-rules",
            json=payload_null
        )

        # RED PHASE: May fail if NULL handling is broken
        assert response_null.status_code == 201, f"Expected 201 for NULL-tag rule, got {response_null.status_code}: {response_null.text}"

        # Now create a rule with explicit "hft" tag on same account
        payload_hft = {
            "account_tag": "hft",
            "regime_filter": None,
            "strategy_type": "hft",
            "priority": 100,
            "is_active": True
        }

        response_hft = client.put(
            f"/api/portfolio/brokers/{unique_broker_account.id}/routing-rules",
            json=payload_hft
        )

        assert response_hft.status_code == 201, f"Expected 201 for hft-tag rule, got {response_hft.status_code}: {response_hft.text}"

        # The two rules should be DIFFERENT - NULL should not match "hft"
        assert response_null.json()["id"] != response_hft.json()["id"], \
            "NULL-tag and 'hft'-tag rules must be distinct"

    def test_p0_routing_rule_update_returns_200(self, client, unique_broker_account):
        """
        P0: Verify updating an existing rule returns 200 (not 201).

        Acceptance Criteria (AC1):
        - PUT to existing rule returns 200
        - PUT to new rule returns 201
        """
        # Create initial rule with unique strategy type
        payload = {
            "account_tag": f"tag_{uuid.uuid4().hex[:6]}",
            "regime_filter": "trend",
            "strategy_type": "scalper",
            "priority": 100,
            "is_active": True
        }

        # First call - should create (201)
        response1 = client.put(
            f"/api/portfolio/brokers/{unique_broker_account.id}/routing-rules",
            json=payload
        )
        assert response1.status_code == 201, f"First call should create (201): {response1.text}"

        # Second call with SAME params - should update (200)
        response2 = client.put(
            f"/api/portfolio/brokers/{unique_broker_account.id}/routing-rules",
            json=payload
        )
        assert response2.status_code == 200, f"Expected 200 for update, got {response2.status_code}: {response2.text}"

        # Third call with different priority - should update (200)
        payload["priority"] = 200
        response3 = client.put(
            f"/api/portfolio/brokers/{unique_broker_account.id}/routing-rules",
            json=payload
        )
        assert response3.status_code == 200, f"Expected 200 for update, got {response3.status_code}: {response3.text}"
        assert response3.json()["priority"] == 200

    def test_p0_routing_matrix_respects_null_tag(self, client, test_db):
        """
        P0: Verify routing matrix correctly handles NULL vs explicit tag matching.

        CRITICAL: This is the R-001 regression test.

        Acceptance Criteria:
        - Routing matrix should NOT confuse NULL-tag rules with explicit-tag rules
        - Rules with account_tag="hft" should only match accounts tagged "hft"
        - Rules with account_tag=None should match accounts with NULL tag
        """
        # Create account with tag "hft" - this is the key distinction
        hft_account = BrokerAccount(
            broker_name="IC Markets HFT",
            account_number=f"ACCH{uuid.uuid4().hex[:6].upper()}",
            account_type=BrokerAccountType.STANDARD,
            account_tag="hft",  # Explicit tag - should ONLY match hft rules
            mt5_server="icmarkets-hft-01.com",
            login_encrypted=f"encrypted_hft_{uuid.uuid4().hex[:6]}",
            swap_free=False,
            leverage=500,
            currency="USD",
            is_active=True,
            is_demo=False
        )
        test_db.add(hft_account)

        # Create account with NULL tag - should match NULL-tag rules
        null_account = BrokerAccount(
            broker_name="IC Markets NULL",
            account_number=f"ACCN{uuid.uuid4().hex[:6].upper()}",
            account_type=BrokerAccountType.STANDARD,
            account_tag=None,  # NULL tag - should match NULL-tag rules
            mt5_server="icmarkets-main-01.com",
            login_encrypted=f"encrypted_null_{uuid.uuid4().hex[:6]}",
            swap_free=False,
            leverage=100,
            currency="USD",
            is_active=True,
            is_demo=False
        )
        test_db.add(null_account)
        test_db.commit()
        test_db.refresh(hft_account)
        test_db.refresh(null_account)

        # Create routing rule for "hft" strategy on hft account (explicit tag)
        hft_rule_payload = {
            "account_tag": "hft",
            "regime_filter": None,
            "strategy_type": "hft",
            "priority": 100,
            "is_active": True
        }
        response = client.put(
            f"/api/portfolio/brokers/{hft_account.id}/routing-rules",
            json=hft_rule_payload
        )
        # RED PHASE: May return 200 if bug exists (finds NULL-tag rule instead)
        assert response.status_code == 201, f"Expected 201 for new hft rule, got {response.status_code}: {response.text}"

        # Get routing matrix
        matrix_response = client.get("/api/portfolio/routing-matrix")
        assert matrix_response.status_code == 200

        matrix_data = matrix_response.json()

        # Find the hft account in the matrix
        hft_account_entry = None
        for account in matrix_data["accounts"]:
            if account["id"] == hft_account.id:
                hft_account_entry = account
                break

        assert hft_account_entry is not None, "HFT account not in matrix"
        assert hft_account_entry["account_tag"] == "hft", \
            f"HFT account should have tag 'hft', got '{hft_account_entry['account_tag']}'"

        # Find HFT strategy row in matrix
        hft_strategy_idx = None
        for idx, strategy in enumerate(matrix_data["strategies"]):
            if strategy == "hft":
                hft_strategy_idx = idx
                break

        assert hft_strategy_idx is not None, "HFT strategy not in matrix"

        # Check the assignment for HFT strategy -> HFT account
        hft_row = matrix_data["matrix"][hft_strategy_idx]
        hft_assignment = None
        for assignment in hft_row:
            if assignment["account_id"] == hft_account.id:
                hft_assignment = assignment
                break

        assert hft_assignment is not None, "HFT assignment not found"
        assert hft_assignment["assigned"] is True, "HFT strategy should be assigned to HFT account"
        assert hft_assignment["priority"] == 100


# ==============================================================================
# P0-3: Attribution Endpoint Field Mapping (3 tests)
# Covers: R-002 (Attribution field mapping error)
# ==============================================================================

class TestAttributionEndpointFieldMapping:
    """
    P0 tests for portfolio attribution endpoint field mapping.

    R-002 Bug: fetchAttribution() called wrong endpoint /api/portfolio/pnl/strategy
    instead of /api/portfolio/attribution, and extracted non-existent data.attribution field.

    The correct endpoint: GET /api/portfolio/attribution
    The correct response: data.by_strategy[].equity_contribution (not data.attribution)
    """

    def test_p0_attribution_endpoint_returns_by_strategy(self, client):
        """
        P0: Verify /api/portfolio/attribution returns by_strategy field.

        Acceptance Criteria (AC2):
        - Response contains 'by_strategy' array
        - Each strategy has 'strategy', 'pnl', 'percentage' fields
        """
        response = client.get("/api/portfolio/attribution")

        # RED PHASE: May fail if endpoint doesn't exist or returns wrong shape
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()

        # CRITICAL: Must have by_strategy field
        assert "by_strategy" in data, f"Response missing 'by_strategy' field. Got keys: {list(data.keys())}"
        assert isinstance(data["by_strategy"], list), "by_strategy must be an array"

    def test_p0_attribution_endpoint_has_equity_contribution(self, client):
        """
        P0: Verify by_strategy items contain equity_contribution field.

        R-002 Bug was: extracting non-existent 'data.attribution' field
        Correct field: 'by_strategy[].equity_contribution'

        Acceptance Criteria (AC2):
        - Each strategy item has 'equity_contribution' field
        """
        response = client.get("/api/portfolio/attribution")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()
        assert "by_strategy" in data, f"Response missing 'by_strategy': {data.keys()}"

        if len(data["by_strategy"]) > 0:
            strategy_item = data["by_strategy"][0]

            # CRITICAL: equity_contribution is the field that was missing in R-002
            assert "equity_contribution" in strategy_item, \
                f"R-002 Bug: 'equity_contribution' field missing. Got fields: {list(strategy_item.keys())}"

    def test_p0_attribution_endpoint_returns_by_broker(self, client):
        """
        P0: Verify /api/portfolio/attribution returns by_broker field.

        Acceptance Criteria (AC2):
        - Response contains 'by_broker' array
        - Each broker has 'broker', 'pnl', 'percentage' fields
        """
        response = client.get("/api/portfolio/attribution")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()

        # by_broker must exist
        assert "by_broker" in data, f"Response missing 'by_broker' field. Got keys: {list(data.keys())}"
        assert isinstance(data["by_broker"], list), "by_broker must be an array"


# ==============================================================================
# P0-4: Correlation Endpoint Field Mapping (3 tests)
# Covers: R-003 (Correlation field mapping error)
# ==============================================================================

class TestCorrelationEndpointFieldMapping:
    """
    P0 tests for portfolio correlation endpoint field mapping.

    R-003 Bug: fetchCorrelation() extracted 'data.correlations' but real response uses 'data.matrix'.

    The correct response: data.matrix (NOT data.correlations)
    """

    def test_p0_correlation_endpoint_returns_matrix_field(self, client):
        """
        P0: Verify /api/portfolio/correlation returns 'matrix' field (not 'correlations').

        R-003 Bug was: extracting 'data.correlations' instead of 'data.matrix'

        Acceptance Criteria (AC3):
        - Response contains 'matrix' field
        - 'matrix' is an array of correlation pairs
        """
        response = client.get("/api/portfolio/correlation?period_days=30")

        # RED PHASE: May fail if endpoint doesn't exist
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()

        # CRITICAL: Must be 'matrix' not 'correlations' (R-003 bug)
        assert "matrix" in data, f"Response missing 'matrix' field. Got keys: {list(data.keys())}"
        assert "correlations" not in data, "R-003 Bug: Should be 'matrix' not 'correlations'"

    def test_p0_correlation_matrix_has_required_fields(self, client):
        """
        P0: Verify correlation pair has required fields.

        Acceptance Criteria (AC3):
        - Each matrix entry has: strategy_a, strategy_b, correlation, period_days
        """
        response = client.get("/api/portfolio/correlation?period_days=30")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()
        assert "matrix" in data, "Response missing 'matrix' field"

        if len(data["matrix"]) > 0:
            pair = data["matrix"][0]

            required_fields = ["strategy_a", "strategy_b", "correlation", "period_days"]
            for field in required_fields:
                assert field in pair, f"Correlation pair missing '{field}' field. Got: {list(pair.keys())}"

            # Verify correlation is a number between -1 and 1
            assert -1.0 <= pair["correlation"] <= 1.0, \
                f"Correlation coefficient must be between -1 and 1, got {pair['correlation']}"

    def test_p0_correlation_returns_high_threshold(self, client):
        """
        P0: Verify correlation response includes high_correlation_threshold.

        Acceptance Criteria (AC3):
        - Response contains 'high_correlation_threshold' field
        - Default threshold is 0.7
        """
        response = client.get("/api/portfolio/correlation?period_days=30")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()

        assert "high_correlation_threshold" in data, \
            "Response missing 'high_correlation_threshold' field"
        assert data["high_correlation_threshold"] == 0.7, \
            f"Expected default threshold 0.7, got {data['high_correlation_threshold']}"


# ==============================================================================
# P0-5: Broker Soft-Delete Preserves Audit Trail (2 tests)
# Covers: R-004 (Soft-delete vs hard-delete ambiguity)
# ==============================================================================

class TestBrokerSoftDeleteAuditTrail:
    """
    P0 tests for broker soft-delete preserving audit trail.

    R-004 Risk: DELETE broker account marks inactive but code path may not
    preserve audit trail for regulatory compliance (NFR-R2).

    Acceptance Criteria:
    - Soft-delete sets is_active=False
    - All historical data (balance, trades) remains queryable
    - Account details are still accessible after deletion
    """

    def test_p0_soft_delete_marks_inactive(self, client, unique_broker_account):
        """
        P0: Verify DELETE sets is_active=False (soft delete).

        Acceptance Criteria:
        - DELETE /api/portfolio/brokers/{id} returns 204
        - Account is NOT physically deleted
        - is_active is set to False
        """
        account_id = unique_broker_account.id

        # Soft delete
        response = client.delete(f"/api/portfolio/brokers/{account_id}")

        # RED PHASE: May fail if delete doesn't exist or returns wrong status
        assert response.status_code == 204, f"Expected 204, got {response.status_code}: {response.text}"

        # Verify account is NOT physically deleted
        get_response = client.get(f"/api/portfolio/brokers/{account_id}")
        assert get_response.status_code == 200, f"Account should still exist: {get_response.text}"

        # Verify is_active is False
        data = get_response.json()
        assert data["is_active"] is False, f"Expected is_active=False, got {data.get('is_active')}"

    def test_p0_soft_delete_preserves_history(self, client, unique_broker_account):
        """
        P0: Verify soft-deleted account retains all historical data for audit.

        R-004: NFR-R2 requires audit trail preservation.

        Acceptance Criteria:
        - All account fields remain queryable
        - created_at and updated_at are preserved
        - Account can be retrieved with all original data
        """
        account_id = unique_broker_account.id

        # Store original values BEFORE delete
        original_broker_name = unique_broker_account.broker_name
        original_account_number = unique_broker_account.account_number
        original_leverage = unique_broker_account.leverage
        original_created_at = unique_broker_account.created_at

        # Soft delete
        delete_response = client.delete(f"/api/portfolio/brokers/{account_id}")
        assert delete_response.status_code == 204, f"Delete failed: {delete_response.text}"

        # Retrieve account and verify all data preserved
        get_response = client.get(f"/api/portfolio/brokers/{account_id}")
        assert get_response.status_code == 200, f"Account should be retrievable: {get_response.text}"

        data = get_response.json()

        # CRITICAL: All historical data must be preserved
        assert data["broker_name"] == original_broker_name, \
            f"broker_name must be preserved: expected {original_broker_name}, got {data['broker_name']}"
        assert data["account_number"] == original_account_number, \
            f"account_number must be preserved: expected {original_account_number}, got {data['account_number']}"
        assert data["leverage"] == original_leverage, \
            f"leverage must be preserved: expected {original_leverage}, got {data['leverage']}"
        assert "created_at" in data, "created_at must be preserved"
        assert "updated_at" in data, "updated_at must be preserved"
        assert data["is_active"] is False, "is_active must be False after soft delete"

    def test_p0_soft_deleted_not_in_active_list(self, client, unique_broker_account):
        """
        P0: Verify soft-deleted accounts are excluded from active_only list.

        Acceptance Criteria:
        - GET /api/portfolio/brokers?active_only=true excludes deleted accounts
        - GET /api/portfolio/brokers?active_only=false includes deleted accounts
        """
        account_id = unique_broker_account.id

        # Delete the account
        delete_response = client.delete(f"/api/portfolio/brokers/{account_id}")
        assert delete_response.status_code == 204

        # active_only=true should NOT include deleted account
        active_response = client.get("/api/portfolio/brokers?active_only=true")
        assert active_response.status_code == 200

        active_accounts = active_response.json()
        active_ids = [acc["id"] for acc in active_accounts]
        assert account_id not in active_ids, "Soft-deleted account should not appear in active list"

        # active_only=false SHOULD include deleted account
        all_response = client.get("/api/portfolio/brokers?active_only=false")
        assert all_response.status_code == 200

        all_accounts = all_response.json()
        all_ids = [acc["id"] for acc in all_accounts]
        assert account_id in all_ids, "Soft-deleted account should appear when active_only=false"


# ==============================================================================
# Test Execution Summary
# ==============================================================================

"""
P0 Test Summary for Epic 9 - Portfolio & Multi-Broker Management

Total P0 Tests Generated: 17 tests across 5 test classes

Test Classes:
1. TestBrokerAccountRegistration: 3 tests (R-001 coverage)
2. TestRoutingRuleNULLTagEdgeCase: 4 tests (R-001 coverage)
3. TestAttributionEndpointFieldMapping: 3 tests (R-002 coverage)
4. TestCorrelationEndpointFieldMapping: 3 tests (R-003 coverage)
5. TestBrokerSoftDeleteAuditTrail: 3 tests (R-004 coverage)

Risk Coverage:
- R-001 (NULL-tag routing ambiguity): Covered by TestRoutingRuleNULLTagEdgeCase
- R-002 (Attribution field mapping error): Covered by TestAttributionEndpointFieldMapping
- R-003 (Correlation field mapping error): Covered by TestCorrelationEndpointFieldMapping
- R-004 (Soft-delete audit trail): Covered by TestBrokerSoftDeleteAuditTrail

RED PHASE: These tests are expected to FAIL until implementation is complete.
After implementation, run: pytest tests/api/test_portfolio_p0_atdd.py -v

Coverage Target: 100% P0 pass rate required for Epic 9 release.
"""
