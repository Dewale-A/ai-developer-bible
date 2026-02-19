# Part 4, Section 4: Testing Agent Systems

## The Testing Challenge

Multi-agent systems are non-deterministic:

```
Same input → Different output each time
```

Traditional unit tests don't work well:

```python
# This will fail randomly
def test_loan_decision():
    result = process_loan("APP001")
    assert result == "APPROVED"  # Might say "APPROVED" differently
```

---

## Testing Strategies

### 1. Tool Unit Tests

Test tools independently — they're deterministic:

```python
def test_credit_check_tool():
    tool = CreditCheckTool()
    
    # Test excellent credit
    result = tool._run(credit_score=780)
    assert "Tier 1" in result or "tier 1" in result.lower()
    
    # Test poor credit
    result = tool._run(credit_score=550)
    assert "Tier 5" in result or "tier 5" in result.lower()

def test_dti_calculator():
    tool = DTICalculatorTool()
    
    result = json.loads(tool._run(
        annual_income=100000,
        monthly_debts=2000,
        proposed_payment=1500
    ))
    
    assert result["current_dti"] == 24.0  # 2000/(100000/12)*100
    assert result["proposed_dti"] == 42.0
```

### 2. Input Validation Tests

Test that bad inputs are handled:

```python
def test_application_loader_missing_file():
    tool = ApplicationLoaderTool()
    result = tool._run("NONEXISTENT")
    assert "error" in result.lower() or "not found" in result.lower()

def test_application_loader_invalid_json():
    # Create invalid JSON file
    with open("applications/BAD.json", "w") as f:
        f.write("not valid json")
    
    tool = ApplicationLoaderTool()
    result = tool._run("BAD")
    assert "error" in result.lower()
```

### 3. Output Structure Tests

Test that outputs have required fields:

```python
def test_loan_output_structure():
    result = process_loan("APP001")
    
    # Should mention decision
    assert any(word in result.upper() for word in 
               ["APPROVED", "DENIED", "CONDITIONAL"])
    
    # Should include key information
    result_lower = result.lower()
    assert "credit" in result_lower
    assert "risk" in result_lower

def test_report_has_sections():
    result = run_policy_analysis()
    
    assert "executive summary" in result.lower()
    assert "findings" in result.lower()
    assert "recommendations" in result.lower()
```

### 4. Boundary Tests

Test edge cases:

```python
@pytest.mark.parametrize("credit_score,expected_tier", [
    (850, 1),  # Maximum
    (300, 5),  # Minimum
    (749, 2),  # Just below tier 1
    (750, 1),  # Exactly tier 1
])
def test_credit_tiers(credit_score, expected_tier):
    tool = CreditCheckTool()
    result = json.loads(tool._run(credit_score=credit_score))
    assert result["credit_tier"] == expected_tier
```

### 5. Smoke Tests

Run full crew and check it completes:

```python
def test_loan_crew_completes():
    """Smoke test - full execution should complete without error."""
    crew = create_loan_origination_crew("APP001")
    
    try:
        result = crew.kickoff()
        assert result is not None
        assert len(str(result)) > 100  # Produced meaningful output
    except Exception as e:
        pytest.fail(f"Crew execution failed: {e}")
```

### 6. Regression Tests

Save known-good outputs, compare:

```python
def test_regression_app001():
    result = process_loan("APP001")
    
    # Load expected output patterns
    with open("tests/fixtures/APP001_expected.txt") as f:
        expected_patterns = f.read().splitlines()
    
    # Check key phrases appear
    for pattern in expected_patterns:
        assert pattern.lower() in result.lower(), \
            f"Expected pattern not found: {pattern}"
```

### 7. Agent Isolation Tests

Test each agent individually:

```python
def test_credit_analyst_agent():
    agent = credit_analyst_agent()
    task = Task(
        description="Analyze credit for: score=720, payment_history=5yrs",
        expected_output="Credit assessment",
        agent=agent
    )
    
    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    
    assert "tier 2" in result.lower() or "720" in result
```

---

## Test Fixtures

Create consistent test data:

```python
# tests/fixtures/test_applications.py

GOOD_APPLICANT = {
    "application_id": "TEST001",
    "applicant": {"name": "Test Good"},
    "financial": {
        "annual_income": 100000,
        "credit_score": 750,
        "monthly_debts": 1000
    },
    "loan_request": {"amount": 20000}
}

BAD_APPLICANT = {
    "application_id": "TEST002",
    "applicant": {"name": "Test Bad"},
    "financial": {
        "annual_income": 30000,
        "credit_score": 520,
        "monthly_debts": 2000
    },
    "loan_request": {"amount": 50000}
}
```

---

## Mocking LLMs for Speed

Full LLM calls are slow. Mock for unit tests:

```python
from unittest.mock import patch, MagicMock

def test_tool_usage_without_llm():
    # Mock LLM response
    mock_response = MagicMock()
    mock_response.content = "Credit tier: 2. Recommendation: APPROVE"
    
    with patch('crewai.LLM') as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_response
        
        # Test runs without real LLM
        result = process_loan("APP001")
        
        assert mock_llm.return_value.invoke.called
```

---

## Continuous Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run unit tests (no LLM)
        run: pytest tests/unit -v
      
      - name: Run tool tests
        run: pytest tests/tools -v
      
      # Integration tests only on main (cost $)
      - name: Run integration tests
        if: github.ref == 'refs/heads/main'
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: pytest tests/integration -v --timeout=120
```

---

## Test Organization

```
tests/
├── unit/
│   ├── test_tools.py        # Tool unit tests (no LLM)
│   └── test_utilities.py    # Helper function tests
├── tools/
│   ├── test_credit_check.py
│   ├── test_dti_calculator.py
│   └── test_risk_scoring.py
├── integration/
│   ├── test_loan_crew.py    # Full crew tests (needs LLM)
│   └── test_policy_crew.py
├── fixtures/
│   ├── applications/
│   └── expected_outputs/
└── conftest.py              # Pytest configuration
```

---

## Interview Questions

**Q: How do you test multi-agent systems?**

> I use layered testing. Tools get traditional unit tests since they're deterministic. For agents, I test output structure (does it have required fields?) rather than exact values. I use smoke tests to ensure crews complete without error. I save known-good outputs for regression testing. Integration tests run on CI but less frequently due to LLM costs.

---

## Part 4 Complete!

You now understand production patterns:
- Error handling and graceful degradation
- Observability and debugging
- Cost management
- Testing strategies

**Next:** Part 5 — Interview Ready.
