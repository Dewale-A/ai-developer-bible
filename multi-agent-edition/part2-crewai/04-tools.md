# Part 2, Section 4: Tools — Extending Agent Capabilities

## Why Tools?

LLMs can think, but they can't act. Tools bridge that gap.

```
Without Tools:
Agent: "I need to check the credit score..."
       "But I can only guess."

With Tools:
Agent: "I need to check the credit score."
       [calls credit_check(applicant_id)]
       "Credit score is 720. Tier 2."
```

---

## Tool Anatomy

```python
from crewai.tools import BaseTool
from pydantic import Field

class CreditCheckTool(BaseTool):
    name: str = "credit_check"
    description: str = """Check credit score and return credit tier.
    
    Args:
        credit_score: Integer between 300-850
        payment_history: Years of on-time payments
        
    Returns:
        Credit tier (1-5) and risk assessment"""
    
    def _run(self, credit_score: int, payment_history: int = 0) -> str:
        """Execute the tool."""
        tier = self._calculate_tier(credit_score)
        risk = self._assess_risk(credit_score, payment_history)
        return f"Credit Tier: {tier}, Risk Level: {risk}"
    
    def _calculate_tier(self, score: int) -> int:
        if score >= 750: return 1
        if score >= 700: return 2
        if score >= 650: return 3
        if score >= 600: return 4
        return 5
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `name` | How agent refers to the tool |
| `description` | When and how to use it |
| `_run()` | Actual implementation |

---

## The Description is Critical

The agent uses the description to decide:
1. **When** to use this tool
2. **What arguments** to pass
3. **What to expect** back

### Good Description

```python
description="""Calculate debt-to-income ratio for loan assessment.

Use this when you need to evaluate an applicant's ability to take on
additional debt payments.

Args:
    annual_income: Applicant's yearly gross income in dollars
    monthly_debts: Total existing monthly debt payments
    proposed_payment: New loan monthly payment
    
Returns:
    Current DTI percentage, proposed DTI, and assessment
    (GOOD: <36%, ACCEPTABLE: 36-43%, HIGH: >43%)"""
```

### Bad Description

```python
description="Calculates DTI"  # Too vague - agent won't know when/how to use
```

---

## Tools from Your Projects

### Application Loader Tool

```python
class ApplicationLoaderTool(BaseTool):
    name: str = "application_loader"
    description: str = """Load a loan application by ID and return all details.
    
    Args:
        application_id: The application ID (e.g., 'APP001')
        
    Returns:
        Complete application data including applicant info,
        loan details, income, employment, and existing debts."""
    
    def _run(self, application_id: str) -> str:
        app_path = f"applications/{application_id}.json"
        
        if not os.path.exists(app_path):
            return f"Error: Application {application_id} not found"
        
        with open(app_path, 'r') as f:
            data = json.load(f)
        
        return json.dumps(data, indent=2)
```

### Document Reader Tool

```python
class DocumentReaderTool(BaseTool):
    name: str = "document_reader"
    description: str = """Read policy documents from the policy_documents directory.
    
    Args:
        filename: Name of file to read, or 'list' to see available files
        
    Returns:
        Document content or list of available documents."""
    
    def _run(self, filename: str = "list") -> str:
        docs_dir = "policy_documents"
        
        if filename == "list":
            files = os.listdir(docs_dir)
            return f"Available documents: {files}"
        
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, 'r') as f:
            return f.read()
```

### Data Profiler Tool

```python
class ProfilerTool(BaseTool):
    name: str = "data_profiler"
    description: str = """Profile a dataset column and return statistics.
    
    Args:
        column_name: Name of column to profile
        
    Returns:
        Statistics including count, nulls, unique values, 
        min/max for numeric, sample values."""
    
    def _run(self, column_name: str) -> str:
        if self._df is None:
            return "Error: No data loaded"
        
        col = self._df[column_name]
        stats = {
            "total": len(col),
            "nulls": col.isnull().sum(),
            "unique": col.nunique(),
        }
        
        if col.dtype in ['int64', 'float64']:
            stats.update({
                "min": col.min(),
                "max": col.max(),
                "mean": col.mean()
            })
        
        return json.dumps(stats, indent=2)
```

---

## Tool Design Principles

### 1. Single Responsibility

```python
# Bad: One tool does everything
class SuperTool:
    def _run(self, action, ...):
        if action == "credit": ...
        elif action == "dti": ...
        elif action == "risk": ...

# Good: Separate tools
class CreditCheckTool: ...
class DTICalculatorTool: ...
class RiskScoringTool: ...
```

### 2. Clear Error Handling

```python
def _run(self, application_id: str) -> str:
    if not application_id:
        return "Error: application_id is required"
    
    app_path = f"applications/{application_id}.json"
    
    if not os.path.exists(app_path):
        return f"Error: Application {application_id} not found"
    
    try:
        with open(app_path, 'r') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except json.JSONDecodeError:
        return f"Error: Invalid JSON in {application_id}"
```

### 3. Structured Output

```python
# Bad: Unstructured string
return f"Score is {score} and tier is {tier}"

# Good: Structured, parseable
return json.dumps({
    "credit_score": score,
    "credit_tier": tier,
    "risk_level": risk,
    "recommendation": "APPROVE" if tier <= 3 else "REVIEW"
})
```

---

## Using LangChain Tools

CrewAI works with LangChain tools:

```python
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

agent = Agent(
    role="Researcher",
    tools=[search_tool],
    ...
)
```

---

## Tool Assignment Strategy

### Match Tools to Roles

```python
# Intake agent: document access
intake_agent.tools = [application_loader]

# Credit analyst: credit evaluation
credit_agent.tools = [credit_check, dti_calculator]

# Risk assessor: risk scoring
risk_agent.tools = [risk_scoring]
```

### Don't Over-Tool

```python
# Bad: Agent has too many tools, gets confused
agent.tools = [tool1, tool2, tool3, tool4, tool5, tool6, tool7, tool8]

# Good: Only tools this agent needs
agent.tools = [credit_check, dti_calculator]
```

---

## Interview Questions

**Q: How does an agent decide which tool to use?**

> The agent reads the tool descriptions in its system prompt. When it needs to take an action, it matches its need to the tool descriptions. Good descriptions are critical — if the description doesn't clearly explain when to use the tool, the agent might use the wrong one or not use it at all.

**Q: How do you handle tool errors?**

> Tools should return clear error messages that help the agent understand what went wrong. Instead of throwing exceptions, return strings like "Error: Application not found" so the agent can reason about what to do next — maybe try a different approach or report the issue.

---

## Next Up

Section 5: Crews — orchestrating agents and tasks.
