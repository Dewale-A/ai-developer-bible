# Part 4, Section 1: Error Handling in Agent Systems

## Why Agent Errors Are Different

Multi-agent systems have compounding failure modes:

```
Single Agent:
  Error → Task fails

Multi-Agent (6 agents):
  Agent 3 error → Context corrupted → Agents 4,5,6 produce garbage
```

---

## Common Failure Modes

### 1. Tool Failures

```python
# Agent tries to load non-existent application
[Tool Error] application_loader: File not found: APP999.json
```

**Solution:** Tools should return error strings, not raise exceptions:

```python
class ApplicationLoaderTool(BaseTool):
    def _run(self, application_id: str) -> str:
        filepath = f"applications/{application_id}.json"
        
        if not os.path.exists(filepath):
            return f"ERROR: Application {application_id} not found. Available: {self._list_apps()}"
        
        try:
            with open(filepath) as f:
                return json.dumps(json.load(f), indent=2)
        except json.JSONDecodeError:
            return f"ERROR: {application_id} contains invalid JSON"
```

### 2. LLM Rate Limits

```
openai.RateLimitError: Rate limit exceeded
```

**Solution:** Retry with backoff:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def run_crew_with_retry(crew):
    return crew.kickoff()
```

### 3. Context Overflow

Agent receives too much context and gets confused.

**Solution:** Selective context passing:

```python
# Bad: Pass everything
task5.context = [task1, task2, task3, task4]

# Good: Pass only what's needed
task5.context = [task3, task4]  # Only analysis tasks
```

### 4. Agent Loops

Agent keeps calling tools without progressing.

**Solution:** Set max iterations:

```python
agent = Agent(
    ...,
    max_iter=15,  # Limit reasoning loops
)
```

### 5. Hallucinated Tool Calls

Agent invents tools that don't exist.

**Solution:** Clear tool descriptions:

```python
class CreditCheckTool(BaseTool):
    name = "credit_check"  # Exact name agent must use
    description = """ONLY use this to check credit scores.
    
    Required args: credit_score (int 300-850)
    Returns: Credit tier and risk assessment"""
```

---

## Defensive Crew Design

### Wrap Kickoff

```python
def process_application(app_id: str) -> dict:
    try:
        crew = create_loan_crew(app_id)
        result = crew.kickoff()
        return {"status": "success", "result": str(result)}
    
    except FileNotFoundError as e:
        return {"status": "error", "type": "missing_file", "message": str(e)}
    
    except Exception as e:
        logger.error(f"Crew failed for {app_id}: {e}")
        return {"status": "error", "type": "unknown", "message": str(e)}
```

### Validate Inputs

```python
def create_loan_crew(app_id: str) -> Crew:
    # Validate before creating crew
    if not app_id or not app_id.startswith("APP"):
        raise ValueError(f"Invalid application ID: {app_id}")
    
    app_path = f"applications/{app_id}.json"
    if not os.path.exists(app_path):
        raise FileNotFoundError(f"Application not found: {app_id}")
    
    # Now safe to create crew
    return _build_crew(app_id)
```

### Timeout Protection

```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Crew execution timed out")

def run_with_timeout(crew, timeout_seconds=300):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = crew.kickoff()
        signal.alarm(0)  # Cancel alarm
        return result
    except TimeoutError:
        logger.error("Crew timed out")
        raise
```

---

## Graceful Degradation

When full workflow fails, provide partial value:

```python
def process_loan(app_id: str):
    try:
        # Full crew
        result = run_full_crew(app_id)
        return {"complete": True, "result": result}
    
    except CreditCheckFailure:
        # Partial crew without credit analysis
        result = run_partial_crew(app_id, skip=["credit"])
        return {
            "complete": False,
            "result": result,
            "warning": "Credit analysis unavailable - manual review required"
        }
    
    except Exception as e:
        # Minimum viable response
        return {
            "complete": False,
            "result": None,
            "error": str(e),
            "suggestion": "Please retry or contact support"
        }
```

---

## Interview Questions

**Q: How do you handle errors in multi-agent systems?**

> I design tools to return error strings rather than throw exceptions, so agents can reason about failures. I wrap crew execution in try-catch blocks, validate inputs before starting, and implement graceful degradation — if credit analysis fails, I can still provide partial results and flag for manual review. I also set max iterations to prevent agent loops.

---

## Next Up

Section 2: Observability — seeing inside your agent systems.
