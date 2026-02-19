# Part 4, Section 2: Observability & Debugging

## The Visibility Problem

Multi-agent systems are black boxes without observability:

```
Input --> [???] --> Output

"Why did it deny the loan?"
"I don't know, it just did."
```

---

## Verbose Mode

The simplest observability — see agent thinking:

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    verbose=True  # Shows everything
)
```

**Output:**
```
[Agent: Credit Analyst] Starting task...
[Agent: Credit Analyst] Thinking: I need to check the credit score first.
[Agent: Credit Analyst] Using tool: credit_check
[Tool: credit_check] Input: {"credit_score": 680}
[Tool: credit_check] Output: {"tier": 3, "risk": "moderate"}
[Agent: Credit Analyst] Thinking: Credit tier 3 indicates moderate risk...
[Agent: Credit Analyst] Task complete.
```

---

## Structured Logging

Log key events for production monitoring:

```python
import structlog

logger = structlog.get_logger()

def process_loan(app_id: str):
    logger.info("loan_processing_started", application_id=app_id)
    
    try:
        crew = create_loan_crew(app_id)
        
        start_time = time.time()
        result = crew.kickoff()
        duration = time.time() - start_time
        
        logger.info("loan_processing_complete",
            application_id=app_id,
            duration_seconds=round(duration, 2),
            decision=extract_decision(result)
        )
        
        return result
        
    except Exception as e:
        logger.error("loan_processing_failed",
            application_id=app_id,
            error=str(e),
            error_type=type(e).__name__
        )
        raise
```

---

## Key Metrics to Track

### Per-Crew Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `crew_duration_seconds` | Total execution time | > 120s |
| `tasks_completed` | Number of tasks finished | < expected |
| `tool_calls_total` | Total tool invocations | > 50 |
| `llm_tokens_used` | Token consumption | > budget |

### Per-Agent Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `agent_iterations` | ReAct loops | > 10 |
| `agent_tool_calls` | Tools used | > 15 |
| `agent_duration` | Time per agent | > 30s |

### Business Metrics

| Metric | Description |
|--------|-------------|
| `loans_approved` | Approvals count |
| `loans_denied` | Denials count |
| `approval_rate` | Approved / Total |
| `avg_processing_time` | Mean duration |

---

## Log File Output

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    verbose=True,
    output_log_file="logs/crew_execution.log"
)
```

---

## Custom Callbacks

Track events during execution:

```python
class CrewObserver:
    def __init__(self):
        self.events = []
    
    def on_task_start(self, task):
        self.events.append({
            "event": "task_start",
            "task": task.description[:50],
            "agent": task.agent.role,
            "timestamp": time.time()
        })
    
    def on_task_complete(self, task, output):
        self.events.append({
            "event": "task_complete",
            "task": task.description[:50],
            "output_length": len(str(output)),
            "timestamp": time.time()
        })
    
    def on_tool_use(self, tool_name, inputs, output):
        self.events.append({
            "event": "tool_use",
            "tool": tool_name,
            "timestamp": time.time()
        })
```

---

## Debugging Techniques

### 1. Isolate the Problem

Run one agent at a time:

```python
# Instead of full crew
result = crew.kickoff()

# Test each agent individually
for task in tasks:
    mini_crew = Crew(agents=[task.agent], tasks=[task], verbose=True)
    print(f"\n=== Testing: {task.agent.role} ===")
    result = mini_crew.kickoff()
    print(f"Output: {result[:200]}...")
    input("Press Enter to continue...")
```

### 2. Check Context Flow

Print what each agent receives:

```python
def debug_context(task):
    print(f"\nTask: {task.description[:50]}...")
    print(f"Agent: {task.agent.role}")
    if task.context:
        for ctx_task in task.context:
            print(f"  Context from: {ctx_task.agent.role}")
```

### 3. Tool Testing

Test tools outside of agents:

```python
# Test credit check tool directly
tool = CreditCheckTool()
result = tool._run(credit_score=720, payment_history=5)
print(f"Tool output: {result}")

# Verify it's what agent expects
assert "tier" in result.lower()
```

### 4. Compare Runs

Log outputs and compare:

```python
def compare_runs(app_id, run1_log, run2_log):
    """Find where two runs diverged."""
    for i, (e1, e2) in enumerate(zip(run1_log, run2_log)):
        if e1 != e2:
            print(f"Divergence at step {i}:")
            print(f"  Run 1: {e1}")
            print(f"  Run 2: {e2}")
            break
```

---

## Production Dashboard Ideas

```
+------------------------------------------+
|  Multi-Agent System Dashboard             |
+------------------------------------------+
|                                          |
|  Active Crews: 3    Completed: 127       |
|  Avg Duration: 45s  Error Rate: 2.1%     |
|                                          |
|  Last 24 Hours:                          |
|  ████████████████████░░░░ 85% success    |
|                                          |
|  By Agent:                               |
|  Credit Analyst  - 98% success, 12s avg  |
|  Underwriter     - 95% success, 18s avg  |
|  Risk Assessor   - 99% success, 8s avg   |
|                                          |
|  Recent Errors:                          |
|  - APP234: Tool timeout (credit_check)   |
|  - APP238: Context overflow              |
+------------------------------------------+
```

---

## Interview Questions

**Q: How do you debug a multi-agent system?**

> First, enable verbose mode to see agent thinking. Then isolate — run one agent at a time to find which one fails. Check context flow to ensure agents receive the right information. Test tools independently to verify they work. Log key events and compare successful vs failed runs to find divergence points.

---

## Next Up

Section 3: Cost Management — controlling your LLM spend.
