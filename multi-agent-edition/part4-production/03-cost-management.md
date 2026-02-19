# Part 4, Section 3: Cost Management

## The Cost Challenge

Multi-agent systems multiply LLM costs:

```
Single Agent:
  1 task × ~2K tokens = $0.01

Multi-Agent (6 agents):
  6 tasks × ~2K tokens = $0.06
  + tool calls
  + retries
  = $0.10-0.20 per execution
```

Scale to 1000 executions/day = $100-200/day

---

## Cost Breakdown

### Token Usage by Component

| Component | Typical Tokens | Cost (GPT-4o-mini) |
|-----------|---------------|-------------------|
| Agent system prompt | 500-1000 | $0.001-0.002 |
| Task description | 200-500 | $0.0005-0.001 |
| Context (per task) | 1000-3000 | $0.002-0.006 |
| Agent reasoning | 500-2000 | $0.001-0.004 |
| Tool outputs | 200-1000 | $0.0005-0.002 |
| Final output | 500-2000 | $0.001-0.004 |

### Per-Execution Estimate

```
6 agents × ~3000 tokens each = 18,000 tokens
At $0.15/1M input + $0.60/1M output (GPT-4o-mini):
≈ $0.01-0.05 per loan application
```

---

## Optimization Strategies

### 1. Right-Size Models

Use cheaper models for simple tasks:

```python
# Simple tasks: cheap model
intake_agent = Agent(
    role="Document Intake",
    llm=LLM(model="openai/gpt-4o-mini", temperature=0.1)
)

# Complex decisions: better model
underwriter_agent = Agent(
    role="Senior Underwriter",
    llm=LLM(model="openai/gpt-4o", temperature=0.1)
)
```

**Cost impact:**
- GPT-4o-mini: $0.15/$0.60 per 1M tokens (input/output)
- GPT-4o: $2.50/$10 per 1M tokens
- ~16x cheaper for simple tasks!

### 2. Minimize Context

Don't pass everything to every agent:

```python
# Expensive: Full context chain
task6.context = [task1, task2, task3, task4, task5]  # ~15K tokens

# Cheaper: Selective context
task6.context = [task4, task5]  # ~5K tokens, still has what it needs
```

### 3. Concise Prompts

```python
# Verbose (costs more)
backstory="""You are a highly experienced senior credit analyst with 
over 15 years of experience in consumer lending, mortgage evaluation, 
and commercial credit assessment. Throughout your distinguished career,
you have evaluated thousands of applications...""" (200 tokens)

# Concise (same effect, fewer tokens)
backstory="""Senior credit analyst, 15 years consumer lending. 
Expert at evaluating creditworthiness.""" (20 tokens)
```

### 4. Cache Tool Results

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    cache=True  # Don't re-run identical tool calls
)
```

### 5. Limit Agent Iterations

```python
agent = Agent(
    role="...",
    max_iter=10,  # Prevent runaway reasoning (and cost)
)
```

### 6. Batch Processing

Process multiple items in one crew run:

```python
# Expensive: One crew per application
for app_id in app_ids:
    crew = create_crew(app_id)
    crew.kickoff()  # Full LLM overhead each time

# Cheaper: Batch in single crew
crew = create_batch_crew(app_ids)  # One crew, multiple items
crew.kickoff()
```

---

## Cost Tracking

### Track Per Execution

```python
import tiktoken

def count_tokens(text, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

class CostTracker:
    def __init__(self, input_rate=0.15, output_rate=0.60):
        self.input_tokens = 0
        self.output_tokens = 0
        self.input_rate = input_rate / 1_000_000
        self.output_rate = output_rate / 1_000_000
    
    def add_input(self, text):
        self.input_tokens += count_tokens(text)
    
    def add_output(self, text):
        self.output_tokens += count_tokens(text)
    
    @property
    def total_cost(self):
        return (self.input_tokens * self.input_rate + 
                self.output_tokens * self.output_rate)
    
    def report(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost": f"${self.total_cost:.4f}"
        }
```

### Budget Alerts

```python
DAILY_BUDGET = 50.00  # $50/day

def check_budget():
    today_cost = get_today_cost()  # From your tracking
    
    if today_cost > DAILY_BUDGET * 0.8:
        alert("80% of daily budget consumed")
    
    if today_cost > DAILY_BUDGET:
        alert("BUDGET EXCEEDED - pausing operations")
        pause_crews()
```

---

## Model Selection Guide

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| Document intake | GPT-4o-mini | Simple extraction |
| Verification | GPT-4o-mini | Rule-based checks |
| Credit analysis | GPT-4o-mini | Tool-heavy, models don't add much |
| Risk assessment | GPT-4o-mini | Calculations from tools |
| Underwriting decision | GPT-4o | Complex judgment needed |
| Report writing | GPT-4o-mini | Synthesis, not reasoning |
| Executive polish | GPT-4o | Quality matters most |

---

## Cost Comparison: Your Projects

| Project | Agents | Est. Cost/Run |
|---------|--------|---------------|
| Loan Origination | 6 | $0.03-0.08 |
| Policy Documents | 3 | $0.02-0.05 |
| Data Quality | 4-5 | $0.03-0.10 |
| Data Quality + Polish | 5 | $0.05-0.15 |

**With GPT-4o for all agents:** 5-10x higher

---

## Interview Questions

**Q: How do you manage costs in multi-agent systems?**

> Several strategies: Use cheaper models (GPT-4o-mini) for simple tasks, reserve expensive models for complex decisions. Minimize context by only passing what each agent needs. Keep prompts concise. Enable caching for repeated tool calls. Track token usage and set budget alerts. The key is matching model capability to task complexity.

---

## Next Up

Section 4: Testing Agent Systems — ensuring reliability.
