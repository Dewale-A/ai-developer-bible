# Part 1, Section 2: Agent Architectures

## What is an Agent?

An agent is an LLM enhanced with:
1. **Identity** — Role, goals, personality
2. **Memory** — Context from previous steps
3. **Tools** — Functions it can call
4. **Autonomy** — Ability to decide what to do

```
Basic LLM:
  Input --> [LLM] --> Output

Agent:
  Input --> [LLM + Identity + Tools + Memory] --> Actions + Output
```

---

## Core Agent Patterns

### 1. ReAct (Reasoning + Acting)

The most common pattern. Agent thinks, then acts, then observes.

```
Question: "What's the credit score for applicant John Smith?"

Thought: I need to look up the credit score. I have a credit_check tool.
Action: credit_check(name="John Smith")
Observation: Credit score is 720

Thought: I have the answer.
Final Answer: John Smith's credit score is 720.
```

**The Loop:**
```
Think --> Act --> Observe --> Think --> Act --> Observe --> ... --> Answer
```

### 2. Plan-and-Execute

Agent creates a full plan first, then executes steps.

```
Question: "Process loan application APP001"

Plan:
1. Load application data
2. Verify applicant identity
3. Check credit score
4. Calculate DTI ratio
5. Assess risk
6. Make decision

Execution:
- Step 1: [Executing...] Done.
- Step 2: [Executing...] Done.
- ...
```

**Best for:** Complex, multi-step tasks with clear phases.

### 3. Self-Reflection

Agent evaluates its own output and improves it.

```
Draft: "The loan is approved for $50,000"

Reflection: "I should include the interest rate and term."

Revised: "The loan is approved for $50,000 at 6.5% APR for 60 months."

Reflection: "I should add monthly payment."

Final: "The loan is approved for $50,000 at 6.5% APR for 60 months. 
        Monthly payment: $980.52"
```

---

## Agent Components (CrewAI Model)

### Role

What the agent is:
```python
role="Senior Credit Analyst"
```

### Goal

What the agent is trying to achieve:
```python
goal="Analyze applicant creditworthiness and provide detailed assessment"
```

### Backstory

Why the agent has expertise (affects behavior):
```python
backstory="""You are a senior credit analyst with 10 years experience 
in consumer lending. You evaluate credit reports, payment histories,
and outstanding debts. Your analysis forms the foundation of 
lending decisions."""
```

### Tools

What the agent can do:
```python
tools=[credit_check_tool, dti_calculator_tool]
```

### LLM

The underlying model:
```python
llm=LLM(model="openai/gpt-4o-mini", temperature=0.3)
```

---

## The Anatomy of an Agent Call

When an agent receives a task:

```
1. SYSTEM PROMPT (built from role + goal + backstory)
   "You are a Senior Credit Analyst. Your goal is to..."

2. AVAILABLE TOOLS (injected)
   "You have access to: credit_check, dti_calculator"

3. TASK DESCRIPTION
   "Analyze the creditworthiness of this applicant..."

4. CONTEXT (from previous tasks)
   "Previous agent found: income=$85,000, debts=$12,000"

5. LLM PROCESSES
   - Reasons about the task
   - Decides which tools to use
   - Executes tools
   - Synthesizes results

6. OUTPUT
   "Credit assessment: Tier 2, DTI 28%, Recommend: APPROVE"
```

---

## Tool Design for Agents

Tools are functions agents can call. Design matters!

### Good Tool Design

```python
class CreditCheckTool(BaseTool):
    name = "credit_check"
    description = """Check credit score and history for an applicant.
    
    Args:
        credit_score: The applicant's credit score (300-850)
        payment_history: Years of on-time payments
        
    Returns:
        Credit tier (1-5) and risk assessment
    """
    
    def _run(self, credit_score: int, payment_history: int) -> str:
        # Implementation
        tier = self._calculate_tier(credit_score)
        return f"Credit Tier: {tier}, Risk: {self._assess_risk(...)}"
```

**Key principles:**

1. **Clear name** — Agent knows when to use it
2. **Detailed description** — Agent knows how to use it
3. **Typed arguments** — Agent knows what to pass
4. **Structured output** — Agent can parse results

### Bad Tool Design

```python
class Tool:
    name = "check"  # Too vague
    description = "Checks stuff"  # Not helpful
    
    def _run(self, data):  # No types, unclear input
        # Returns unstructured blob
        return some_complex_object
```

---

## Memory in Agents

### Short-term Memory (Context)

Information from the current conversation/workflow:
```python
task.context = [previous_task]  # Pass output of previous task
```

### Long-term Memory (Persistence)

Some frameworks support persistent memory:
```python
agent = Agent(
    memory=True,  # Remember across sessions
    ...
)
```

### Shared Memory (Crew-level)

All agents can access:
```python
crew = Crew(
    agents=[...],
    memory=True,  # Shared crew memory
)
```

---

## Agent Behavior Tuning

### Temperature

Controls randomness:
```python
# Low temperature (0.1-0.3): Consistent, factual
llm = LLM(model="openai/gpt-4o-mini", temperature=0.2)

# High temperature (0.7-0.9): Creative, varied
llm = LLM(model="openai/gpt-4o-mini", temperature=0.8)
```

**For business workflows:** Use low temperature (0.1-0.3)

### Verbose Mode

See agent's thinking:
```python
agent = Agent(
    ...
    verbose=True  # Shows reasoning in logs
)
```

### Allow Delegation

Let agents ask other agents for help:
```python
agent = Agent(
    ...
    allow_delegation=True  # Can delegate to other crew members
)
```

---

## Interview Questions

**Q: What's the difference between an LLM and an agent?**

> An LLM is the raw model — it takes text in and produces text out. An agent wraps the LLM with identity (role, goal), capabilities (tools), and memory (context). The agent can reason about what to do, call external functions, and maintain state across interactions.

**Q: What is the ReAct pattern?**

> ReAct stands for Reasoning + Acting. The agent loops through: Think about what to do, Take an action (often calling a tool), Observe the result, then Think again. This continues until the agent has enough information to produce a final answer.

**Q: How do you make agents more reliable?**

> Several techniques: Lower temperature for consistency, clear tool descriptions so the agent knows when to use them, explicit instructions in the goal/backstory, and structured output formats. Also, giving agents narrow scope — one job per agent — reduces errors.

---

## Next Up

Section 3: Orchestration Patterns — how multiple agents work together.
