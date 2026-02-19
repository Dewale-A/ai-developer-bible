# Part 1, Section 3: Orchestration Patterns

## How Agents Work Together

Orchestration defines how tasks flow between agents. Three main patterns:

---

## Pattern 1: Sequential

Agents work one after another, like an assembly line.

```
[Agent A] --> [Agent B] --> [Agent C] --> [Agent D]
   |              |              |              |
   v              v              v              v
Output A --> Output B --> Output C --> Final Output
```

### Example: Loan Origination

```
[Intake] --> [Verification] --> [Credit] --> [Risk] --> [Underwriter] --> [Offer]
```

Each agent:
1. Receives context from previous agent(s)
2. Does its specialized work
3. Passes results to next agent

### Code (CrewAI)

```python
crew = Crew(
    agents=[intake, verification, credit, risk, underwriter, offer],
    tasks=[task1, task2, task3, task4, task5, task6],
    process=Process.sequential,  # One at a time
)
```

### When to Use

- Clear step-by-step process
- Each step depends on previous
- Order matters
- Like: Loan processing, document pipelines

---

## Pattern 2: Hierarchical

A manager agent coordinates worker agents.

```
                [Manager Agent]
                /      |      \
               /       |       \
              v        v        v
        [Worker A] [Worker B] [Worker C]
```

The manager:
1. Receives the overall task
2. Breaks it into subtasks
3. Delegates to appropriate workers
4. Synthesizes results

### Example: Research Task

```
        [Research Manager]
        /        |        \
       /         |         \
      v          v          v
[Web Search] [Analysis] [Writing]
```

### Code (CrewAI)

```python
crew = Crew(
    agents=[manager, worker1, worker2, worker3],
    tasks=[main_task],
    process=Process.hierarchical,
    manager_llm=LLM(model="openai/gpt-4o")  # Manager uses better model
)
```

### When to Use

- Complex tasks needing coordination
- Multiple possible approaches
- Need dynamic task assignment
- Like: Research, creative projects, problem-solving

---

## Pattern 3: Collaborative (Parallel)

Multiple agents work simultaneously on different aspects.

```
                Input
                  |
        +---------+---------+
        |         |         |
        v         v         v
    [Agent A] [Agent B] [Agent C]
        |         |         |
        +---------+---------+
                  |
                  v
            [Aggregator]
```

### Example: Due Diligence

```
                Application
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
[ID Verify]  [Credit Check]  [Employment]
        |           |           |
        +-----------+-----------+
                    |
                    v
            [Decision Maker]
```

### When to Use

- Independent subtasks
- Latency matters (parallel = faster)
- Like: Verification steps, multi-source analysis

---

## Context Passing

How information flows between agents:

### Explicit Context

```python
task2 = Task(
    description="...",
    agent=agent2,
    context=[task1]  # Receives task1's output
)
```

### Context Chain

```python
# Task 3 sees output from Task 1 AND Task 2
task3.context = [task1, task2]
```

### Selective Context

```python
# Underwriter sees verification, credit, and risk
# But NOT the raw intake (too much detail)
underwriting_task.context = [
    verification_task,
    credit_task,
    risk_task
]
```

---

## Designing Agent Workflows

### Step 1: Identify the Process

Map out how humans do this task:
```
Loan Processing:
1. Receive application
2. Verify identity and income
3. Pull credit report
4. Calculate risk score
5. Make decision
6. Generate offer/denial
```

### Step 2: Define Agents

One agent per major role:
```python
agents = {
    "intake": document_intake_agent(),
    "verification": verification_agent(),
    "credit": credit_analyst_agent(),
    "risk": risk_assessor_agent(),
    "underwriter": underwriter_agent(),
    "offer": offer_generator_agent(),
}
```

### Step 3: Design Task Dependencies

```
intake ----+
           |
           v
verification --+
               |
               v
credit --------+---> risk ---> underwriter ---> offer
```

### Step 4: Choose Orchestration

- Most business workflows: **Sequential**
- Complex analysis: **Hierarchical**
- Independent checks: **Parallel** (if supported)

---

## Common Patterns in Practice

### The Analyst-Writer Pattern

```
[Analyst Agent] --> [Writer Agent]
     |                    |
     v                    v
  Analysis         Formatted Report
```

Used in: Policy Documents Application

### The Pipeline Pattern

```
[Ingest] --> [Process] --> [Validate] --> [Report]
```

Used in: Data Quality Assessment

### The Review Pattern

```
[Doer Agent] --> [Reviewer Agent] --> [Final Output]
                       |
                       v
                 (may send back)
```

Used in: Data Quality with Polish flag

---

## Interview Questions

**Q: What's the difference between sequential and hierarchical orchestration?**

> In sequential, tasks run one after another in a fixed order — each agent knows exactly when it runs. In hierarchical, a manager agent dynamically decides which workers to use and how to combine their results. Sequential is simpler and predictable; hierarchical is more flexible but complex.

**Q: How do you decide which pattern to use?**

> I ask: "Is there a clear step-by-step process?" If yes, sequential. "Do tasks need dynamic coordination?" If yes, hierarchical. "Can tasks run independently?" If yes, consider parallel. Most enterprise workflows (loan processing, compliance) fit sequential because they mirror existing business processes.

**Q: How do agents share information?**

> Through context. In CrewAI, you set `task.context = [previous_task]` so the agent receives the previous task's output. You can chain multiple contexts, and selectively choose which previous tasks to include to keep context manageable.

---

## Next Up

Section 4: When to Use Multi-Agent — making the right architectural decisions.
