# Part 2, Section 5: Crews — Orchestrating Collaboration

## What is a Crew?

A Crew is a team of agents working together on tasks:

```python
crew = Crew(
    agents=[agent1, agent2, agent3],
    tasks=[task1, task2, task3],
    process=Process.sequential
)

result = crew.kickoff()
```

---

## Crew Configuration

### Basic Crew

```python
from crewai import Crew, Process

crew = Crew(
    agents=[intake, verification, credit, underwriter],
    tasks=[intake_task, verify_task, credit_task, decision_task],
    process=Process.sequential,
    verbose=True
)
```

### All Options

```python
crew = Crew(
    agents=[...],           # List of agents
    tasks=[...],            # List of tasks
    process=Process.sequential,  # or hierarchical
    verbose=True,           # Show execution details
    memory=False,           # Shared crew memory
    cache=True,             # Cache tool results
    max_rpm=10,             # Rate limit (requests per minute)
    full_output=False,      # Return all task outputs
    output_log_file="crew.log",  # Log to file
)
```

---

## Process Types

### Sequential

Tasks run one after another:

```
Task 1 --> Task 2 --> Task 3 --> Task 4
```

```python
crew = Crew(
    agents=[a1, a2, a3, a4],
    tasks=[t1, t2, t3, t4],
    process=Process.sequential
)
```

**Use when:**
- Clear step-by-step workflow
- Each task depends on previous
- Order matters

### Hierarchical

A manager delegates to workers:

```
         [Manager]
        /    |    \
       /     |     \
   [W1]    [W2]    [W3]
```

```python
crew = Crew(
    agents=[manager, worker1, worker2, worker3],
    tasks=[main_task],
    process=Process.hierarchical,
    manager_llm=LLM(model="openai/gpt-4o")  # Manager needs good model
)
```

**Use when:**
- Complex tasks needing dynamic assignment
- Multiple approaches possible
- Coordination required

---

## Building Crews: Patterns from Your Projects

### Loan Origination Crew

```python
def create_loan_origination_crew(application_id: str) -> Crew:
    # Create agents
    intake = document_intake_agent()
    verifier = verification_agent()
    credit = credit_analyst_agent()
    risk = risk_assessor_agent()
    underwriter = underwriter_agent()
    offer_gen = offer_generator_agent()
    
    # Create tasks with dependencies
    intake_task = create_intake_task(intake, application_id)
    verify_task = create_verification_task(verifier, application_id)
    credit_task = create_credit_analysis_task(credit)
    risk_task = create_risk_assessment_task(risk)
    decision_task = create_underwriting_task(underwriter)
    offer_task = create_offer_generation_task(offer_gen)
    
    # Set up context chain
    verify_task.context = [intake_task]
    credit_task.context = [intake_task, verify_task]
    risk_task.context = [intake_task, credit_task]
    decision_task.context = [verify_task, credit_task, risk_task]
    offer_task.context = [intake_task, decision_task]
    
    return Crew(
        agents=[intake, verifier, credit, risk, underwriter, offer_gen],
        tasks=[intake_task, verify_task, credit_task, 
               risk_task, decision_task, offer_task],
        process=Process.sequential,
        verbose=True
    )
```

### Data Quality Crew (with Optional Agent)

```python
class DataQualityCrew:
    def __init__(self, data_file: str, polish: bool = False):
        self.polish = polish
        
        # Core agents
        self.profiler = create_profiler_agent()
        self.validator = create_validator_agent()
        self.anomaly = create_anomaly_detector_agent()
        self.writer = create_report_writer_agent()
        
        # Optional polish agent
        if self.polish:
            self.editor = create_senior_editor_agent()
    
    def create_crew(self) -> Crew:
        agents = [self.profiler, self.validator, 
                  self.anomaly, self.writer]
        tasks = [self.profile_task, self.validate_task,
                 self.anomaly_task, self.report_task]
        
        if self.polish:
            agents.append(self.editor)
            tasks.append(self.polish_task)
        
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
```

---

## Running Crews

### Basic Execution

```python
crew = create_my_crew()
result = crew.kickoff()
print(result)
```

### With Inputs

```python
result = crew.kickoff(inputs={
    "application_id": "APP001",
    "loan_amount": 50000
})
```

### Async Execution

```python
import asyncio

async def process_applications(app_ids):
    tasks = []
    for app_id in app_ids:
        crew = create_loan_crew(app_id)
        tasks.append(crew.kickoff_async())
    
    results = await asyncio.gather(*tasks)
    return results
```

---

## Error Handling

### Try-Catch Pattern

```python
def process_loan(application_id: str):
    try:
        crew = create_loan_origination_crew(application_id)
        result = crew.kickoff()
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

### Timeout

```python
from crewai import Crew

crew = Crew(
    agents=[...],
    tasks=[...],
    # max_execution_time=300  # 5 minute timeout (if supported)
)
```

---

## Crew Output

### Raw Output

```python
result = crew.kickoff()
# Returns: Final task's output as string
```

### Full Output

```python
crew = Crew(
    ...,
    full_output=True
)

result = crew.kickoff()
# Returns: All task outputs
for task_output in result.tasks_output:
    print(f"Task: {task_output.description}")
    print(f"Output: {task_output.raw}")
```

---

## Performance Optimization

### 1. Model Selection

```python
# Use cheaper model for simple tasks
intake_agent = Agent(llm=LLM(model="openai/gpt-4o-mini"))

# Use powerful model for complex tasks
underwriter_agent = Agent(llm=LLM(model="openai/gpt-4o"))
```

### 2. Task Granularity

```python
# Too granular (too many LLM calls)
tasks = [load_task, parse_task, validate_task, format_task]

# Right level (meaningful chunks)
tasks = [ingest_task, analyze_task, report_task]
```

### 3. Caching

```python
crew = Crew(
    ...,
    cache=True  # Cache repeated tool calls
)
```

---

## Debugging Crews

### Verbose Mode

```python
crew = Crew(..., verbose=True)
# Shows: Agent thinking, tool calls, outputs
```

### Logging

```python
crew = Crew(
    ...,
    output_log_file="crew_execution.log"
)
```

### Step-by-Step

```python
# Run one task at a time for debugging
for task in tasks:
    mini_crew = Crew(
        agents=[task.agent],
        tasks=[task],
        verbose=True
    )
    result = mini_crew.kickoff()
    print(f"Task result: {result}")
    input("Press Enter for next task...")
```

---

## Interview Questions

**Q: How do you orchestrate multiple agents in CrewAI?**

> Create a Crew with your agents and tasks. Set `process=Process.sequential` for step-by-step workflows or `process=Process.hierarchical` for manager-delegated work. Link tasks with context to pass information between them. Call `crew.kickoff()` to execute.

**Q: How do you handle errors in a crew?**

> Wrap `crew.kickoff()` in try-catch for runtime errors. Design tools to return error strings instead of throwing exceptions so agents can reason about failures. Use verbose mode during development to see where issues occur. For production, log outputs and implement retries.

---

## Part 2 Complete!

You now understand CrewAI:
- Architecture overview
- Agent design (roles, goals, backstories)
- Task design (descriptions, outputs, context)
- Tool design (names, descriptions, implementation)
- Crew orchestration (processes, execution)

**Next:** Part 3 — Project Walkthroughs with real code.
