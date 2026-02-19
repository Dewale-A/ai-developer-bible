# Part 2, Section 1: CrewAI Architecture

## What is CrewAI?

CrewAI is a framework for building multi-agent AI systems. It provides:

- **Agents** — AI entities with roles and capabilities
- **Tasks** — Work items for agents to complete
- **Tools** — Functions agents can call
- **Crews** — Teams of agents working together

---

## The CrewAI Mental Model

Think of it like a company:

```
Crew (Company)
  |
  +-- Agents (Employees)
  |     |
  |     +-- Role (Job title)
  |     +-- Goal (Objectives)
  |     +-- Backstory (Experience)
  |     +-- Tools (Skills/access)
  |
  +-- Tasks (Projects)
  |     |
  |     +-- Description (Requirements)
  |     +-- Expected Output (Deliverables)
  |     +-- Context (Dependencies)
  |
  +-- Process (Workflow)
        |
        +-- Sequential (Assembly line)
        +-- Hierarchical (Manager delegates)
```

---

## Core Components

### 1. Agent

```python
from crewai import Agent

agent = Agent(
    role="Senior Credit Analyst",
    goal="Analyze creditworthiness thoroughly",
    backstory="10 years lending experience...",
    tools=[credit_check, dti_calculator],
    llm=llm,
    verbose=True
)
```

### 2. Task

```python
from crewai import Task

task = Task(
    description="Analyze the credit profile...",
    expected_output="Credit assessment report...",
    agent=agent,
    context=[previous_task]  # Optional
)
```

### 3. Tool

```python
from crewai.tools import BaseTool

class CreditCheckTool(BaseTool):
    name = "credit_check"
    description = "Check credit score"
    
    def _run(self, score: int) -> str:
        return f"Credit tier: {calculate_tier(score)}"
```

### 4. Crew

```python
from crewai import Crew, Process

crew = Crew(
    agents=[agent1, agent2, agent3],
    tasks=[task1, task2, task3],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()
```

---

## Installation

```bash
pip install crewai crewai-tools
```

With LangChain integration:
```bash
pip install crewai[tools]
```

---

## Project Structure

```
my_project/
|-- main.py                 # Entry point
|-- src/
|   |-- __init__.py
|   |-- config/
|   |   |-- __init__.py
|   |   +-- settings.py     # Configuration
|   |-- agents/
|   |   |-- __init__.py
|   |   +-- my_agents.py    # Agent definitions
|   |-- tasks/
|   |   |-- __init__.py
|   |   +-- my_tasks.py     # Task definitions
|   |-- tools/
|   |   |-- __init__.py
|   |   +-- my_tools.py     # Custom tools
|   +-- crew.py             # Crew orchestration
|-- requirements.txt
+-- .env                    # API keys
```

---

## Configuration Pattern

```python
# src/config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
```

```python
# src/agents/my_agents.py
from crewai import Agent, LLM
from src.config.settings import OPENAI_API_KEY, OPENAI_MODEL

llm = LLM(
    model=f"openai/{OPENAI_MODEL}",
    api_key=OPENAI_API_KEY,
    temperature=0.3
)
```

---

## Execution Flow

```
1. crew.kickoff() called
         |
         v
2. First task assigned to its agent
         |
         v
3. Agent receives:
   - System prompt (role + goal + backstory)
   - Task description
   - Available tools
   - Context from previous tasks (if any)
         |
         v
4. Agent reasons and acts (ReAct loop):
   - Thinks about approach
   - Calls tools if needed
   - Produces output
         |
         v
5. Task output stored
         |
         v
6. Next task begins (with context)
         |
         v
7. Repeat until all tasks complete
         |
         v
8. Final result returned
```

---

## LLM Configuration

### OpenAI

```python
from crewai import LLM

llm = LLM(
    model="openai/gpt-4o-mini",
    api_key="sk-...",
    temperature=0.3
)
```

### Anthropic

```python
llm = LLM(
    model="anthropic/claude-3-sonnet-20240229",
    api_key="sk-ant-..."
)
```

### Azure OpenAI

```python
llm = LLM(
    model="azure/gpt-4",
    api_key="...",
    base_url="https://your-resource.openai.azure.com/"
)
```

### Local (Ollama)

```python
llm = LLM(
    model="ollama/llama2",
    base_url="http://localhost:11434"
)
```

---

## Key Configuration Options

### Agent Options

| Option | Default | Description |
|--------|---------|-------------|
| `verbose` | False | Show reasoning |
| `allow_delegation` | True | Can ask other agents |
| `memory` | False | Persist memory |
| `max_iter` | 15 | Max reasoning loops |
| `max_rpm` | None | Rate limit |

### Crew Options

| Option | Default | Description |
|--------|---------|-------------|
| `process` | Sequential | How to run tasks |
| `verbose` | False | Show all output |
| `memory` | False | Shared memory |
| `cache` | True | Cache tool results |
| `full_output` | False | Return all task outputs |

---

## Quick Start Example

```python
from crewai import Agent, Task, Crew, LLM

# 1. Configure LLM
llm = LLM(model="openai/gpt-4o-mini", api_key="sk-...")

# 2. Create agents
researcher = Agent(
    role="Researcher",
    goal="Find accurate information",
    backstory="Expert researcher",
    llm=llm
)

writer = Agent(
    role="Writer",
    goal="Write clear reports",
    backstory="Technical writer",
    llm=llm
)

# 3. Create tasks
research_task = Task(
    description="Research the topic: AI agents",
    expected_output="Key findings",
    agent=researcher
)

write_task = Task(
    description="Write a summary report",
    expected_output="Summary document",
    agent=writer,
    context=[research_task]
)

# 4. Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    verbose=True
)

# 5. Execute
result = crew.kickoff()
print(result)
```

---

## Next Up

Section 2: Agents — designing effective agent roles, goals, and backstories.
