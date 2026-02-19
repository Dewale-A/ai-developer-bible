# Part 2, Section 3: Tasks — Dependencies and Context

## What is a Task?

A task is a unit of work assigned to an agent:

```python
task = Task(
    description="What to do",
    expected_output="What success looks like",
    agent=who_does_it,
    context=[what_they_need_to_know]
)
```

---

## Task Components

### Description

The instructions for what to do:

```python
description="""Analyze the credit profile for this loan application.

Your responsibilities:
1. Use credit_check tool with the applicant's credit score
2. Use dti_calculator tool to compute debt-to-income ratio
3. Evaluate credit tier and identify risk factors
4. Assess ability to take on the proposed loan payment

Use the applicant information from the previous tasks."""
```

**Good descriptions:**
- Numbered steps
- Mention specific tools to use
- Reference where to get input
- Clear deliverable

### Expected Output

What the agent should produce:

```python
expected_output="""A credit analysis report containing:
- Credit score evaluation and tier
- Debt-to-income ratio (current and proposed)
- Credit risk factors (positive and negative)
- Credit recommendation (PASS / CONDITIONAL / FAIL)
- Specific concerns or strengths to highlight"""
```

**This matters because:**
- Guides the agent toward the right format
- Makes validation easier
- Ensures consistency across runs

### Agent Assignment

Who does this task:

```python
agent=credit_analyst_agent
```

One agent per task. Match expertise to task requirements.

---

## Context: Connecting Tasks

Context passes information between tasks.

### Basic Context

```python
task1 = Task(
    description="Load the application",
    expected_output="Application data",
    agent=intake_agent
)

task2 = Task(
    description="Verify the application",
    expected_output="Verification report",
    agent=verification_agent,
    context=[task1]  # Receives task1's output
)
```

**What happens:**
1. task1 runs, produces output
2. task2 receives task1's output as context
3. task2's agent sees: "Previous task output: [task1 result]"

### Multiple Context Sources

```python
underwriting_task = Task(
    description="Make the final underwriting decision",
    expected_output="APPROVED / DENIED with reasoning",
    agent=underwriter,
    context=[
        verification_task,
        credit_task,
        risk_task
    ]
)
```

The underwriter sees outputs from all three tasks.

### Selective Context

Not every task needs all previous outputs:

```python
# Intake doesn't need context (first task)
intake_task = create_intake_task(intake_agent, app_id)

# Verification only needs intake
verification_task.context = [intake_task]

# Credit needs intake and verification
credit_task.context = [intake_task, verification_task]

# Risk needs intake and credit (not verification details)
risk_task.context = [intake_task, credit_task]

# Underwriter needs verification, credit, and risk
underwriting_task.context = [verification_task, credit_task, risk_task]

# Offer needs intake (for amounts) and decision
offer_task.context = [intake_task, underwriting_task]
```

---

## Task Design Patterns

### The Analysis Task

```python
Task(
    description="""Analyze [input] for [purpose].
    
    Consider:
    1. [Factor 1]
    2. [Factor 2]
    3. [Factor 3]
    
    Use [tool] to [specific action].""",
    expected_output="""Analysis containing:
    - [Finding type 1]
    - [Finding type 2]
    - Recommendation: [format]"""
)
```

### The Decision Task

```python
Task(
    description="""Make [decision type] based on previous analyses.
    
    Consider:
    - [Input 1] from [source 1]
    - [Input 2] from [source 2]
    
    Decision must be: [OPTION_A] / [OPTION_B] / [OPTION_C]
    
    Document reasoning clearly.""",
    expected_output="""Decision: [OPTION]
    Reasoning: [explanation]
    Conditions: [if applicable]"""
)
```

### The Report Task

```python
Task(
    description="""Synthesize all findings into a [report type].
    
    Include:
    1. Executive Summary
    2. Key Findings
    3. Detailed Analysis
    4. Recommendations
    
    Format for [audience].""",
    expected_output="""Professional report in markdown with:
    - Clear structure
    - Actionable insights
    - Evidence citations""",
    output_file="output/report.md"
)
```

---

## Output Files

Save task output to a file:

```python
Task(
    description="Generate the compliance report",
    expected_output="Full compliance report",
    agent=report_agent,
    output_file="output/compliance_report.md"  # Saved here
)
```

---

## Task Execution Details

### What the Agent Receives

When a task runs, the agent's prompt includes:

```
SYSTEM: You are [role]. Your goal is [goal]. [backstory]

TOOLS AVAILABLE:
- tool_1: description
- tool_2: description

YOUR TASK:
[description]

EXPECTED OUTPUT:
[expected_output]

CONTEXT FROM PREVIOUS TASKS:
---
Task: [previous_task_description]
Output: [previous_task_output]
---

Now complete your task.
```

### The Execution Loop

```
1. Agent reads task + context
2. Agent thinks about approach
3. Agent may call tools
4. Agent produces output
5. Output is validated against expected_output
6. Output stored for next task's context
```

---

## Tasks from Your Projects

### Loan Origination: Credit Analysis Task

```python
Task(
    description="""Perform comprehensive credit analysis for this loan application.
    
    Your responsibilities:
    1. Use the credit_check tool with the applicant's credit score and history
    2. Use the dti_calculator tool to compute debt-to-income ratio
    3. Evaluate credit tier and identify risk factors
    4. Assess ability to take on the proposed loan payment
    
    Use the applicant information from the previous tasks.""",
    expected_output="""A credit analysis report containing:
    - Credit score evaluation and tier
    - Debt-to-income ratio (current and proposed)
    - Credit risk factors (positive and negative)
    - Credit recommendation (PASS / CONDITIONAL / FAIL)
    - Specific concerns or strengths to highlight""",
    agent=credit_analyst_agent,
)
```

### Data Quality: Validation Task

```python
Task(
    description=f"""Validate the dataset against business rules and CDE requirements.

    DATA FILE: {data_file}
    CDE CONFIG: {cde_config}

    Validation checks must include:
    1. Format validation (emails, phones, dates, IDs)
    2. Range validation (credit scores 300-850, no negative balances)
    3. Date logic (no future dates of birth)
    4. CDE-specific rules
    
    Categorize all issues by severity: CRITICAL, HIGH, MEDIUM, LOW""",
    expected_output="""A validation report including:
    - Total records validated
    - Issue counts by severity
    - Detailed issue list with field, rule violated, count
    - CDE violations highlighted separately
    - Overall validity score""",
    agent=validator_agent,
)
```

---

## Common Mistakes

### 1. Vague Descriptions

```python
# Bad
description="Analyze the data"

# Good
description="""Analyze the credit data to determine loan eligibility.
1. Check credit score (must be > 620)
2. Calculate DTI (must be < 43%)
3. Verify no recent bankruptcies"""
```

### 2. Missing Expected Output

```python
# Bad
expected_output="Analysis"

# Good
expected_output="""Credit analysis containing:
- Score: [number] / Tier: [1-5]
- DTI: [percentage]
- Decision: ELIGIBLE / NOT_ELIGIBLE
- Key factors: [list]"""
```

### 3. Wrong Context

```python
# Bad - underwriter gets raw intake data
underwriting_task.context = [intake_task]

# Good - underwriter gets analyzed data
underwriting_task.context = [credit_task, risk_task]
```

---

## Next Up

Section 4: Tools — extending what agents can do.
