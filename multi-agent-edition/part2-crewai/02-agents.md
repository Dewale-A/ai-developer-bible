# Part 2, Section 2: Agents — Roles, Goals, Backstories

## The Three Pillars of Agent Identity

Every effective agent needs:
1. **Role** — What they are
2. **Goal** — What they're trying to achieve
3. **Backstory** — Why they're qualified

---

## Role: The Job Title

The role tells the LLM what persona to adopt.

### Good Roles

```python
role="Senior Credit Analyst"
role="Policy Compliance Analyst"
role="Data Quality Report Writer"
role="Loan Underwriter"
```

Specific, professional, clear expertise.

### Bad Roles

```python
role="Helper"          # Too vague
role="AI Assistant"    # No expertise
role="Agent 1"         # Meaningless
```

### The Role's Effect

The role shapes how the LLM responds:

```python
# Same question, different roles:

role="Junior Intern"
# Response: Basic, uncertain, asks questions

role="Senior Credit Analyst with 10 years experience"  
# Response: Confident, thorough, uses industry terms
```

---

## Goal: The Mission

The goal defines success for this agent.

### Good Goals

```python
goal="Analyze applicant creditworthiness and provide detailed credit assessment"
goal="Identify all data quality issues with special attention to Critical Data Elements"
goal="Make final underwriting decisions that balance risk management with customer service"
```

**Characteristics:**
- Specific outcome
- Clear scope
- Measurable (implicitly)

### Bad Goals

```python
goal="Help with stuff"          # Too vague
goal="Do your best"             # No direction
goal="Process everything"       # No focus
```

### Advanced: Goals with Constraints

```python
goal="""Analyze creditworthiness thoroughly while:
1. Flagging any fraud indicators immediately
2. Considering compensating factors for borderline cases
3. Documenting reasoning for audit purposes"""
```

---

## Backstory: The Experience

The backstory is the most powerful tuning lever. It shapes agent behavior.

### Example: Credit Analyst

```python
backstory="""You are a senior credit analyst with over 10 years of experience 
in consumer lending. You evaluate credit reports, payment histories, and 
outstanding debts to determine creditworthiness. Your analysis forms the 
foundation of lending decisions and you're known for thorough, balanced 
assessments."""
```

**What this does:**
- Establishes expertise level
- Defines working style ("thorough, balanced")
- Sets expectations ("foundation of lending decisions")

### Example: Risk Assessor

```python
backstory="""You are a risk assessment specialist who evaluates loan applications
using quantitative models and qualitative judgment. You consider credit metrics,
income stability, employment history, and market conditions to produce accurate
risk assessments that protect the institution while treating applicants fairly."""
```

**Notice:**
- Multiple skills (quantitative + qualitative)
- Multiple factors to consider
- Dual objective (protect institution + fair to applicants)

### Example: Report Writer

```python
backstory="""You are a Technical Writer specializing in data governance documentation.
You have the unique ability to translate complex technical findings into clear, 
business-friendly reports that executives can understand and act upon. Your reports 
are known for being thorough yet accessible, helping organizations understand their 
compliance posture and next steps."""
```

**The Effect:**
- Output will be clear, not overly technical
- Will include actionable next steps
- Balances thoroughness with accessibility

---

## Agent Design Patterns

### The Expert Pattern

```python
Agent(
    role="Senior [Domain] Expert",
    goal="Provide expert [domain] analysis",
    backstory="20 years of experience in [domain]..."
)
```

### The Specialist Pattern

```python
Agent(
    role="[Specific Function] Specialist",
    goal="Execute [function] with high accuracy",
    backstory="Specializes exclusively in [function]..."
)
```

### The Reviewer Pattern

```python
Agent(
    role="Senior [Domain] Reviewer",
    goal="Review and improve [output type]",
    backstory="Known for catching errors others miss..."
)
```

---

## Agents from Your Projects

### Loan Origination: Document Intake Agent

```python
Agent(
    role="Document Intake Specialist",
    goal="Load and organize loan application documents, extracting all relevant data",
    backstory="""You are an experienced document intake specialist at a lending 
    institution. Your job is to receive loan applications, ensure all required 
    information is present, and organize the data for downstream processing. 
    You're meticulous about details and flag any missing or inconsistent 
    information immediately.""",
    tools=[application_loader],
)
```

### Policy Documents: Analysis Agent

```python
Agent(
    role="Policy Compliance Analyst",
    goal="Analyze policy documents to identify compliance requirements, gaps, and areas of concern",
    backstory="""You are a senior compliance analyst with deep expertise in 
    financial regulations, data governance, and enterprise risk management. 
    You have helped numerous organizations navigate complex regulatory landscapes 
    and implement effective compliance programs.""",
    tools=[DocumentSearchTool()],
    allow_delegation=True,
)
```

### Data Quality: Validator Agent

```python
Agent(
    role="Data Validator",
    goal="Validate data against business rules and CDE requirements",
    backstory="""You are a meticulous Data Validator specializing in regulatory 
    compliance and data integrity. With experience across banking, healthcare, 
    and financial services, you understand the business impact of data quality 
    issues. You treat CDEs with extra scrutiny as they directly impact business 
    decisions and regulatory reporting.""",
    tools=[ValidatorTool()],
)
```

---

## Tips for Writing Backstories

### DO:
- Mention years of experience
- Include specific domains
- Describe working style
- Set quality expectations
- Reference relevant skills

### DON'T:
- Write a novel (keep it focused)
- Include irrelevant details
- Be vague about expertise
- Forget the goal alignment

### Template:

```python
backstory="""You are a [level] [role] with [X years] of experience in [domain].
You specialize in [specific skills]. Your work is known for [quality attributes].
You [key behavior that aligns with goal]."""
```

---

## Interview Questions

**Q: Why do backstories matter for agents?**

> Backstories establish expertise level, working style, and implicit quality standards. An agent with "10 years of credit analysis experience" will produce more thorough, industry-appropriate output than one with no backstory. It's prompt engineering at the identity level.

**Q: How do you decide what goes in an agent's goal vs backstory?**

> Goal is the outcome you want — what success looks like. Backstory is the expertise and style that gets you there. "Analyze creditworthiness" is the goal. "Senior analyst known for thorough assessments" is the backstory that shapes how the goal is achieved.

---

## Next Up

Section 3: Tasks — defining work and managing dependencies.
