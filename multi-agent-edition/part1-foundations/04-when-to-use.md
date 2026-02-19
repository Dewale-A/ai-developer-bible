# Part 1, Section 4: When to Use Multi-Agent

## The Decision Framework

Multi-agent systems add complexity. Use them when benefits outweigh costs.

---

## Use Multi-Agent When...

### 1. The Task Has Distinct Roles

```
Loan Processing:
- Intake Specialist (documents)
- Verification Analyst (identity)
- Credit Analyst (creditworthiness)
- Risk Assessor (scoring)
- Underwriter (decision)
- Offer Generator (terms)

Each role has different expertise and tools.
```

### 2. Context Would Overwhelm One Agent

```
Single agent for loan:
- Must remember: application data, verification results,
  credit analysis, risk factors, underwriting rules,
  pricing models, compliance requirements...
  
Token limit: 128K
Your context: Might exceed it on complex cases
```

### 3. You Need Separation of Concerns

```
Want to improve credit analysis?
- Multi-agent: Update credit_analyst_agent only
- Single agent: Risk breaking everything else
```

### 4. Debugging Needs Clarity

```
"The loan decision was wrong"

Multi-agent: Check each agent's output
- Intake: Correct
- Verification: Correct
- Credit: Bug found here!

Single agent: "Something in this 5000-line output is wrong"
```

### 5. Different Parts Need Different Models

```python
# Fast model for simple tasks
intake_agent = Agent(llm=LLM(model="gpt-4o-mini"))

# Powerful model for complex analysis
underwriter_agent = Agent(llm=LLM(model="gpt-4o"))

# Specialized model for final polish
editor_agent = Agent(llm=LLM(model="gpt-4o"))
```

---

## DON'T Use Multi-Agent When...

### 1. Task is Simple

```
"Summarize this document"
"Answer this question"
"Translate this text"

One agent is sufficient. Don't over-engineer.
```

### 2. Latency is Critical

```
Multi-agent loan processing:
- 6 agents x ~2 seconds each = ~12 seconds minimum

If you need sub-second responses, multi-agent adds too much latency.
```

### 3. Budget is Tight

```
6 agents processing one loan:
- 6 LLM calls (input + output tokens)
- Potentially 6x the cost of one agent

Single agent: 1 call
Multi-agent: 6+ calls
```

### 4. No Clear Role Separation

```
"I need an agent to help with stuff"

If you can't name distinct roles, you don't need multiple agents.
```

### 5. You're Just Starting

```
Day 1: Build single agent, see what works
Day 30: Hit limitations, consider multi-agent
Day 60: Refactor to multi-agent with clear roles

Don't start with multi-agent complexity.
```

---

## The Complexity Tradeoff

```
Complexity
    ^
    |
    |     Multi-Agent
    |       /
    |      /
    |     /
    |    /  Single Agent
    |   /
    |  /
    | /
    +-------------------------> Problem Difficulty
    
Simple problems: Single agent wins (less overhead)
Complex problems: Multi-agent wins (better quality)
```

---

## Decision Checklist

Before going multi-agent, answer these:

- [ ] Can you name 3+ distinct roles?
- [ ] Would one agent exceed context limits?
- [ ] Do different parts need different expertise?
- [ ] Is debugging difficulty a concern?
- [ ] Can you accept higher latency (5-30 seconds)?
- [ ] Can you accept higher cost (3-10x)?
- [ ] Is this a repeating workflow (not one-off)?

**If 4+ yes: Consider multi-agent**
**If 2 or fewer yes: Stick with single agent**

---

## Hybrid Approaches

You don't have to go all-in:

### Start Simple, Scale Up

```
v1: Single agent does everything
v2: Split into 2 agents (analysis + report)
v3: Full multi-agent pipeline
```

### Multi-Agent for Core, Single for Edges

```
[Simple Intake] --> [Multi-Agent Core Processing] --> [Simple Output]
```

### Optional Agents

```python
class DataQualityCrew:
    def __init__(self, polish: bool = False):
        # Core agents always present
        self.profiler = create_profiler_agent()
        self.validator = create_validator_agent()
        
        # Optional agent for special cases
        if polish:
            self.editor = create_editor_agent()
```

---

## Cost-Benefit Analysis

### Example: Loan Origination

**Single Agent:**
- Tokens: ~10K input + ~2K output = ~12K tokens
- Cost: ~$0.02 per application
- Time: ~15 seconds
- Quality: Medium (misses nuances)

**Multi-Agent (6 agents):**
- Tokens: ~60K input + ~12K output = ~72K tokens
- Cost: ~$0.12 per application
- Time: ~45 seconds
- Quality: High (thorough analysis)

**Decision:** For a financial institution processing loans, the 6x cost increase is negligible compared to the risk of bad decisions. Multi-agent is worth it.

### Example: Chat Support

**Single Agent:**
- Cost: $0.01 per conversation
- Time: 2 seconds
- Quality: Good for common questions

**Multi-Agent:**
- Cost: $0.05 per conversation
- Time: 10 seconds
- Quality: Better, but users don't wait

**Decision:** Single agent for real-time chat. Multi-agent for complex escalations.

---

## Interview Question

**Q: How do you decide between single and multi-agent architecture?**

Strong answer:
> "I start with single agent and only go multi-agent when I hit specific pain points: context overflow, need for distinct expertise, debugging difficulty, or different parts needing different models. 
>
> For my loan origination project, I chose multi-agent because the process has clear roles — intake, verification, credit, risk, underwriting, offer. Each role has different tools and expertise. A single agent trying to do all of this would lose track of earlier analysis and make inconsistent decisions.
>
> But for simple tasks like summarization, I'd never use multi-agent — it's over-engineering."

---

## Next Up

Part 2: CrewAI Framework Deep Dive — the tools to build multi-agent systems.
