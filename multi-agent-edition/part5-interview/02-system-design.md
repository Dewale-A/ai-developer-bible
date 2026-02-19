# Part 5, Section 2: System Design Scenarios

## Scenario 1: Automated Insurance Claims Processing

**Prompt:** "Design a multi-agent system to process insurance claims."

### Requirements Clarification

Ask:
- What types of claims? (Auto, health, property)
- What's the volume? (1000/day)
- What decisions are needed? (Approve, deny, investigate)
- Integration requirements? (Policy database, payment system)

### Architecture

```
Claim Submission
      |
      v
[Document Intake Agent]
  - Extract claim details
  - Identify claim type
  - Check completeness
      |
      v
[Policy Verification Agent]
  - Verify coverage
  - Check policy status
  - Identify exclusions
      |
      v
[Fraud Detection Agent]
  - Pattern analysis
  - Red flag identification
  - Risk scoring
      |
      v
[Assessment Agent]
  - Damage evaluation
  - Coverage calculation
  - Deductible application
      |
      v
[Decision Agent]
  - Approve/Deny/Investigate
  - Calculate payout
  - Generate explanation
      |
      v
[Communication Agent]
  - Draft customer letter
  - Prepare internal report
```

### Key Design Decisions

1. **Sequential process** — Each step depends on previous
2. **Fraud detection early** — Before spending time on assessment
3. **Separate communication** — Customer-facing vs internal needs differ
4. **Tools**: Policy lookup, fraud pattern database, payout calculator

---

## Scenario 2: Regulatory Compliance Monitoring

**Prompt:** "Design a system to monitor regulatory changes and assess impact."

### Architecture

```
                Regulatory Sources
               (SEC, Fed, State sites)
                        |
                        v
           [Monitoring Agent] (scheduled)
             - Scan for new regulations
             - Extract key changes
             - Classify by domain
                        |
                        v
           [Impact Analysis Agent]
             - Map to internal policies
             - Identify affected processes
             - Assess compliance gaps
                        |
                        v
           [Prioritization Agent]
             - Risk scoring
             - Deadline tracking
             - Resource estimation
                        |
                        v
           [Report Agent]
             - Executive summary
             - Detailed action items
             - Timeline recommendations
```

### Key Decisions

1. **Scheduled execution** — Run monitoring daily/weekly
2. **Hierarchical for analysis** — Different regs need different expertise
3. **Tools**: Web scraping, internal policy search, compliance database
4. **Output**: Actionable report with deadlines

---

## Scenario 3: Customer Support Escalation

**Prompt:** "Design a multi-agent system for complex support tickets."

### Architecture

```
                Escalated Ticket
                      |
                      v
             [Triage Agent]
               - Categorize issue
               - Assess urgency
               - Identify customer tier
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
    [Technical    [Billing    [Product
     Agent]        Agent]      Agent]
          |           |           |
          +-----------+-----------+
                      |
                      v
             [Resolution Agent]
               - Synthesize findings
               - Propose solution
               - Draft response
                      |
                      v
             [QA Agent]
               - Verify accuracy
               - Check tone
               - Ensure completeness
```

### Key Decisions

1. **Parallel specialists** — Technical, billing, product can work simultaneously
2. **QA at the end** — Catch errors before customer sees response
3. **Customer context** — Pass account history to all agents
4. **Escalation path** — If confidence low, flag for human review

---

## Scenario 4: M&A Due Diligence

**Prompt:** "Design a system to analyze target companies for acquisition."

### Architecture

```
            Target Company Data
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
  [Financial    [Legal      [Operations
   Analyst]     Analyst]    Analyst]
   - Revenue    - Contracts  - Supply chain
   - Margins    - Litigation - Capacity
   - Debt       - IP rights  - Technology
        |           |           |
        +-----------+-----------+
                    |
                    v
           [Risk Assessment Agent]
             - Consolidated risks
             - Deal breakers
             - Mitigation options
                    |
                    v
           [Valuation Agent]
             - Financial models
             - Synergy estimates
             - Price recommendation
                    |
                    v
           [Report Agent]
             - Executive summary
             - Go/No-go recommendation
             - Key findings
```

### Key Decisions

1. **Parallel analysis** — Financial, legal, operations are independent
2. **Risk consolidation** — Separate agent to synthesize risks
3. **Valuation last** — Needs all inputs
4. **Tools**: Financial data APIs, legal database search, industry benchmarks

---

## Design Framework

For any multi-agent system design:

### 1. Process Mapping (2 min)
- How is this done manually today?
- What are the distinct phases?

### 2. Agent Identification (3 min)
- What roles are needed?
- What expertise does each need?
- What tools does each need?

### 3. Flow Design (3 min)
- Sequential, parallel, or hierarchical?
- What context does each agent need?
- Where are the decision points?

### 4. Error Handling (2 min)
- What if an agent fails?
- Where are human checkpoints?
- How do you gracefully degrade?

### 5. Scale & Cost (2 min)
- How many executions per day?
- Which agents need powerful models?
- What's the cost per execution?

---

## Common Follow-ups

**"How would you handle errors?"**
> Tools return error strings. Crew wrapped in try-catch. Graceful degradation — return partial results. Human escalation for low-confidence decisions.

**"How would you scale this?"**
> Run crews in parallel for independent cases. Cache repeated lookups. Use cheaper models for simple agents. Batch similar cases.

**"How would you measure success?"**
> Accuracy vs human decisions. Processing time. Cost per case. Error rate. Customer satisfaction for support scenarios.

**"What if latency is a concern?"**
> Reduce agents. Parallel where possible. Faster models. Pre-compute common patterns. Stream partial results.

---

## Next Up

Section 3: Code Explanation Practice — walking through your implementations.
