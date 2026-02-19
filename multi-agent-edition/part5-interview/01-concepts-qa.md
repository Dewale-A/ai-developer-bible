# Part 5, Section 1: Multi-Agent Concepts Q&A

## Core Concept Questions

### Q: What is a multi-agent system and why use it?

**Strong Answer:**
> A multi-agent system uses multiple specialized AI agents working together instead of one agent doing everything. Each agent has a specific role, goal, and tools.
>
> I use multi-agent when:
> 1. The task has distinct roles (like loan processing with intake, credit, underwriting)
> 2. One agent would get overwhelmed with context
> 3. I need different expertise for different parts
> 4. I want to debug and improve components independently
>
> For my loan origination project, I have 6 agents because each mirrors a real specialist — intake, verification, credit analyst, risk assessor, underwriter, and offer generator.

---

### Q: Explain the difference between sequential and hierarchical orchestration.

**Strong Answer:**
> In **sequential**, tasks run one after another in a fixed order. Agent A finishes, then Agent B starts with A's output. It's predictable and great for workflows with clear steps.
>
> In **hierarchical**, a manager agent receives the task and dynamically delegates to worker agents. The manager decides which workers to use and how to combine their results.
>
> I use sequential for business workflows like loan processing because they mirror existing processes. I'd use hierarchical for open-ended research where the approach isn't predetermined.

---

### Q: How do agents communicate in CrewAI?

**Strong Answer:**
> Through context passing. When I define a task, I set `task.context = [previous_task]`. The agent receives the previous task's output in its prompt.
>
> I'm selective about context — I don't pass everything to every agent. For example, my underwriter agent gets context from verification, credit, and risk tasks, but not the raw intake. This keeps context manageable and focused.

---

### Q: What makes a good agent backstory?

**Strong Answer:**
> A good backstory establishes expertise level, working style, and quality expectations. It shapes how the LLM approaches the task.
>
> For example, my credit analyst has: "10 years of consumer lending experience, known for thorough, balanced assessments." This tells the LLM to be experienced and balanced, not aggressive or superficial.
>
> Key elements: years of experience, specific domain, working style, quality standards. Keep it focused — a paragraph, not a page.

---

### Q: How do tools extend agent capabilities?

**Strong Answer:**
> LLMs can think but can't act. Tools bridge that gap — they're functions agents can call.
>
> For example, my credit analyst has a `credit_check` tool that calculates credit tiers from a score, and a `dti_calculator` that computes debt-to-income ratios. The agent reasons about what to do, calls the tools, and interprets results.
>
> Good tools have clear names, detailed descriptions (so agents know when to use them), and return structured output the agent can parse.

---

## Architecture Questions

### Q: Walk me through how you'd design a multi-agent system.

**Strong Answer:**
> I follow four steps:
>
> 1. **Map the process**: How do humans do this today? For loan processing, I mapped out intake → verification → credit → risk → decision → offer.
>
> 2. **Define agents**: One agent per major role. Each gets a role, goal, backstory, and relevant tools.
>
> 3. **Design tasks**: What does each agent need to do? What inputs do they need? What outputs should they produce?
>
> 4. **Connect with context**: Which tasks depend on which? I set `task.context` to pass information between agents.
>
> Then I test iteratively — run with verbose mode, see what works, adjust.

---

### Q: How do you decide how many agents to use?

**Strong Answer:**
> I start with the minimum that makes sense. Each agent adds latency and cost.
>
> I ask: "Can this task be split into distinct roles with different expertise?" If yes, those become agents. If roles blur together, I keep them as one agent.
>
> For loan origination, I have 6 agents because each has genuinely different expertise and tools. For policy analysis, I only need 3 — ingestion, analysis, reporting.
>
> I wouldn't use 6 agents if 3 could do it. I also wouldn't force everything into 1 agent if it makes the task too complex.

---

### Q: What's the role of context in task design?

**Strong Answer:**
> Context is how information flows between agents. But more isn't always better.
>
> If I pass all previous outputs to every agent, context grows huge — slower, more expensive, and agents lose focus. Instead, I'm selective.
>
> My underwriter doesn't need the raw application data — they need the analyzed results from verification, credit, and risk. So I set:
> ```python
> underwriting_task.context = [verification_task, credit_task, risk_task]
> ```
>
> Not `[intake_task, verification_task, credit_task, risk_task]`.

---

## Practical Questions

### Q: How do you handle errors in multi-agent systems?

**Strong Answer:**
> At multiple levels:
>
> 1. **Tools** return error strings instead of throwing exceptions. So if a file isn't found, the agent sees "Error: not found" and can reason about it.
>
> 2. **Crew execution** is wrapped in try-catch. If something fails, I log it and can return partial results.
>
> 3. **Graceful degradation**: If credit analysis fails, I might still return verification results and flag for manual review.
>
> 4. **Max iterations**: I set limits so agents don't loop forever if confused.

---

### Q: How do you manage costs?

**Strong Answer:**
> Several strategies:
>
> 1. **Right-size models**: GPT-4o-mini for simple tasks, GPT-4o only for complex decisions.
>
> 2. **Minimize context**: Only pass what each agent needs.
>
> 3. **Concise prompts**: Keep backstories focused.
>
> 4. **Caching**: Don't re-run identical tool calls.
>
> 5. **Tracking**: Log token usage, set budget alerts.
>
> My loan origination crew costs about $0.03-0.08 per application with GPT-4o-mini. Using GPT-4o everywhere would be 5-10x more.

---

### Q: How do you test multi-agent systems?

**Strong Answer:**
> Layered approach:
>
> 1. **Tool unit tests**: Tools are deterministic, so I test them traditionally.
>
> 2. **Output structure tests**: Does the result contain required fields? I don't check exact values because LLMs vary.
>
> 3. **Smoke tests**: Does the crew complete without errors?
>
> 4. **Regression tests**: I save known-good outputs and check that key patterns still appear.
>
> 5. **Agent isolation**: Test each agent individually before testing the full crew.
>
> Integration tests with real LLMs run on CI but less frequently due to cost.

---

### Q: When would you NOT use multi-agent?

**Strong Answer:**
> When it's overkill:
>
> - Simple tasks that one agent handles fine
> - Latency-critical applications (multi-agent adds seconds)
> - Tight budgets (more agents = more LLM calls)
> - No clear role separation
>
> I start with single agent and only go multi-agent when I hit specific pain points: context overflow, need for distinct expertise, or debugging difficulty.

---

## Next Up

Section 2: System Design Scenarios — whiteboard problems for multi-agent.
