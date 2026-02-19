# Part 3, Section 1: Agentic Loan Origination

## Project Overview

A multi-agent system that processes loan applications through a complete underwriting workflow.

**GitHub:** `Dewale-A/AgenticLoanOrigination`

---

## The Business Problem

Loan origination involves multiple specialists:

```
Traditional Process (Human):
1. Document clerk receives application
2. Verification team checks information
3. Credit analyst evaluates creditworthiness
4. Risk team calculates risk scores
5. Underwriter makes decision
6. Loan officer structures the offer
```

**Each role requires different expertise and tools.**

---

## Architecture

```
                    Application JSON
                          |
                          v
              +-- Document Intake Agent --+
              |     (loads application)   |
              v                           
        +-- Verification Agent --+
        |   (validates info)     |
        v
  +-- Credit Analyst Agent --+
  |   (credit_check, dti)    |
  v
  +-- Risk Assessor Agent --+
  |   (risk_scoring)         |
  v
  +-- Underwriter Agent --+
  |   (APPROVE/DENY)      |
  v
  +-- Offer Generator Agent --+
  |   (loan_pricing)          |
  v
              Loan Decision + Offer
```

**6 Agents, Sequential Process**

---

## The Agents

### 1. Document Intake Agent

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

**Job:** First contact. Loads the JSON application, extracts key fields.

### 2. Verification Agent

```python
Agent(
    role="Verification Analyst",
    goal="Verify applicant information including income, employment, and identity",
    backstory="""You are a verification analyst responsible for ensuring all 
    applicant information is accurate and consistent. You cross-reference data 
    points to identify discrepancies and flag potential fraud indicators.""",
    tools=[application_loader],
)
```

**Job:** Checks consistency. Does income match employment? Are dates reasonable?

### 3. Credit Analyst Agent

```python
Agent(
    role="Senior Credit Analyst",
    goal="Analyze applicant creditworthiness and provide detailed credit assessment",
    backstory="""You are a senior credit analyst with over 10 years of experience 
    in consumer lending. You evaluate credit reports, payment histories, and 
    outstanding debts to determine creditworthiness.""",
    tools=[credit_check, dti_calculator],
)
```

**Job:** Evaluates credit score, calculates DTI, determines credit tier.

### 4. Risk Assessor Agent

```python
Agent(
    role="Risk Assessment Specialist",
    goal="Calculate comprehensive risk scores and identify all risk factors",
    backstory="""You are a risk assessment specialist who evaluates loan 
    applications using quantitative models and qualitative judgment.""",
    tools=[risk_scoring],
)
```

**Job:** Produces risk score (0-100) considering all factors.

### 5. Underwriter Agent

```python
Agent(
    role="Senior Loan Underwriter",
    goal="Make final credit decisions based on all available analysis",
    backstory="""You are a senior loan underwriter with authority to approve or 
    deny loan applications. You review all analysis, weigh compensating factors, 
    and make fair, defensible decisions.""",
    tools=[risk_scoring],
)
```

**Job:** The decision maker. APPROVED / APPROVED_WITH_CONDITIONS / DENIED.

### 6. Offer Generator Agent

```python
Agent(
    role="Loan Structuring Specialist",
    goal="Structure approved loans with optimal terms for both lender and borrower",
    backstory="""You are a loan structuring specialist who designs loan offers
    for approved applications. You balance profitability with competitive pricing.""",
    tools=[loan_pricing],
)
```

**Job:** If approved, calculates rate, term, monthly payment.

---

## The Tools

### Application Loader

```python
class ApplicationLoaderTool(BaseTool):
    name = "application_loader"
    description = "Load loan application by ID"
    
    def _run(self, application_id: str) -> str:
        with open(f"applications/{application_id}.json") as f:
            return json.dumps(json.load(f), indent=2)
```

### Credit Check

```python
class CreditCheckTool(BaseTool):
    name = "credit_check"
    description = "Evaluate credit score and return tier"
    
    def _run(self, credit_score: int, payment_history: int = 5) -> str:
        tier = self._get_tier(credit_score)
        return json.dumps({
            "credit_score": credit_score,
            "credit_tier": tier,
            "tier_description": self._tier_desc(tier)
        })
```

### DTI Calculator

```python
class DTICalculatorTool(BaseTool):
    name = "dti_calculator"
    description = "Calculate debt-to-income ratio"
    
    def _run(self, annual_income: float, monthly_debts: float, 
             proposed_payment: float) -> str:
        monthly_income = annual_income / 12
        current_dti = (monthly_debts / monthly_income) * 100
        proposed_dti = ((monthly_debts + proposed_payment) / monthly_income) * 100
        
        return json.dumps({
            "current_dti": round(current_dti, 2),
            "proposed_dti": round(proposed_dti, 2),
            "assessment": "PASS" if proposed_dti < 43 else "HIGH"
        })
```

### Risk Scoring

```python
class RiskScoringTool(BaseTool):
    name = "risk_scoring"
    description = "Calculate composite risk score"
    
    def _run(self, credit_score, dti, income, years_employed, 
             bankruptcies, loan_amount) -> str:
        # Weighted scoring model
        score = (
            self._credit_component(credit_score) * 0.35 +
            self._dti_component(dti) * 0.25 +
            self._income_component(income, loan_amount) * 0.20 +
            self._employment_component(years_employed) * 0.15 +
            self._history_component(bankruptcies) * 0.05
        )
        return json.dumps({
            "risk_score": round(score, 1),
            "risk_level": self._get_level(score)
        })
```

---

## Task Flow with Context

```python
# Task 1: Intake (no context - first task)
intake_task = create_intake_task(intake_agent, application_id)

# Task 2: Verification (needs intake)
verification_task = create_verification_task(verifier, application_id)
verification_task.context = [intake_task]

# Task 3: Credit (needs intake + verification)
credit_task = create_credit_analysis_task(credit_analyst)
credit_task.context = [intake_task, verification_task]

# Task 4: Risk (needs intake + credit)
risk_task = create_risk_assessment_task(risk_assessor)
risk_task.context = [intake_task, credit_task]

# Task 5: Underwriting (needs verification + credit + risk)
underwriting_task = create_underwriting_task(underwriter)
underwriting_task.context = [verification_task, credit_task, risk_task]

# Task 6: Offer (needs intake + decision)
offer_task = create_offer_generation_task(offer_generator)
offer_task.context = [intake_task, underwriting_task]
```

---

## Sample Application

```json
{
  "application_id": "APP001",
  "applicant": {
    "name": "John Smith",
    "ssn_last_four": "1234",
    "date_of_birth": "1985-03-15",
    "email": "john.smith@email.com"
  },
  "loan_request": {
    "amount": 25000,
    "purpose": "debt_consolidation",
    "term_months": 60
  },
  "financial": {
    "annual_income": 85000,
    "employment_status": "full_time",
    "employer": "Tech Corp",
    "years_employed": 5,
    "credit_score": 720,
    "monthly_debts": 1200
  }
}
```

---

## Sample Output

```
=== LOAN DECISION ===

Application: APP001
Applicant: John Smith

DECISION: APPROVED

Loan Terms:
- Amount: $25,000
- APR: 8.99%
- Term: 60 months
- Monthly Payment: $518.96
- Total Interest: $6,137.60

Key Factors:
+ Strong credit score (720, Tier 2)
+ Stable employment (5 years)
+ DTI within limits (28% current, 35% proposed)
- Higher debt consolidation amount

Conditions:
- Verify employment within 30 days of closing
- Provide two recent pay stubs
```

---

## Running the Project

```bash
# Clone
git clone https://github.com/Dewale-A/AgenticLoanOrigination.git
cd AgenticLoanOrigination

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add OPENAI_API_KEY

# Run
python main.py APP001
```

---

## Key Learnings

1. **Sequential process mirrors real workflow** — Each agent does what a human specialist would do
2. **Context passing is selective** — Underwriter doesn't need raw application, just analyses
3. **Tools enable real calculations** — DTI and risk scoring are actual formulas
4. **Clear decisions** — APPROVED/DENIED with documented reasoning

---

## Next Up

Section 2: Policy Documents Application — compliance analysis with multi-agent AI.
