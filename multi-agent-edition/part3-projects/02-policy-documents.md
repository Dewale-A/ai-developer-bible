# Part 3, Section 2: Policy Documents Application

## Project Overview

A multi-agent system that analyzes policy documents for compliance gaps and generates actionable reports.

**GitHub:** `Dewale-A/AgenticAI-Policy-Documents-Application`

---

## The Business Problem

Organizations have policy documents spread across systems:
- Data governance policies
- Privacy policies
- Risk management frameworks
- Regulatory compliance docs

**Challenges:**
- Documents may be outdated
- Gaps in regulatory coverage
- Inconsistencies between policies
- Hard to assess overall compliance posture

---

## Architecture

```
        Policy Documents Directory
                  |
                  v
     +-- Ingestion Agent --+
     |   (reads documents) |
     v
     +-- Analysis Agent --+
     |   (finds gaps)     |
     v
     +-- Report Agent --+
     |   (writes report) |
     v
         Compliance Report
```

**3 Agents, Sequential Process**

---

## The Agents

### 1. Ingestion Agent

```python
Agent(
    role="Policy Document Ingestion Specialist",
    goal="""Thoroughly read and extract all relevant information from policy 
    documents. Identify key sections, requirements, controls, and compliance 
    obligations. Organize extracted information for analysis.""",
    backstory="""You are an expert document analyst with years of experience 
    in financial services and regulatory compliance. You have a keen eye for 
    detail and can quickly identify important policy requirements, controls, 
    and obligations. You understand regulatory frameworks like GDPR, SOX, 
    Basel III, and industry standards for data governance.""",
    tools=[DocumentReaderTool(), DocumentSearchTool()],
    allow_delegation=False,
)
```

**Job:** Discover and read all policy documents, extract structure and requirements.

### 2. Analysis Agent

```python
Agent(
    role="Policy Compliance Analyst",
    goal="""Analyze policy documents to identify compliance requirements, gaps, 
    and areas of concern. Map policies to relevant regulatory frameworks and 
    assess organizational compliance posture.""",
    backstory="""You are a senior compliance analyst with deep expertise in 
    financial regulations, data governance, and enterprise risk management. 
    You have helped numerous organizations navigate complex regulatory 
    landscapes and implement effective compliance programs.""",
    tools=[DocumentSearchTool()],
    allow_delegation=True,
)
```

**Job:** Map policies to regulations, find gaps, assess risk levels.

### 3. Report Agent

```python
Agent(
    role="Compliance Report Writer",
    goal="""Create clear, comprehensive, and actionable compliance reports 
    based on policy analysis. Provide executive summaries for leadership and 
    detailed findings for implementation teams.""",
    backstory="""You are a skilled technical writer with expertise in 
    compliance reporting and executive communications. You can distill 
    complex regulatory analysis into clear, actionable insights.""",
    allow_delegation=False,
)
```

**Job:** Synthesize findings into professional report with recommendations.

---

## The Tools

### Document Reader Tool

```python
class DocumentReaderTool(BaseTool):
    name = "document_reader"
    description = """Read policy documents from the policy_documents directory.
    
    Args:
        filename: Name of file to read, or 'list' to see available files
        
    Returns:
        Document content or list of available documents."""
    
    def _run(self, filename: str = "list") -> str:
        docs_dir = "policy_documents"
        
        if filename == "list":
            files = [f for f in os.listdir(docs_dir) 
                     if f.endswith('.md')]
            return f"Available documents:\n" + "\n".join(f"- {f}" for f in files)
        
        filepath = os.path.join(docs_dir, filename)
        if not os.path.exists(filepath):
            return f"Error: {filename} not found"
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        return f"=== {filename} ===\n\n{content}"
```

### Document Search Tool

```python
class DocumentSearchTool(BaseTool):
    name = "document_search"
    description = """Search across all policy documents for specific terms.
    
    Args:
        search_term: Term to search for
        
    Returns:
        Matching excerpts with document names."""
    
    def _run(self, search_term: str) -> str:
        results = []
        docs_dir = "policy_documents"
        
        for filename in os.listdir(docs_dir):
            filepath = os.path.join(docs_dir, filename)
            with open(filepath, 'r') as f:
                content = f.read()
            
            if search_term.lower() in content.lower():
                # Find relevant excerpt
                idx = content.lower().find(search_term.lower())
                excerpt = content[max(0, idx-100):idx+200]
                results.append(f"[{filename}]: ...{excerpt}...")
        
        if not results:
            return f"No matches found for '{search_term}'"
        
        return "\n\n".join(results)
```

---

## Sample Policy Documents

### Data Governance Policy

```markdown
# Data Governance Policy

## Purpose
Establish framework for data management across the organization.

## Scope
All business units handling customer or financial data.

## Data Classification
- Public: Marketing materials
- Internal: Business operations
- Confidential: Customer PII
- Restricted: Financial records, credentials

## Roles and Responsibilities
- Data Owner: Business accountability
- Data Steward: Day-to-day management
- Data Custodian: Technical implementation

## Compliance Requirements
- GDPR Article 5: Data processing principles
- SOX Section 404: Internal controls
- Basel III: Operational risk data
```

---

## Task Definitions

### Ingestion Task

```python
Task(
    description="""Perform comprehensive ingestion of all policy documents.
    
    1. List all available policy documents
    2. Read each document thoroughly
    3. For each document, identify:
       - Document title and purpose
       - Key policy statements
       - Compliance obligations
       - Referenced regulations
    4. Note cross-references between documents
    5. Flag unclear or incomplete areas""",
    expected_output="""Structured extraction containing:
    1. Document inventory with metadata
    2. Key requirements by theme
    3. Compliance controls
    4. Cross-reference map
    5. Initial observations"""
)
```

### Analysis Task

```python
Task(
    description="""Analyze extracted policy content for compliance posture.
    
    1. Regulatory Mapping
       - Map policies to GDPR, SOX, Basel, etc.
       - Identify coverage and gaps
    
    2. Gap Analysis
       - Missing policies
       - Inconsistencies
       - Outdated sections
    
    3. Risk Assessment
       - Prioritize gaps by impact
       - Consider enforcement trends
    
    4. Control Effectiveness
       - Are stated controls adequate?
       - Missing enforcement mechanisms?""",
    expected_output="""Analysis report containing:
    1. Regulatory mapping matrix
    2. Prioritized gap inventory
    3. Control assessment
    4. Risk heat map"""
)
```

### Report Task

```python
Task(
    description="""Generate professional compliance report.
    
    EXECUTIVE SUMMARY:
    - Overall compliance score
    - Key findings (top 5)
    - Strategic recommendations
    
    DETAILED FINDINGS:
    - Complete gap analysis
    - Regulatory mapping
    - Remediation roadmap
    
    APPENDICES:
    - Document inventory
    - Glossary""",
    expected_output="Professional compliance report in markdown",
    output_file="output/compliance_report.md"
)
```

---

## Configurable Options

```python
def create_policy_analysis_crew(
    document_focus: str = None,    # Focus on specific doc/topic
    focus_areas: list = None,      # e.g., ["GDPR", "data governance"]
    report_type: str = "full"      # "executive", "detailed", "full"
) -> Crew:
```

---

## Sample Output

```markdown
# Compliance Assessment Report

## Executive Summary

**Overall Compliance Score: 72/100**

### Key Findings
1. **CRITICAL**: No data retention policy defined
2. **HIGH**: Privacy policy missing CCPA requirements  
3. **HIGH**: Risk framework lacks operational risk metrics
4. **MEDIUM**: Data classification inconsistent across docs
5. **LOW**: Document version control needs improvement

### Recommended Actions
1. Develop data retention policy (2 weeks)
2. Update privacy policy for CCPA (1 week)
3. Add operational risk metrics to framework (3 weeks)

## Regulatory Coverage Matrix

| Regulation | Coverage | Gaps |
|------------|----------|------|
| GDPR | 75% | Retention, DPIA |
| SOX | 80% | IT controls documentation |
| Basel III | 60% | Operational risk data |
| CCPA | 40% | Opt-out, disclosure |

## Detailed Findings
...
```

---

## Running the Project

```bash
# Clone
git clone https://github.com/Dewale-A/AgenticAI-Policy-Documents-Application.git
cd AgenticAI-Policy-Documents-Application

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add OPENAI_API_KEY (or ANTHROPIC_API_KEY)

# Run
python main.py

# With options
python main.py --focus "data governance" --report-type executive
```

---

## Key Learnings

1. **Document tools are simple** — Just file I/O, but critical for agent access
2. **Analysis agent does the heavy lifting** — Mapping, gaps, risk assessment
3. **Delegation enabled** — Analysis agent can ask for more document searches
4. **Output file** — Report saved directly to markdown file

---

## Next Up

Section 3: Agentic Data Quality — automated data assessment with CDE focus.
