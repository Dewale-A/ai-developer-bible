# Part 3, Section 3: Agentic Data Quality

## Project Overview

A multi-agent system that assesses data quality with special focus on Critical Data Elements (CDEs).

**GitHub:** `Dewale-A/AgenticDataQuality`

---

## The Business Problem

Data quality issues cost organizations millions:
- Bad decisions from bad data
- Regulatory fines for data governance failures
- Customer trust erosion
- Operational inefficiencies

**Critical Data Elements (CDEs)** are the most important fields — customer IDs, account numbers, transaction amounts. These need extra scrutiny.

---

## Architecture

```
            Data File (CSV)
                  |
           CDE Config (JSON)
                  |
     +-------------------------+
     |                         |
     v                         v
+-- Profiler --+        +-- Validator --+
|  (statistics)|        |  (rules)      |
+------+-------+        +-------+-------+
       |                        |
       |    +-- Anomaly --+     |
       |    |  (outliers) |     |
       |    +------+------+     |
       |           |            |
       +-----+-----+-----+------+
             |
             v
    +-- Report Writer --+
    |  (synthesize)     |
    +--------+----------+
             |
             v (optional)
    +-- Senior Editor --+
    |  (polish)         |
    +-------------------+
             |
             v
      Quality Report
```

**4-5 Agents (Editor optional), Sequential Process**

---

## The Agents

### 1. Data Profiler Agent

```python
Agent(
    role="Data Profiler",
    goal="Thoroughly profile datasets with special attention to CDEs",
    backstory="""You are an expert Data Profiler with 15 years of experience 
    in data governance. You have a keen eye for identifying data patterns,
    anomalies, and quality issues. You understand the critical importance 
    of CDEs in enterprise data management.""",
    tools=[DataLoaderTool(), CDELoaderTool(), ProfilerTool()],
)
```

**Job:** Analyze data structure, completeness, distributions. Flag CDE issues.

### 2. Data Validator Agent

```python
Agent(
    role="Data Validator",
    goal="Validate data against business rules and CDE requirements",
    backstory="""You are a meticulous Data Validator specializing in 
    regulatory compliance and data integrity. You've developed validation
    frameworks used by Fortune 500 companies and prevented millions in 
    regulatory fines. You treat CDEs with extra scrutiny.""",
    tools=[ValidatorTool()],
)
```

**Job:** Check formats, ranges, business rules. Categorize by severity.

### 3. Anomaly Detector Agent

```python
Agent(
    role="Anomaly Detector",
    goal="Identify statistical outliers and unusual patterns",
    backstory="""You are a Statistical Analyst specializing in anomaly 
    detection. With a PhD in Statistics, you've developed methods for 
    identifying anomalies that traditional rules miss.""",
    tools=[AnomalyDetectorTool()],
)
```

**Job:** Find statistical outliers using Z-score and IQR methods.

### 4. Report Writer Agent

```python
Agent(
    role="Data Quality Report Writer",
    goal="Synthesize findings into clear, actionable reports",
    backstory="""You are a Technical Writer specializing in data governance.
    You translate complex findings into business-friendly reports that 
    executives can understand and act upon.""",
)
```

**Job:** Create comprehensive report with scores and recommendations.

### 5. Senior Editor Agent (Optional)

```python
Agent(
    role="Senior Report Editor",
    goal="Polish reports to executive presentation standards",
    backstory="""You are a Senior Editor with 20 years in corporate 
    communications. Your edited reports have been presented to boards 
    of directors. You transform good reports into exceptional ones.""",
    llm="gpt-4o",  # More powerful model for polish
)
```

**Job:** Refine language, fix formatting, ensure C-suite ready.

---

## The Tools

### Data Loader Tool

```python
class DataLoaderTool(BaseTool):
    name = "data_loader"
    description = "Load a CSV data file for analysis"
    
    def _run(self, filepath: str) -> str:
        df = pd.read_csv(filepath)
        return f"Loaded {len(df)} rows, {len(df.columns)} columns: {list(df.columns)}"
```

### CDE Loader Tool

```python
class CDELoaderTool(BaseTool):
    name = "cde_loader"
    description = "Load CDE configuration defining critical fields"
    
    def _run(self, config_path: str) -> str:
        with open(config_path) as f:
            config = json.load(f)
        return json.dumps(config, indent=2)
```

### Profiler Tool

```python
class ProfilerTool(BaseTool):
    name = "data_profiler"
    description = "Profile a column with statistics"
    
    def _run(self, column_name: str) -> str:
        col = self._df[column_name]
        stats = {
            "total_count": len(col),
            "null_count": col.isnull().sum(),
            "null_percent": round(col.isnull().sum() / len(col) * 100, 2),
            "unique_count": col.nunique(),
            "unique_percent": round(col.nunique() / len(col) * 100, 2),
        }
        if col.dtype in ['int64', 'float64']:
            stats.update({
                "min": col.min(),
                "max": col.max(),
                "mean": round(col.mean(), 2),
                "std": round(col.std(), 2)
            })
        return json.dumps(stats, indent=2)
```

### Validator Tool

```python
class ValidatorTool(BaseTool):
    name = "data_validator"
    description = "Validate data against business rules"
    
    def _run(self, validation_type: str, column: str) -> str:
        issues = []
        
        if validation_type == "email":
            invalid = self._df[~self._df[column].str.match(r'^[\w.-]+@[\w.-]+\.\w+$')]
            issues = [{"row": i, "value": v} for i, v in invalid[column].items()]
        
        elif validation_type == "credit_score":
            invalid = self._df[(self._df[column] < 300) | (self._df[column] > 850)]
            issues = [{"row": i, "value": v} for i, v in invalid[column].items()]
        
        return json.dumps({
            "validation": validation_type,
            "column": column,
            "issues_found": len(issues),
            "sample_issues": issues[:5]
        })
```

### Anomaly Detector Tool

```python
class AnomalyDetectorTool(BaseTool):
    name = "anomaly_detector"
    description = "Detect statistical outliers in numeric columns"
    
    def _run(self, column: str, method: str = "zscore") -> str:
        col = self._df[column].dropna()
        
        if method == "zscore":
            z_scores = np.abs((col - col.mean()) / col.std())
            outliers = col[z_scores > 3]
        else:  # IQR
            Q1, Q3 = col.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = col[(col < Q1 - 1.5*IQR) | (col > Q3 + 1.5*IQR)]
        
        return json.dumps({
            "column": column,
            "method": method,
            "outlier_count": len(outliers),
            "outlier_percent": round(len(outliers) / len(col) * 100, 2),
            "sample_outliers": outliers.head(5).tolist()
        })
```

---

## CDE Configuration

```json
{
  "critical_data_elements": [
    {
      "field_name": "customer_id",
      "business_definition": "Unique identifier for each customer",
      "data_owner": "Customer Data Management",
      "nullable": false,
      "unique": true,
      "format": "CUS-\\d{6}"
    },
    {
      "field_name": "account_balance",
      "business_definition": "Current balance in customer account",
      "data_owner": "Finance",
      "nullable": false,
      "min_value": 0,
      "regulatory_requirement": "SOX Section 404"
    },
    {
      "field_name": "credit_score",
      "business_definition": "Customer credit rating",
      "data_owner": "Risk Management",
      "nullable": false,
      "min_value": 300,
      "max_value": 850
    }
  ]
}
```

---

## The Polish Feature

The `--polish` flag adds a Senior Editor agent:

```python
class DataQualityCrew:
    def __init__(self, data_file: str, polish: bool = False):
        self.polish = polish
        
        # Core agents always created
        self.profiler = create_profiler_agent()
        self.validator = create_validator_agent()
        self.anomaly = create_anomaly_detector_agent()
        self.writer = create_report_writer_agent()
        
        # Optional editor for executive polish
        if self.polish:
            self.editor = create_senior_editor_agent()
```

**Without polish:** Good technical report
**With polish:** Executive-ready, boardroom-quality document

---

## Sample Output

```markdown
# Data Quality Assessment Report

**Dataset:** customers.csv
**Assessment Date:** 2024-02-19
**Overall Score:** 78/100

---

## Executive Summary

The customer dataset shows **good overall quality** with specific 
issues requiring attention in email validation and credit score ranges.

### Key Metrics
| Dimension | Score | Status |
|-----------|-------|--------|
| Completeness | 94% | GOOD |
| Validity | 72% | NEEDS ATTENTION |
| Uniqueness | 100% | EXCELLENT |
| CDE Compliance | 85% | GOOD |

### Critical Findings
1. **HIGH** - 127 invalid email addresses (8.5%)
2. **HIGH** - 23 credit scores outside valid range
3. **MEDIUM** - Missing phone numbers for 45 records

---

## CDE Analysis

### customer_id
- Status: COMPLIANT
- Completeness: 100%
- Uniqueness: 100%

### account_balance  
- Status: WARNING
- Issue: 3 negative balances detected
- Impact: SOX 404 compliance risk

### credit_score
- Status: NON-COMPLIANT
- Issue: 23 values outside 300-850 range
- Action Required: Investigate data source

---

## Recommendations

### Immediate (This Week)
1. Correct invalid credit scores
2. Validate email addresses at entry

### Short-term (This Month)  
1. Implement real-time validation
2. Add monitoring for CDE fields

### Long-term (This Quarter)
1. Establish data quality dashboard
2. Automate monthly assessments
```

---

## Running the Project

```bash
# Clone
git clone https://github.com/Dewale-A/AgenticDataQuality.git
cd AgenticDataQuality

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add OPENAI_API_KEY

# Run basic assessment
python main.py --data sample_data/customers.csv

# Run with CDE config
python main.py --data sample_data/customers.csv --cde sample_data/cde_config.json

# Run with executive polish
python main.py --data sample_data/customers.csv --cde sample_data/cde_config.json --polish
```

---

## Key Learnings

1. **CDE focus differentiates** — Not just profiling, but business-critical field emphasis
2. **Multiple quality dimensions** — Completeness, validity, uniqueness, anomalies
3. **Optional agents** — `--polish` flag shows how to add agents conditionally
4. **Different models per agent** — Editor uses GPT-4o for better quality
5. **Severity categorization** — CRITICAL/HIGH/MEDIUM/LOW helps prioritization

---

## Part 3 Complete!

You've seen three production multi-agent projects:
- **Loan Origination**: 6-agent sequential workflow with tools
- **Policy Documents**: 3-agent document analysis pipeline
- **Data Quality**: 4-5 agent assessment with optional polish

**Next:** Part 4 — Production Patterns for deploying multi-agent systems.
