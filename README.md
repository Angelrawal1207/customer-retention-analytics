# Customer Churn Risk Model

> An end-to-end machine learning pipeline for predicting customer churn, quantifying churn risk, and translating model outputs into actionable retention insights.

## Overview

Customer churn is a major business challenge for companies that depend on repeat customers. Identifying customers who are likely to leave can help businesses intervene early through targeted retention strategies.

This project builds a complete **customer churn prediction and risk-scoring pipeline** using a deliberately messy retail customer dataset.

The workflow covers the journey from **raw business data → data quality → machine learning → evaluation → customer-level risk prioritization**.

---

## Business Objective

### Problem

How can a retail business identify customers who are most likely to churn before they leave?

### Solution

A **Logistic Regression classification model** is trained using customer attributes to estimate the probability of churn.

Rather than producing only a binary prediction, the pipeline generates a **churn probability and risk label**, allowing customers to be ranked according to their predicted risk.

```text
Raw Customer Data
       ↓
Data Inspection
       ↓
Data Cleaning
       ↓
Outlier Treatment
       ↓
Feature Selection
       ↓
Train / Test Split
       ↓
Feature Standardisation
       ↓
Logistic Regression
       ↓
Churn Probability
       ↓
Risk Classification
       ↓
Retention Prioritisation
```

---

## Dataset

The project uses a deliberately messy dataset containing **100 customer records**.

### Features

| Feature         | Description                              |
| --------------- | ---------------------------------------- |
| `Customer_ID`   | Unique customer identifier               |
| `Age`           | Customer age                             |
| `Monthly_Spend` | Customer monthly spending                |
| `Complaints`    | Number of customer complaints            |
| `Churn`         | Target variable: 1 = churn, 0 = no churn |

The dataset contains intentionally introduced data-quality problems such as:

* Duplicate records
* Missing values
* Text stored in numeric fields
* Invalid age values
* Negative spending
* Extreme spending values
* Extreme complaint counts

This creates a realistic preprocessing scenario rather than assuming perfectly clean data.

---

## Machine Learning Approach

### 1. Data Cleaning

The raw dataset is inspected and cleaned before modelling.

The pipeline:

* Removes duplicate records
* Strips unnecessary whitespace
* Converts text-based numerical values
* Handles invalid age values
* Converts invalid spending values to missing
* Imputes missing numerical values using the median

The dataset contains **100 original records and 95 records after duplicate removal**.

### 2. Outlier Treatment

Extreme values are handled using the **Interquartile Range (IQR)** method.

Instead of deleting affected customer records, detected outliers are capped using the calculated IQR boundaries.

This preserves the customer record while reducing the influence of extreme observations on the model.

### 3. Feature Selection

The model uses three business-relevant predictors:

```text
Age
Monthly_Spend
Complaints
```

`Customer_ID` is excluded because it is an identifier rather than a meaningful predictive variable.

### 4. Train-Test Split

The cleaned dataset is divided into:

* **80% training data — 76 customers**
* **20% testing data — 19 customers**

Stratification is used to maintain a similar churn distribution across the training and testing sets.

### 5. Feature Standardisation

`StandardScaler` is applied to the numerical features.

The scaler is fitted only on the training data and then applied to the test data, preventing test-set information from leaking into the training process.

---

## Model

### Logistic Regression

Logistic Regression is used because the target is binary:

```text
0 → No Churn
1 → Churn
```

The model provides both:

* A binary churn prediction
* A probability of churn

The probability output is particularly useful for **customer risk ranking and retention prioritisation**.

---

## Model Performance

The evaluated model produced the following results on the 19-customer test set:

| Metric    |      Result |
| --------- | ----------: |
| Accuracy  |  **89.47%** |
| Precision |  **83.33%** |
| Recall    | **100.00%** |
| F1-Score  |  **90.91%** |

Confusion matrix:

```text
                Predicted
              No Churn  Churn

Actual
No Churn          7       2
Churn             0      10
```

The model correctly identified all **10 churners in the test set**, resulting in **100% recall for churn detection**.

> **Important:** The dataset is intentionally small, so these metrics should not be interpreted as production-level performance.

---

## Model Interpretation

The Logistic Regression coefficients provide insight into how the selected variables relate to predicted churn risk.

### Monthly Spend

The model learned a **strong negative coefficient** for monthly spending.

Within this dataset, higher monthly spending is associated with lower predicted churn risk.

### Complaints

Complaints have a **strong positive coefficient**.

Customers with more complaints receive substantially higher predicted churn risk.

This represents the most actionable business signal in the model.

### Age

Age has a smaller positive coefficient compared with the other variables, indicating a relatively weaker relationship with predicted churn risk in this dataset.

---

## Risk Scoring

The pipeline goes beyond a simple `Churn / No Churn` prediction.

Each test customer receives:

* Customer ID
* Customer attributes
* Actual churn status
* Predicted churn status
* Churn probability
* Human-readable risk label

Example risk output:

```text
Customer_ID    Churn Probability    Risk
------------------------------------------------
C037                 99.7%          Likely to Churn
C018                 99.7%          Likely to Churn
C011                 99.7%          Likely to Churn
C002                 99.0%          Likely to Churn
C095                 98.1%          Likely to Churn
```

The results are sorted from highest to lowest predicted churn probability so that retention teams can focus on the highest-risk customers first.

---

## Business Application

The model can support a proactive retention workflow:

```text
Customer Data
      ↓
Churn Risk Model
      ↓
Risk Probability
      ↓
Customer Prioritisation
      ↓
Retention Intervention
```

Potential interventions could include:

* Proactive customer support
* Complaint-resolution follow-ups
* Personalised retention offers
* Loyalty incentives
* Targeted engagement campaigns

The key business insight from this dataset is that **customer complaints are strongly associated with churn risk**, making complaint resolution a potential retention lever.

---

## Output

The pipeline generates:

**`smartkart_churn_risk_report.csv`**

The report contains the model's customer-level predictions and risk classifications, providing a business-oriented output rather than only technical evaluation metrics.

---

## Project Structure

```text
customer-churn-risk-model/
│
├── data/
│   └── SmartKart_dirty_100_rows.csv
│
├── notebooks/
│   └── customer_churn_prediction.ipynb
│
├── outputs/
│   └── smartkart_churn_risk_report.csv
│
└── README.md
```

---

## Tech Stack

* **Python**
* **Pandas** — Data manipulation
* **NumPy** — Numerical computing
* **Scikit-learn** — Preprocessing, Logistic Regression and evaluation
* **Matplotlib** — Visualization
* **Seaborn** — Confusion matrix visualization
* **Google Colab / Jupyter Notebook** — Development environment

---

## End-to-End Pipeline

| Stage              | Implementation                            |
| ------------------ | ----------------------------------------- |
| Data Collection    | CSV ingestion                             |
| Data Understanding | Shape, types, missing values & statistics |
| Data Cleaning      | Duplicates, invalid values & missing data |
| Outlier Treatment  | IQR-based capping                         |
| Feature Selection  | Age, Monthly Spend, Complaints            |
| Target             | Binary Churn                              |
| Data Split         | Stratified 80/20 split                    |
| Scaling            | StandardScaler                            |
| Model              | Logistic Regression                       |
| Prediction         | Class + probability                       |
| Evaluation         | Accuracy, Precision, Recall, F1           |
| Interpretation     | Model coefficients                        |
| Business Output    | Ranked churn-risk report                  |

---

## Limitations

This implementation is a **baseline proof of concept**, not a production-ready churn system.

The dataset contains only 100 original records and therefore provides limited statistical evidence for generalising model performance.

A production implementation would require:

* A substantially larger customer dataset
* Historical behavioural data
* Customer tenure
* Purchase frequency
* Recency and monetary value
* Customer-service interactions
* Marketing engagement
* Temporal validation
* Cross-validation
* Model calibration
* Monitoring for data and model drift

---

## Future Improvements

Potential extensions include:

* Compare Logistic Regression with Random Forest, XGBoost and other classifiers
* Perform cross-validation
* Tune model hyperparameters
* Add ROC-AUC and PR-AUC
* Calibrate predicted probabilities
* Build a churn-risk dashboard
* Add SHAP-based explainability
* Introduce automated retention recommendations
* Deploy the model through an API
* Integrate predictions with CRM workflows

---

## Key Takeaway

This project demonstrates that a useful machine learning solution is more than simply training a model.

The complete process is:

**Data Quality → Feature Preparation → Modelling → Evaluation → Interpretation → Business Action**

The final goal is not just to predict **who may churn**, but to provide a structured way for a business to **prioritise customers and take preventive action**.

---

## Author

**Angel Rawal**

AI/ML • Data Analytics • Business Intelligence
