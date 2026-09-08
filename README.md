# Predicting Disease Flares in Rheumatoid Arthritis from Routine Clinical Data using Machine Learning
> Master's Thesis — MSc in Data Science  
> International Hellenic University, Thessaloniki, Greece  
> Author: Konstantina Tzeikou | Supervisor: Dr. Paraskevas Koukaras  
> September 2026

---

## Overview

This repository contains the full analytical pipeline for investigating the prediction of Rheumatoid 
Arthritis (RA) disease flares at the next scheduled clinical visit using routine outpatient data. Four supervised machine
learning classifiers — Logistic Regression, Support Vector Machine, Random Forest, and XGBoost — are developed and
evaluated under a patient-aware nested cross-validation framework designed to eliminate data leakage across all stages of model development.

As a secondary contribution, the ML models are compared against zero-shot and few-shot Large Language Model inference, examining whether general medical knowledge
encoded during LLM pretraining can substitute for patient-specific supervised learning. The study uses a longitudinal cohort of 106 patients from a Greek rheumatology outpatient setting. 
All models are evaluated under a patient-aware Stratified Group K-Fold nested cross-validation framework to prevent data leakage.

**Best model:** Logistic Regression  
**AP:** 0.508 | **ROC-AUC:** 0.695  
**Flare Recall at threshold 0.295:** 83%

## Dataset

The dataset consists of longitudinal clinical records from 106 pseudonymized RA patients attending a rheumatology outpatient clinic in Thessaloniki, Greece.
**Due to patient data privacy constraints, the raw dataset is not publicly available.**   

**Target variable:** `flare_next_visit` (binary)  
**Class distribution:** 130 flare (33.1%) / 263 no-flare (66.9%)  
**Missing data:** DAS28 37%, CRP 17%, ESR 8%  
**Imputation:** MICE-Bayesian

---

## Repository Structure
```bash
RA-flare-prediction/
│
├── data/
│ └── README.md # Dataset description
│
├── notebooks/
│ ├── 01_EDA.ipynb # Exploratory data analysis
│ ├── 02_Feature_Selection.ipynb # Multi-criteria feature selection
│ ├── 03_Modeling.ipynb # Imputation, nested CV, all classifiers
│ └── 04_LLM_Evaluation.ipynb # Zero-shot and few-shot pipeline
│
├── src/
│ └── preprocessing.py # Preprocessing pipeline
│
├── requirements.txt
├── LICENSE
└── README.md
```
---

  
### 01 - Exploratory Data Analysis
```
notebooks/01_EDA.ipynb
```
Descriptive statistics, missingness analysis, class distribution, feature correlations, and cohort
characteristics across clinical, treatment, and demographic variables.

  
### 02 - Feature Selection
```
notebooks/02_Feature_Selection.ipynb
```
Four-step multi-criteria framework:
- **Univariate screening:** Mann-Whitney U (continuous),
  Chi-square (binary/categorical) with effect sizes
- **Correlation analysis:** Pairwise Spearman correlation,
  redundancy removal at threshold r > 0.85
- **RF importance:** Cross-validated Random Forest importance
  computed on training folds only (leakage prevented)
- **Clinical review:** Domain knowledge retention of clinically justified variables regardless of
  univariate significance

Output: final 18-variable feature set :

| Variable Type | Count | Examples |
|---|---|---|
| Patient-level | 3 | Age at visit, Anti-CCP status, Disease duration |
| Disease activity | 2 | DAS28 score, DAS28 change |
| Inflammatory markers | 4 | ESR, CRP, Inflammation flag, ESR lag |
| Treatment | 4 | MTX dose, Steroid dose, csDMARD use, Treatment decision |
| Comorbidities | 3 | Cardiovascular, Musculoskeletal, Metabolic/Endocrine |
| History | 2 | Previous flare, Days since last visit |
| **Final feature set** | **18** |  |

  

### 03 - ML Modeling
```
notebooks/03_Modeling.ipynb
```
**Nested cross-validation framework:**
- Outer loop: 5-fold Stratified Group K-Fold   (performance estimation)
- Inner loop: 3-fold Stratified Group K-Fold   (hyperparameter tuning)
- Grouping variable: patient_id
- Primary metric: Average Precision
- Secondary metric: ROC-AUC

**Hyperparameter search:**
- Logistic Regression: GridSearchCV (40 combinations)
- SVM: RandomizedSearchCV (50 iterations, per kernel)
- Random Forest: RandomizedSearchCV (50 iterations)
- XGBoost: RandomizedSearchCV (50 iterations,
  fold-specific scale_pos_weight)

**Outputs:** ROC curves, PR curves, calibration plots, confusion matrices, learning curve, odds ratio table

  
### 04 - LLM Evaluation
```
notebooks/04_LLM_Evaluation.ipynb
```
Secondary analysis benchmarking LLM inference against the trained ML classifiers.

**Model:** Claude Haiku (`claude-haiku-4-5-20251001`)
**Strategies:** Zero-shot and few-shot
**Serialization:** List template and text template
**Evaluation set:** 360 visits (6 excluded as
few-shot examples, plus all visits from those patients)
Requires Anthropic API key 


---
## Key Results
---
### ML Model Performance
*All models exceed the no-skill baselines (ROC-AUC = 0.500, AP = 0.331). Logistic Regression achieves the best performance on both metrics.*

<img width="1585" height="585" alt="image" src="https://github.com/user-attachments/assets/76aafbc0-a56b-45f2-ba94-70dff84e9134" />


  

| Model | AP | ROC-AUC | Recall (Flare) | Specificity | Brier |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.508 [0.432–0.603]** | **0.695 [0.641–0.752]** | **0.831** | 0.483 | **0.201** |
| SVM | 0.470 [0.395–0.562] | 0.639 [0.581–0.697]| 0.923 | 0.298 | 0.207 |
| XGBoost | 0.450 [0.380–0.540] | 0.666 0.613–0.719] | 0.877 | 0.399 | 0.213 |
| Random Forest | 0.414 [0.354–0.501] | 0.646 [0.591–0.700] | 0.877 | 0.380 | 0.224 |
| No-skill baseline | 0.331 | 0.500 | — | — | 0.221 |

---

### Logistic Regression — Prediction Equation
```
log-odds(flare) = 0.511 × DAS28_score + 0.053 × ESR + 0.032 × disease_duration² + 0.028 × CRP − 0.023 × age_at_visit - 2.015
```
| Feature | β | Odds Ratio |
|---|---|---|
| DAS28 score | +0.511 | 1.667 |
| ESR | +0.053 | 1.055 |
| Disease duration² | +0.032 | 1.032 |
| CRP | +0.028 | 1.029 |
| Age at visit | −0.023 | 0.977 |

---
### LLM Benchmark

| Strategy | AP | ROC-AUC |
|---|---|---|
| Few-shot / List | 0.42 [0.35–0.48] | 0.64 [0.59–0.68] |
| Zero-shot / List | 0.41 [0.35–0.47] | 0.63 [0.58–0.68] |
| Zero-shot / Text | 0.40 [0.34–0.46] | 0.62 [0.57–0.67] |
| Few-shot / Text | 0.40 [0.34–0.46] | 0.61 [0.56–0.66] |

Patient-specific longitudinal training (LR: AP = 0.508)
outperforms LLM zero-shot and few-shot inference
(AP = 0.40–0.42) on this dataset.

<img width="1590" height="590" alt="image" src="https://github.com/user-attachments/assets/996c061e-e908-4a22-9708-ce1f4d4ce060" />

---
## Conclusions

- **Feasibility:** Routine outpatient clinical data contains genuine predictive signal for next-visit flare
prediction. Logistic Regression achieves AP = 0.508 against a no-skill baseline of 0.331 — a 53% improvement over chance.

- **Model Selection:** Regularized Logistic Regression outperforms all three ensemble methods and produces a
clinically interpretable five-variable prediction equation dominated by DAS28 score (OR = 1.667). In a small cohort
where predictive signal is predominantly linear, regularization outperforms complexity.

- **LLM Comparison:** Pre-trained general medical knowledge cannot substitute for patient-specific supervised training.
The AP gap of 0.088–0.108 between LR and the best LLM configuration is supported by non-overlapping confidence
intervals. Few-shot LLM configurations offer a viable training-free baseline in data-scarce settings.

---

## Acknowledgements

Clinical data access was provided by Professor Theodoros
Dimitroulas (Ippokrateio General Hospital, Thessaloniki).
Data collection assistance was provided by Nikolaos
Papadopoulos.

---
