# Acute Stroke Mortality Risk Prediction: Machine Learning Benchmark & Clinical Data Science

[![R 4.5+](https://img.shields.io/badge/R-4.5%2B-blue.svg)](https://www.r-project.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Interpretability-red.svg)](https://shap.readthedocs.io/)

---

## 📌 Executive Summary

This repository presents an end-to-end clinical machine learning project predicting in-hospital mortality risk following acute stroke presentation. Using a synthetic clinical dataset incorporating patient demographics, physiological baseline metrics, presenting symptoms, neuroimaging findings, and acute treatment regimens, this project benchmarks **Parametric Regularized Logistic Regression (LASSO)** against **Non-Linear Tree Ensemble Boosting (XGBoost)**.

The project is fully implemented in **R (R Markdown)**

---

## 🔬 Clinical Motivation & Analytical Workflow

Acute stroke care requires rapid, data-driven mortality risk stratification to inform clinical decision-making. 

### Key Methodology & Pipeline Steps:
1. **Exploratory Data Analysis & Statistical Testing**:
   - Continuous physiological features analyzed via non-parametric **Wilcoxon Rank-Sum (Mann-Whitney U) Tests**.
   - Bivariate categorical associations evaluated via Monte-Carlo simulated **Fisher's Exact Tests**.
   - Feature collinearity assessed via pairwise Pearson correlation heatmaps of binary clinical indicators.
2. **Addressing Imbalance & Data Pre-processing**:
   - 80/20 Stratified Train/Validation split preserving outcome ratio.
   - Empirical inverse class reweighting ($w_{\text{pos}} = N_{\text{survived}} / N_{\text{died}}$) applied across models.
3. **Regularized Linear Modeling (LASSO - $L_1$)**:
   - 10-Fold Cross-Validation tuning penalized deviance (log loss), benchmarking optimal $\lambda_{\min}$ against the parsimonious 1-standard-error rule ($\lambda_{1\text{se}}$).
   - 4-way feature importance evaluation: comparing individual dummy indicator log-odds ($\beta$) and grouped clinical construct importance via $L_2$-norms ($\sqrt{\sum \beta^2}$) across both $\lambda_{\min}$ and $\lambda_{1\text{se}}$.
4. **Non-Linear Tree Boosting (XGBoost)**:
   - 5-Fold Cross-Validation hyperparameter grid search optimizing **Precision-Recall AUC (PR-AUC)**.
   - Model interpretability extracted via **SHAP (SHapley Additive exPlanations)** values.
5. **Threshold Calibration & Decision Metrics**:
   - Threshold calibration to achieve clinical sensitivity targets ($\ge 80\%$) for high-risk patient capture.
   - Performance evaluated via Sensitivity, Specificity, Positive Predictive Value (PPV), Negative Predictive Value (NPV), ROC-AUC, PR-AUC, and Cross-Entropy Log Loss.

---

## 📊 Key Results & Model Comparison

Both LASSO and XGBoost demonstrated strong predictive capacity on the independent validation dataset, with XGBoost achieving superior log loss calibration and feature interaction modeling.

| Performance Metric | LASSO Regression (Train) | LASSO Regression (Test) | XGBoost (Train) | XGBoost (Test) |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.85 | **0.84** | 0.89 | **0.86** |
| **PR-AUC** | 0.78 | **0.76** | 0.83 | **0.80** |
| **Sensitivity (Recall)** | 0.82 | **0.80** | 0.85 | **0.82** |
| **Specificity** | 0.74 | **0.72** | 0.78 | **0.75** |
| **Balanced Accuracy** | 0.78 | **0.76** | 0.82 | **0.785** |
| **Log Loss** | 0.4215 | **0.4382** | 0.3650 | **0.3920** |

*Note: Models evaluated at calibrated decision thresholds ($t = 0.395$ for LASSO; $t = 0.380$ for XGBoost).*

---

## 🗂 Repository Structure

```gfm
.
├── stroke_code.Rmd     # Fully annotated R Markdown analysis notebook
├── README.md    # Project documentation
└── stroke_code.html      # Final report with code
```

---

## 🚀 How to Run in R Environment

1. Open `stroke_code.Rmd` in RStudio.
2. Install required packages in R:
   ```r
   install.packages(c("tidyverse", "gtsummary", "gt", "caret", "glmnet", 
                      "ggcorrplot", "pROC", "PRROC", "xgboost", "fastshap", 
                      "SHAPforxgboost", "patchwork", "scales"))
   ```
3. Knit to HTML to view full interactive tables and figures.

---

## 💡 Tech Stack & Skills Highlighted

- **Data Science Languages**: R (tidyverse, dplyr, ggplot2)
- **Machine Learning**: `xgboost`, `glmnet`, `caret`, Hyperparameter Grid Search, Cross-Validation
- **Statistical Inference**: Wilcoxon Rank-Sum Tests, Fisher's Exact Tests, Chi-Square Tests
- **Model Explainability**: SHAP (SHapley Additive exPlanations), Grouped Feature L2 Norms
- **Clinical Evaluation**: Sensitivity/Specificity Calibration, ROC-AUC, PR-AUC, Log Loss Benchmarking
