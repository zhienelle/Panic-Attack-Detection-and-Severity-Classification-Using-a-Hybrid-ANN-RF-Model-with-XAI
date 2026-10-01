# Panic Attack Detection and Severity Classification Using Hybrid ANN-RF with Explainable AI

> A machine learning research project that evaluates Artificial Neural Network, Random Forest, and Hybrid ANN-RF models for detecting panic attack occurrence and classifying severity. The project combines predictive modeling with Explainable AI techniques to improve the interpretability of model outputs for non-technical users.

## Project Overview

### What It Is
This undergraduate thesis investigated the use of machine learning for two related prediction tasks:
1. **Panic attack occurrence detection** — determining whether an observation indicates panic disorder occurrence.
2. **Severity classification** — classifying the predicted condition according to severity.

The study compared three modeling approaches:
- Random Forest (RF)
- Artificial Neural Network (ANN)
- Hybrid Artificial Neural Network–Random Forest (ANN-RF)

Rather than evaluating models using accuracy alone, the project examined multiple performance dimensions including **accuracy, precision, recall, F1-score, ROC-AUC, and execution time**.

Explainable Artificial Intelligence techniques were also incorporated to make model predictions more interpretable and identify which input features contributed most strongly to prediction outcomes.

This was developed as a collaborative thesis project. My primary technical responsibility was the **Random Forest component**, together with model evaluation and contributions to Explainable AI implementation and interpretation.

### Key Features
- **Binary occurrence prediction** for identifying potential panic disorder cases.
- **Severity classification** for distinguishing levels of predicted condition severity.
- **Three-model comparative evaluation** across Random Forest, ANN, and Hybrid ANN-RF architectures.
- **Data preprocessing and feature engineering** for mixed categorical and numerical health-related variables.
- **Hyperparameter tuning and cross-validation** to evaluate model robustness and generalization.
- **Class imbalance handling** using techniques including class weighting and SMOTE.
- **Explainable AI analysis** using SHAP and LIME concepts to interpret global and individual predictions.
- **Multi-metric model evaluation** using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and execution time.

---

## My Role & Core Contributions
**Role:** Machine Learning Developer

### Personal Scope & Ownership
My primary responsibility was the **Random Forest modeling and evaluation component** of the research.

My contributions included:
- **Designed and implemented the Random Forest models** used for both panic attack occurrence detection and severity classification.
- **Prepared and transformed model inputs** using data preprocessing techniques including categorical encoding, numerical scaling, feature selection, and class-imbalance handling.
- **Configured and evaluated Random Forest classifiers** using parameters such as number of estimators, tree depth, feature-selection strategy, minimum leaf size, and balanced class weighting.
- **Applied hyperparameter tuning and cross-validation** to evaluate model configurations beyond a single train-test split.
- **Evaluated model performance across multiple metrics**, including accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and processing time.
- **Participated in the comparative evaluation of RF, ANN, and Hybrid ANN-RF models**, examining the trade-offs among predictive accuracy, sensitivity, precision, and computational performance.
- **Contributed to validating the Hybrid ANN-RF model's performance profile**, including its strong precision and testing-time results relative to the standalone models.
- **Applied Explainable AI techniques** to improve transparency and translate model behavior into insights understandable to users without a machine-learning background.

### Engineering Highlights & Problem Solving
#### 1. Evaluated Models Beyond Accuracy
Instead of treating the model with the highest accuracy as automatically superior, the research compared multiple performance dimensions.
The evaluation considered Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Testing Time.

This revealed meaningful trade-offs between the competing architectures.

For example:
| Model Result | Performance Highlight |
|---|---:|
| ANN | Highest observed accuracy: **92.02%** |
| ANN | Highest observed recall: **98.72%** |
| Hybrid ANN-RF | Highest observed precision: **85.62%** |
| Hybrid ANN-RF | Fastest observed testing time: **0.06 s** |

The comparison demonstrated that model selection depends on the operational objective rather than on a single performance metric.

#### 2. Built and Validated the Random Forest Pipeline
The Random Forest implementation included separate modeling workflows for occurrence detection and severity prediction.
The workflow incorporated:

```text
Raw Dataset
     |
     v
Data Inspection and Cleaning
     |
     v
Feature Selection / Engineering
     |
     v
Categorical Encoding
     |
     v
Scaling / Class-Balance Processing
     |
     v
Random Forest Training
     |
     v
Hyperparameter Tuning
     |
     v
Cross-Validation
     |
     v
Hold-Out Testing
     |
     v
Performance Evaluation
```

For occurrence prediction, the implementation used key predictor groups including:
- Medical History
- Lifestyle Factors
- Symptoms
- Current Stressors
- Coping Mechanisms
- Personal History

#### 3. Addressed Class Imbalance and Generalization
The modeling workflow incorporated techniques designed to reduce misleading performance caused by imbalanced target classes.

These included:
- balanced class weighting;
- stratified cross-validation;
- SMOTE-based resampling for severity modeling;
- multi-run validation;
- train-versus-validation-versus-test comparison; and
- F1-oriented model selection.

This provided a more complete picture of how the model generalized beyond its training data.

#### 4. Improved Model Transparency with Explainable AI

Predictive performance alone was not considered sufficient for a health-related machine-learning application.

Explainability techniques were explored to identify both:
- **global feature influence** — which features generally contribute most strongly to predictions; and
- **local explanations** — why the model produced a particular result for an individual observation.

The project used **SHAP** and **LIME** concepts to translate complex prediction behavior into more understandable evidence.

Important contributing feature groups investigated during the project included:
- Symptoms
- Current Stressors
- Psychiatric History
- Medical History
- Lifestyle Factors
- Coping Mechanisms

This work strengthened the connection between machine-learning performance and practical interpretability.

---

## Architecture & Tech Stack

### Language
- Python

### Machine Learning
- Scikit-learn
- TensorFlow
- Keras
- Random Forest
- Artificial Neural Networks
- Hybrid ANN-RF modeling

### Data Processing
- Pandas
- NumPy
- Scikit-learn preprocessing utilities
- One-Hot Encoding
- Label Encoding
- Min-Max Scaling
- SMOTE
- Class weighting

### Model Evaluation
- Stratified K-Fold Cross-Validation
- RandomizedSearchCV
- Grid-search / parameter experimentation
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- ROC Curve
- Precision-Recall analysis
- Log Loss
- Training and testing time measurement

### Explainable AI
- SHAP
- LIME
- Feature Importance Analysis

### Visualization
- Matplotlib
- Seaborn

### Development Environment
- Jupyter Notebook / Google Colab-compatible notebooks
- Git
- GitHub

---

## Machine Learning Pipeline
<img width="589" height="341" alt="Screenshot 2026-09-28 142903" src="https://github.com/user-attachments/assets/a0c9929a-1603-46da-9dc6-2125ede713ef" />

---

## Repository Structure

### Main Notebooks

**`rf.ipynb`**

Random Forest experimentation covering:

- exploratory data analysis;
- preprocessing;
- feature transformation;
- occurrence prediction;
- severity classification;
- hyperparameter tuning;
- cross-validation;
- performance evaluation; and
- feature-importance / explainability analysis.

**`ann.ipynb`**

Artificial Neural Network modeling and performance evaluation.

**`hybrid.ipynb`**

Hybrid ANN-RF experimentation combining the modeling approaches for comparative performance analysis.

---

## Impact & Key Takeaways

### Quantitative Results

The comparative experiments demonstrated that no single architecture dominated every evaluation metric.

The ANN achieved:

- **92.02% accuracy**
- **98.72% recall**

The Hybrid ANN-RF achieved:

- **85.62% precision**
- **0.06-second testing time**

These results demonstrated an important machine-learning engineering trade-off:

> The highest-accuracy model is not necessarily the strongest model for every operational objective.

For problems where sensitivity is prioritized, recall may be more important. Where false positives must be controlled, precision may be more important. Where inference speed matters, testing time introduces another dimension to model selection.

### Key Technical Takeaways

This project strengthened my experience in:
- designing end-to-end machine-learning experiments;
- working with mixed categorical and numerical datasets;
- conducting exploratory data analysis;
- developing preprocessing and feature-engineering pipelines;
- implementing ensemble-learning models;
- configuring Random Forest classifiers;
- handling class imbalance;
- applying SMOTE and class weighting;
- performing hyperparameter tuning;
- designing cross-validation experiments;
- evaluating models using multiple performance metrics;
- comparing competing machine-learning architectures;
- identifying overfitting and generalization behavior;
- analyzing model-performance trade-offs;
- applying Explainable AI techniques;
- translating technical model outputs into understandable insights; and
- communicating machine-learning findings to technical and non-technical audiences.

---

## Responsible Use

This repository represents an **academic machine-learning research project**.

The models and outputs should not be interpreted as medical diagnoses, clinical recommendations, or substitutes for evaluation by qualified healthcare professionals.

A production healthcare application would require substantially more work, including:

- independently validated clinical datasets;
- external model validation;
- bias and fairness assessment;
- privacy and security controls;
- regulatory review;
- model monitoring; and
- involvement of qualified healthcare professionals.

The project is intended to demonstrate machine-learning experimentation, comparative modeling, and explainability techniques.

---

## Project Context
**Project Type:** Undergraduate Thesis / Team Machine Learning Research Project

**Domain:** Machine Learning / Data Science / Explainable AI

**Primary Personal Role:** Machine Learning Developer

---

## Contribution Attribution

This thesis was completed as a collaborative academic project.

My primary technical contribution centered on the **Random Forest modeling pipeline**, including model development, experimentation, evaluation, and participation in the comparative assessment of the ANN, Random Forest, and Hybrid ANN-RF approaches.

The repository also contains components developed collaboratively by the thesis team. Project-wide capabilities should therefore not be interpreted as sole individual contributions unless specifically identified in the **My Role & Core Contributions** section above.
