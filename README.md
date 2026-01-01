Customer Churn Prediction with Explainable AI

Project Overview

This project implements an end-to-end machine learning pipeline to predict customer churn based on historical transactional data from an online retail business. The solution emphasizes reproducibility, clean engineering practices, experiment tracking, and model explainability.

The project was developed following a complete AI project methodology, covering data preprocessing, feature engineering, modeling, evaluation, experiment tracking with MLflow, and explainable AI using SHAP.

Problem Statement

Customer churn prediction aims to identify customers who are likely to stop purchasing in the near future. Early identification of churn enables businesses to take proactive retention actions.

Churn Definition
A customer is labeled as churned (1) if their last purchase occurred more than 90 days before the end of the dataset.
Otherwise, the customer is labeled as non-churned (0).

Dataset

Name: Online Retail Dataset

Source: UCI Machine Learning Repository

URL: https://archive.ics.uci.edu/ml/datasets/online+retail

Description: Transactional data from a UK-based online retail company

Time period: December 2010 – December 2011

The dataset contains invoice-level purchase records and is widely used for academic research in customer analytics.

Project Structure
churn_prediction_project/
│
├── data/ # Raw dataset
├── notebooks/ # Exploration & SHAP explainability
│ ├── 01_exploration.ipynb
│ └── 02_shap_explainability.ipynb
│
├── src/ # Production-style pipeline
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── train_model.py
│ ├── evaluate_model.py
│ └── main.py
│
├── models/ # Model artifacts (optional)
├── reports/ # Figures & outputs
├── requirements.txt
└── README.md

🔧 Methodology
1️⃣ Data Preprocessing

Implemented in data_preprocessing.py:

Removed missing customer identifiers

Filtered cancelled invoices

Removed invalid transactions (negative or zero quantity/price)

Ensured proper datetime formatting

2️⃣ Feature Engineering

Implemented in feature_engineering.py:

Aggregated transaction-level data to customer-level features:

num_orders

total_quantity

total_spent

recency_days

3️⃣ Modeling

Implemented in train_model.py:

Logistic Regression (baseline, interpretable)

Random Forest (non-linear, higher expressive power)

Leakage-free features used (recency only for labeling)

Performance evaluated using ROC-AUC

4️⃣ Experiment Tracking (MLflow)

Logged model parameters, metrics, and trained artifacts

Ensured reproducibility and transparent experiment comparison

5️⃣ Explainable AI (SHAP)

Implemented in 02_shap_explainability.ipynb:

Global explanations using SHAP summary plots

Local explanations using SHAP waterfall plots

Identified key drivers of churn at both population and individual levels

Results

Models achieved realistic and generalizable ROC-AUC scores

Random Forest captured non-linear customer behavior patterns

SHAP analysis revealed that spending and purchase frequency were the most influential features

How to Run the Project
1️⃣ Setup environment
python -m venv venv
source venv/bin/activate # or venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

2️⃣ Run the full pipeline
python -m src.main

3️⃣ Launch MLflow UI
mlflow ui

Then open:
http://127.0.0.1:5000

4️⃣ Run notebooks

Open notebooks in VS Code or Jupyter:

01_exploration.ipynb

02_shap_explainability.ipynb

Production-Grade Design Considerations

Although no deployment is included, the project follows production-grade ML engineering principles:

Modular and reusable codebase

Reproducible experiments with MLflow

Clear separation of concerns

Leakage-free modeling

Explainable predictions

The pipeline can be extended for deployment with minimal refactoring.

🚀 Future Improvements

Survival analysis for time-to-churn modeling

Hyperparameter optimization

Integration of external features (marketing, demographics)

Deployment as an API or batch prediction service

🛠 Technologies Used

Python 3.11

Pandas, NumPy

Scikit-learn

MLflow

SHAP

Jupyter Notebook

Git & GitHub

License & Usage

This project is for educational and academic purposes.
The dataset is publicly available via the UCI Machine Learning Repository.

Final Note

This project demonstrates a complete, reproducible, and explainable machine learning workflow for customer churn prediction, combining strong engineering practices with interpretable AI techniques.
