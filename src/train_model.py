import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def train_models(customer_df: pd.DataFrame) -> dict:
    """
    Train churn prediction models using customer-level features
    and track experiments with MLflow.
    """

    # Create churn label
    customer_df = customer_df.copy()
    customer_df["churn"] = (customer_df["recency_days"] > 90).astype(int)

    # Leakage-free features
    X = customer_df[["num_orders", "total_quantity", "total_spent"]]
    y = customer_df["churn"]

    # Scale features (for Logistic Regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    results = {"scaler": scaler}

    # =========================
    # Logistic Regression (MLflow)
    # =========================
    with mlflow.start_run(run_name="Logistic_Regression"):

        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("churn_threshold_days", 90)
        mlflow.log_param("features", "num_orders,total_quantity,total_spent")

        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X_train, y_train)

        logreg_auc = roc_auc_score(
            y_test, logreg.predict_proba(X_test)[:, 1]
        )

        mlflow.log_metric("roc_auc", logreg_auc)
        mlflow.sklearn.log_model(logreg, "model")

        results["logistic_regression"] = logreg
        results["logistic_regression_auc"] = logreg_auc

    # =========================
    # Random Forest (MLflow)
    # =========================
    with mlflow.start_run(run_name="Random_Forest"):

        mlflow.log_param("model_type", "random_forest")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("churn_threshold_days", 90)

        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )

        rf.fit(X_train, y_train)

        rf_auc = roc_auc_score(
            y_test, rf.predict_proba(X_test)[:, 1]
        )

        mlflow.log_metric("roc_auc", rf_auc)
        mlflow.sklearn.log_model(rf, "model")

        results["random_forest"] = rf
        results["random_forest_auc"] = rf_auc

    return results
