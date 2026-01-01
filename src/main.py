from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import build_customer_features
from src.train_model import train_models
from src.evaluate_model import evaluate_classifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/Online Retail.xlsx"

# Data pipeline
df_clean = load_and_clean_data(DATA_PATH)
customer_df = build_customer_features(df_clean)

# Train models
results = train_models(customer_df)

# Prepare test data again for evaluation
X = customer_df[["num_orders", "total_quantity", "total_spent"]]
y = (customer_df["recency_days"] > 90).astype(int)

scaler = results["scaler"]
X_scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Evaluate Logistic Regression
logreg_metrics = evaluate_classifier(
    results["logistic_regression"], X_test, y_test
)

print("Logistic Regression ROC-AUC:", logreg_metrics["roc_auc"])
print(logreg_metrics["classification_report"])
