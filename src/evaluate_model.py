from sklearn.metrics import classification_report, roc_auc_score


def evaluate_classifier(model, X_test, y_test) -> dict:
    """
    Evaluate a trained classification model.

    Parameters
    ----------
    model
        Trained classifier with predict_proba
    X_test
        Test features
    y_test
        True labels

    Returns
    -------
    dict
        Evaluation metrics
    """

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "classification_report": classification_report(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
