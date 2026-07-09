"""
Definición de modelos, generación de splits Walk-Forward y entrenamiento/evaluación.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb


def get_models():
    """Modelos a comparar. LSTM/Transformer se dejan fuera del MVP por peso de dependencias."""
    return {
        "Logistic Regression (baseline)": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1
        ),
    }


def generate_walk_forward_splits(df, start_year, first_val_year):
    """
    Genera los splits tipo:
    2015-2018 -> train / 2019 -> val
    2015-2019 -> train / 2020 -> val
    ...
    El último año disponible se marca como 'test_final'.
    """
    years = sorted(df.index.year.unique())
    final_year = years[-1]

    splits = []
    val_year = first_val_year
    while val_year < final_year:
        train = df[(df.index.year >= start_year) & (df.index.year < val_year)]
        val = df[df.index.year == val_year]
        if len(train) > 100 and len(val) > 0:
            splits.append({"train": train, "val": val, "val_year": val_year, "type": "validation"})
        val_year += 1

    train_final = df[(df.index.year >= start_year) & (df.index.year < final_year)]
    test_final = df[df.index.year == final_year]
    if len(train_final) > 100 and len(test_final) > 0:
        splits.append({"train": train_final, "val": test_final, "val_year": final_year, "type": "test_final"})

    return splits


def train_and_evaluate(model, train_df, val_df, feature_cols, target_col="target_direction"):
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model.fit(X_train_s, y_train)
    preds = model.predict(X_val_s)
    probs = model.predict_proba(X_val_s)[:, 1] if hasattr(model, "predict_proba") else preds.astype(float)

    metrics = {
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds, zero_division=0),
        "recall": recall_score(y_val, preds, zero_division=0),
        "f1": f1_score(y_val, preds, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(y_val, probs)
    except Exception:
        metrics["auc"] = np.nan

    result_df = val_df.copy()
    result_df["pred"] = preds
    result_df["prob_up"] = probs
    return metrics, result_df
