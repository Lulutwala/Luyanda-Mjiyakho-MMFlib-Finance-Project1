import os
import warnings
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.compose import ColumnTransformer

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
DATA_FILE = r"D:\MASTERS\Luyanda Mjiyakho Project1\Luyanda-Mjiyakho-MMFlib-Finance-Project1\data\merged_stock_news_dataset.csv"

DATE_COL = "date"
TICKER_COL = "ticker"

CLASS_TARGET = "target_up_down"
REG_TARGET = "target_next_return"

# Optional columns to exclude from baseline numeric modeling
EXCLUDE_COLS = {
    DATE_COL,
    TICKER_COL,
    "source_list",
    "all_headlines",
    "all_summaries",
    "all_news_text",
    "latest_news_time",
    "earliest_news_time",
    "target_next_close",
    "target_next_return",
    "target_up_down",
}

# ============================================================
# HELPERS
# ============================================================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, CLASS_TARGET, REG_TARGET])
    df = df.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    return df


def get_numeric_features(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in EXCLUDE_COLS]
    return feature_cols


def time_split(df: pd.DataFrame, train_ratio: float = 0.8):
    unique_dates = sorted(df[DATE_COL].dropna().unique())
    split_idx = int(len(unique_dates) * train_ratio)
    split_date = unique_dates[split_idx]

    train_df = df[df[DATE_COL] < split_date].copy()
    test_df = df[df[DATE_COL] >= split_date].copy()

    return train_df, test_df, split_date


def evaluate_classifier(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    results = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    if y_prob is not None:
        results["roc_auc"] = roc_auc_score(y_test, y_prob)
    else:
        results["roc_auc"] = np.nan

    return results


def evaluate_regressor(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": name,
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": rmse(y_test, y_pred),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading merged dataset...")
    df = load_data(DATA_FILE)
    print(f"Rows: {len(df):,}")
    print(f"Date range: {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")
    print(f"Unique tickers: {df[TICKER_COL].nunique()}")

    feature_cols = get_numeric_features(df)
    print(f"Using {len(feature_cols)} numeric features")

    train_df, test_df, split_date = time_split(df, train_ratio=0.8)
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Split date: {pd.to_datetime(split_date).date()}")

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train_cls = train_df[CLASS_TARGET].astype(int)
    y_test_cls = test_df[CLASS_TARGET].astype(int)

    y_train_reg = train_df[REG_TARGET].astype(float)
    y_test_reg = test_df[REG_TARGET].astype(float)

    # ========================================================
    # CLASSIFICATION MODELS
    # ========================================================
    print("\n=== Classification Models ===")

    logistic_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    xgb_clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ))
    ])

    lgbm_clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ))
    ])

    cls_results = []
    cls_results.append(evaluate_classifier(
        "Logistic Regression", logistic_model,
        X_train, y_train_cls, X_test, y_test_cls
    ))
    cls_results.append(evaluate_classifier(
        "XGBoost Classifier", xgb_clf,
        X_train, y_train_cls, X_test, y_test_cls
    ))
    cls_results.append(evaluate_classifier(
        "LightGBM Classifier", lgbm_clf,
        X_train, y_train_cls, X_test, y_test_cls
    ))

    cls_results_df = pd.DataFrame(cls_results).sort_values("roc_auc", ascending=False)
    print("\nClassification Results:")
    print(cls_results_df.to_string(index=False))

    # ========================================================
    # REGRESSION MODELS
    # ========================================================
    print("\n=== Regression Models ===")

    xgb_reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("reg", XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ))
    ])

    lgbm_reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("reg", LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ))
    ])

    reg_results = []
    reg_results.append(evaluate_regressor(
        "XGBoost Regressor", xgb_reg,
        X_train, y_train_reg, X_test, y_test_reg
    ))
    reg_results.append(evaluate_regressor(
        "LightGBM Regressor", lgbm_reg,
        X_train, y_train_reg, X_test, y_test_reg
    ))

    reg_results_df = pd.DataFrame(reg_results).sort_values("rmse")
    print("\nRegression Results:")
    print(reg_results_df.to_string(index=False))

    # ========================================================
    # SAVE RESULTS
    # ========================================================
    results_dir = os.path.dirname(DATA_FILE)

    cls_results_path = os.path.join(results_dir, "baseline_classification_results.csv")
    reg_results_path = os.path.join(results_dir, "baseline_regression_results.csv")

    cls_results_df.to_csv(cls_results_path, index=False)
    reg_results_df.to_csv(reg_results_path, index=False)

    print("\nSaved results:")
    print(cls_results_path)
    print(reg_results_path)


if __name__ == "__main__":
    main()