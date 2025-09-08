# ===============================================================
# Scam Risk (4-class) — TF-IDF + LogisticRegression + SMOTE + K-Fold
# Prints per-fold class distributions after SMOTE/UNDER + logloss per fold
# Saves the final fitted pipeline, and can also test on a custom text.
# ===============================================================

import os, re, sys, json, string, random, argparse, importlib
import numpy as np
import pandas as pd

from typing import List
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss, mean_absolute_error, mean_squared_error
from sklearn.base import clone
import joblib

# -------------------- Config defaults --------------------
RANDOM_STATE = 42
FOUR_CLASSES = ["legit", "watchlist", "likely_scam", "scam"]
REP_4 = {
    "legit": 0.00,
    "watchlist": (0.30 + 0.475) / 2.0,
    "likely_scam": (0.625 + 0.75) / 2.0,
    "scam": 0.90,
}

# -------------------- imbalanced-learn guard (hard fail) --------------------
def ensure_imbalanced_learn():
    try:
        import imblearn  # noqa: F401
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        return ImbPipeline, SMOTE, RandomUnderSampler
    except Exception as e:
        raise ImportError(
            "imbalanced-learn was not importable. Make sure you've installed the versions "
            "from requirements.txt, then restart the interpreter."
        ) from e

ImbPipeline, SMOTE, RandomUnderSampler = ensure_imbalanced_learn()

# -------------------- utils --------------------
def seed_everything(seed: int = RANDOM_STATE):
    random.seed(seed); np.random.seed(seed)

def normalize_label(s: str) -> str:
    if not isinstance(s, str): return "neutral"
    x = s.strip().lower().replace(" ", "_")
    mapping = {
        "legit": "legitimate",
        "legitmate": "legitimate",
        "scam_response": "scam/scam_response",
        "scam response": "scam/scam_response",
        "scamresponse": "scam/scam_response",
        "highly_suspicious": "higly_suspicious",
    }
    return mapping.get(x, x)

def clean_text(t: str) -> str:
    if not isinstance(t, str): return ""
    t = t.lower()
    t = re.sub(r"http[s]?://\S+", " URL ", t)
    t = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " EMAIL ", t)
    t = re.sub(r"\b(?:\+?\d[\d\-\s]{7,}\d)\b", " PHONE ", t)
    t = re.sub(r"\b\d+\b", " NUM ", t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", t).strip()

def probs_to_expected_scores(proba: np.ndarray, cls_order: List[str]) -> np.ndarray:
    rep_vec = np.array([REP_4[c] for c in cls_order], dtype=np.float32)
    return (proba * rep_vec).sum(axis=1)

def expected_score_from_proba(proba_row: np.ndarray, classes_order: List[str]) -> float:
    rep_vec = np.array([REP_4[c] for c in classes_order], dtype=np.float32)
    return float((proba_row * rep_vec).sum())

def make_features():
    word_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
    char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), min_df=2, max_df=0.95, sublinear_tf=True)
    return FeatureUnion([('word', word_tfidf), ('char', char_tfidf)])

def build_smote_pipeline(features):
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    under = RandomUnderSampler(random_state=RANDOM_STATE, sampling_strategy='auto')
    logreg = LogisticRegression(C=4.0, solver='lbfgs', max_iter=2000, multi_class="multinomial", class_weight="balanced")
    return ImbPipeline(steps=[('features', features), ('smote', smote), ('under', under), ('clf', logreg)])

def predict_texts(texts, model):
    """Utility for quick interactive tests."""
    if isinstance(texts, str):
        texts = [texts]
    X = [clean_text(t) for t in texts]
    proba = model.predict_proba(X)
    preds = model.predict(X)
    classes_order = list(model.classes_)

    rows = []
    for i, txt in enumerate(texts):
        p = proba[i]
        colmap = {c: j for j, c in enumerate(classes_order)}
        rows.append({
                    "pred": preds[i],
                    "p_legit": float(p[colmap["legit"]]),
                    "p_watchlist": float(p[colmap["watchlist"]]),
                    "p_likely_scam": float(p[colmap["likely_scam"]]),
                    "p_scam": float(p[colmap["scam"]]),
                    "expected_score": expected_score_from_proba(p, classes_order),
                    "text": txt[:200] + ("…" if len(txt) > 200 else "")})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return rows

# -------------------- training / CV --------------------
def run_cv_train(data_path: str, folds: int, save_model: str | None, test_text: str | None):
    seed_everything(RANDOM_STATE)

    # Load
    df = pd.read_csv(data_path)
    assert {"TEXT", "SCORE", "LABEL"}.issubset(df.columns), "CSV must include TEXT, SCORE, LABEL"
    df["LABEL"] = df["LABEL"].astype(str).apply(normalize_label)
    merge_to = {
        "legitimate": "legit", "neutral": "legit",
        "slightly_suspicious": "watchlist", "suspicious": "watchlist",
        "higly_suspicious": "likely_scam", "potential_scam": "likely_scam",
        "scam": "scam", "scam/scam_response": "scam",
    }
    df["LABEL_4CLASS"] = df["LABEL"].map(merge_to).fillna("legit")
    df["TEXT"] = df["TEXT"].fillna("").astype(str).apply(clean_text)
    df["SCORE"] = pd.to_numeric(df["SCORE"], errors="coerce").clip(0, 1)
    df = df.dropna(subset=["TEXT", "SCORE"]).reset_index(drop=True)

    print("\n=== Initial distribution (4 classes) ===")
    print(df["LABEL_4CLASS"].value_counts().reindex(FOUR_CLASSES).fillna(0).astype(int))

    X_all = df["TEXT"]
    y_all = df["LABEL_4CLASS"]
    score_all = df["SCORE"].astype(np.float32)

    features = make_features()
    print(f"\n>> Running StratifiedKFold CV: k={folds} | SMOTE=True")

    best_fold = None
    best_logloss = float("inf")
    best_report = None

    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_all, y_all), 1):
        Xtr, Xva = X_all.iloc[tr_idx], X_all.iloc[va_idx]
        ytr, yva = y_all.iloc[tr_idx], y_all.iloc[va_idx]

        print(f"\n---------- FOLD {fold} ----------")
        print("Before resampling (train):", dict(Counter(ytr)))

        # Show post-SMOTE/UNDER distributions (debug on vectorized X)
        feats_dbg = clone(features)
        Xtr_vec = feats_dbg.fit_transform(Xtr)
        sm_dbg = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
        X_sm, y_sm = sm_dbg.fit_resample(Xtr_vec, ytr)
        print("After SMOTE:", dict(Counter(y_sm)))
        under_dbg = RandomUnderSampler(random_state=RANDOM_STATE, sampling_strategy='auto')
        X_bal, y_bal = under_dbg.fit_resample(X_sm, y_sm)
        print("After UNDER :", dict(Counter(y_bal)))

        pipe = build_smote_pipeline(features)
        pipe.fit(Xtr, ytr)

        proba_va = pipe.predict_proba(Xva)
        pred_va = pipe.predict(Xva)
        cls_order = list(pipe.classes_)
        y_idx = pd.Categorical(yva, categories=cls_order).codes

        ll = log_loss(y_idx, proba_va, labels=range(len(cls_order)))
        exp = probs_to_expected_scores(proba_va, cls_order)
        mae = mean_absolute_error(score_all.iloc[va_idx].values, exp)
        mse = mean_squared_error(score_all.iloc[va_idx].values, exp)
        rep = classification_report(yva, pred_va, labels=FOUR_CLASSES, zero_division=0, digits=4)

        print("\nLog-loss (fold): %.5f  |  MAE: %.4f  |  MSE: %.4f" % (ll, mae, mse))
        print("\nClassification report (VAL):\n", rep)

        if ll < best_logloss:
            best_logloss = ll
            best_fold = fold
            best_report = rep

    print("\n==================== SUMMARY ====================")
    print(f"Best fold: {best_fold}  (log-loss = {best_logloss:.5f})")
    print("\nFull report for best fold:\n")
    print(best_report)

    # Fit final model on all data and optionally save
    final_pipe = build_smote_pipeline(features)
    final_pipe.fit(X_all, y_all)
    if save_model:
        os.makedirs(os.path.dirname(save_model), exist_ok=True)
        joblib.dump(final_pipe, save_model)
        print(f"\nModel saved to: {save_model}")

    # Optional quick test
    if test_text:
        print("\n=== Quick test on custom text ===")
        predict_texts(test_text, final_pipe)

    return final_pipe

# -------------------- CLI --------------------
def main():
    parser = argparse.ArgumentParser(description="Train Scam Risk model with TF-IDF + LR + SMOTE.")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV with columns: TEXT, SCORE, LABEL")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--save-model", type=str, default="./artifacts/model_4class_lr_cv/lr_tfidf_4class_cv.joblib",
                        help="Path to save fitted pipeline (default under ./artifacts/...)")
    parser.add_argument("--test-text", type=str, default="", help="Optional text to score after training.")
    args = parser.parse_args()

    run_cv_train(
        data_path=args.data,
        folds=args.folds,
        save_model=args.save_model,
        test_text=args.test_text if args.test_text.strip() else None
    )

if __name__ == "__main__":
    main()
