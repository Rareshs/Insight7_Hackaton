
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scam Risk Ensemble (CV-enabled): TF-IDF+LogReg (multiclass) + DistilRoBERTa (regression)
- Adds stratified K-Fold cross-validation for the Logistic Regression branch
- Optional (heavier) CV for the Transformer regression branch (disabled by default)
- Produces EDA on label balance per split (tables + PNG charts)
- Keeps single-split training path for parity with the original script
"""
import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
from joblib import dump, load

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

import matplotlib.pyplot as plt

# ---------------------------
# Configuration
# ---------------------------

DEFAULT_DATA_PATH = 'Transcripts_scored_risk.csv'
ARTIFACT_DIR = 'artifacts_scams'
TRANSFORMER_NAME = 'distilroberta-base'
SEED = 42

# ---------------------------
# Utilities
# ---------------------------

def seed_everything(seed: int = SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

RISK_BUCKETS = [
    ("legitimate",          0.00, 0.00),  # exactly 0
    ("neutral",             0.01, 0.20),
    ("slightly_suspicious", 0.20, 0.40),
    ("suspicious",          0.40, 0.55),
    ("highly_suspicious",    0.55, 0.70),
    ("potential_scam",      0.70, 0.80),
    ("scam/scam_response",  0.80, 1.00),  # last bucket inclusive on high end
]

BUCKET_REP = {
    'legitimate': 0.00,
    'neutral': (0.01 + 0.20)/2.0,
    'slightly_suspicious': (0.20 + 0.40)/2.0,
    'suspicious': (0.40 + 0.55)/2.0,
    'highly_suspicious': (0.55 + 0.70)/2.0,
    'potential_scam': (0.70 + 0.80)/2.0,
    'scam/scam_response': (0.80 + 1.00)/2.0,
}

def map_score_to_bucket(score: float) -> str:
    for name, lo, hi in RISK_BUCKETS:
        if name == 'scam/scam_response':
            if lo <= score <= hi: return name
        else:
            if lo <= score < hi: return name
    return 'scam/scam_response' if score > 1.0 else 'legitimate'

def expected_score_from_probs(class_names: List[str], probs: np.ndarray) -> float:
    rep = np.array([BUCKET_REP[c] for c in class_names], dtype=np.float32)
    return float(np.dot(rep, probs))

# ---------------------------
# Classic Model: TF-IDF + Logistic Regression (Multiclass)
# ---------------------------

def build_logreg_pipeline() -> Pipeline:
    word_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,2),
                                 max_features=100000, min_df=2, max_df=0.6, sublinear_tf=True)
    char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,6),
                                 max_features=100000, min_df=1, max_df=0.75, sublinear_tf=True)
    features = FeatureUnion([('word', word_tfidf), ('char', char_tfidf)])
    clf = LogisticRegression(C=10.0, solver='lbfgs', max_iter=2000)
    return Pipeline([('features', features), ('clf', clf)])

# ---------------------------
# Transformer Regression Dataset
# ---------------------------

class ScoreDataset(Dataset):
    def __init__(self, encodings, scores):
        self.encodings = encodings
        self.scores = scores
    def __len__(self):
        return len(self.scores)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor([self.scores[idx]], dtype=torch.float)
        return item

# ---------------------------
# Train Transformer as Regression
# ---------------------------

def train_transformer_regression(texts_train: List[str], scores_train: np.ndarray,
                                 texts_val: List[str], scores_val: np.ndarray,
                                 model_name: str = TRANSFORMER_NAME,
                                 epochs: int = 2, batch_size: int = 16, lr: float = 2e-5,
                                 max_length: int = 256, device: str = None) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, problem_type='regression')
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_enc = tokenizer(texts_train, truncation=True, padding=True, max_length=max_length)
    val_enc = tokenizer(texts_val, truncation=True, padding=True, max_length=max_length)

    train_ds = ScoreDataset(train_enc, scores_train.tolist())
    val_ds = ScoreDataset(val_enc, scores_val.tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*num_training_steps), num_training_steps=num_training_steps)
    best_sd = {k: v.cpu() for k, v in model.state_dict().items()}
    best_val_mae = float('inf')

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                logits = out.logits.detach().cpu().numpy().reshape(-1)
                logits = np.clip(logits, 0.0, 1.0)
                logits = np.nan_to_num(logits, nan=0.0)
                preds.extend(logits.tolist())
                gts.extend(batch['labels'].detach().cpu().numpy().reshape(-1).tolist())

        gts = np.nan_to_num(np.asarray(gts, dtype=np.float32), nan=0.0)
        preds = np.asarray(preds, dtype=np.float32)
        val_mae = mean_absolute_error(gts, preds)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_sd = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_sd)
    return model, tokenizer

def predict_transformer_scores(model, tokenizer, texts: List[str], max_length: int = 256, device: str = None) -> np.ndarray:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    ds = ScoreDataset(enc, [0.0]*len(texts))
    dl = DataLoader(ds, batch_size=32, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            logits = out.logits.detach().cpu().numpy().reshape(-1)
            logits = np.clip(logits, 0.0, 1.0)
            preds.extend(logits.tolist())
    return np.array(preds, dtype=np.float32)

# ---------------------------
# Metrics & EDA helpers
# ---------------------------

def evaluate_classification(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return report

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {'mae': float(mae), 'mse': float(mse)}

def eda_label_balance_table(y: List[str], title: str, out_csv: str = None) -> pd.DataFrame:
    counts = pd.Series(y).value_counts().sort_index()
    dist = (counts / counts.sum()).rename("ratio")
    table = pd.concat([counts.rename("count"), dist], axis=1)
    table.index.name = "label"
    if out_csv:
        table.to_csv(out_csv, index=True)
    print(f"\n=== Label balance: {title} ===\n{table}\n")
    return table

def eda_plot_kfold_balance(fold_counts: Dict[int, Counter], classes: List[str], out_png: str):
    # Build a DataFrame: rows = classes, columns = fold_i counts
    df = pd.DataFrame({f"fold_{k}": pd.Series(cnt) for k, cnt in fold_counts.items()}).reindex(classes).fillna(0)
    ax = df.plot(kind="bar", figsize=(10,6))
    ax.set_title("Label counts per CV fold")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# ---------------------------
# Data loading
# ---------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["TEXT", "SCORE", "LABEL"]:
        if col not in df.columns:
            raise ValueError(f"Dataset must have {col} column")

    df["TEXT"] = df["TEXT"].astype(str).str.strip()
    df = df[df["TEXT"].str.len() > 0].copy()

    df["SCORE"] = pd.to_numeric(df["SCORE"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["SCORE"]).copy()
    df["SCORE"] = df["SCORE"].astype(float).clip(0.0, 1.0)

    class_names = [b[0] for b in RISK_BUCKETS]
    def fix_label(row):
        lbl = str(row["LABEL"]) if pd.notna(row["LABEL"]) else ""
        if lbl not in class_names:
            return map_score_to_bucket(float(row["SCORE"]))
        return lbl
    df["LABEL"] = df.apply(fix_label, axis=1)
    return df

# ---------------------------
# Single split training (parity with original)
# ---------------------------

def train_single_split(df: pd.DataFrame, out_dir: str, transformer_epochs: int, ensemble_alpha: float):
    seed_everything(SEED)
    os.makedirs(out_dir, exist_ok=True)

    class_names = [b[0] for b in RISK_BUCKETS]
    X = df['TEXT'].tolist()
    y_labels = df['LABEL'].tolist()
    y_scores = df['SCORE'].values.astype(np.float32)

    # EDA on full dataset
    eda_label_balance_table(y_labels, "full_dataset", os.path.join(out_dir, "eda_full_label_balance.csv"))

    # Split (with safe stratify)
    stratify_arg = y_labels if all(c >= 2 for c in Counter(y_labels).values()) else None
    X_tr, X_te, ylab_tr, ylab_te, ysc_tr, ysc_te = train_test_split(
        X, y_labels, y_scores, test_size=0.2, random_state=SEED, stratify=stratify_arg
    )

    # EDA on split
    eda_label_balance_table(ylab_tr, "train_split", os.path.join(out_dir, "eda_train_label_balance.csv"))
    eda_label_balance_table(ylab_te, "val_split", os.path.join(out_dir, "eda_val_label_balance.csv"))

    # Train LR
    logreg_pipe = build_logreg_pipeline()
    logreg_pipe.fit(X_tr, ylab_tr)
    proba_te = logreg_pipe.predict_proba(X_te)
    class_list = list(logreg_pipe.classes_)
    exp_scores_lr = np.array([expected_score_from_probs(class_list, p) for p in proba_te], dtype=np.float32)

    # Train Transformer regression
    model, tokenizer = train_transformer_regression(X_tr, ysc_tr, X_te, ysc_te, epochs=transformer_epochs)
    pred_scores_tr = predict_transformer_scores(model, tokenizer, X_te)

    # Ensemble
    final_scores = np.clip(ensemble_alpha * pred_scores_tr + (1.0 - ensemble_alpha) * exp_scores_lr, 0.0, 1.0)
    final_labels = [map_score_to_bucket(s) for s in final_scores]

    # Metrics
    cls_report = evaluate_classification(ylab_te, final_labels, class_names)
    reg_logreg = evaluate_regression(ysc_te, exp_scores_lr)
    reg_transformer = evaluate_regression(ysc_te, pred_scores_tr)
    reg_ensemble = evaluate_regression(ysc_te, final_scores)

    metrics = {
        'classification_report': cls_report,
        'regression_logreg_expected': reg_logreg,
        'regression_transformer': reg_transformer,
        'regression_ensemble': reg_ensemble,
        'ensemble_alpha': ensemble_alpha,
        'classes_in_order': class_list,
    }

    # Save artifacts
    dump(logreg_pipe, os.path.join(out_dir, 'logreg_tfidf.joblib'))
    model.save_pretrained(os.path.join(out_dir, 'transformer_reg'))
    tokenizer.save_pretrained(os.path.join(out_dir, 'transformer_reg'))
    with open(os.path.join(out_dir, 'bucket_config.json'), 'w', encoding='utf-8') as f:
        json.dump({'risk_buckets': RISK_BUCKETS, 'bucket_representatives': BUCKET_REP, 'ensemble_alpha': ensemble_alpha}, f, indent=2)
    with open(os.path.join(out_dir, 'metrics_single_split.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print("=== Single-split Metrics (summary) ===")
    print(f"LogReg Expected Score -> MAE: {reg_logreg['mae']:.4f}, MSE: {reg_logreg['mse']:.4f}")
    print(f"Transformer Regression -> MAE: {reg_transformer['mae']:.4f}, MSE: {reg_transformer['mse']:.4f}")
    print(f"Ensemble (alpha={ensemble_alpha:.2f}) -> MAE: {reg_ensemble['mae']:.4f}, MSE: {reg_ensemble['mse']:.4f}")
    return metrics

# ---------------------------
# Cross-validation training (LogReg mandatory, Transformer optional)
# ---------------------------

def train_cv(df: pd.DataFrame, out_dir: str, k_folds: int, transformer_epochs: int, ensemble_alpha: float, cv_transformer: bool):
    seed_everything(SEED)
    os.makedirs(out_dir, exist_ok=True)

    class_names = [b[0] for b in RISK_BUCKETS]
    X = np.array(df['TEXT'].tolist(), dtype=object)
    y_labels = np.array(df['LABEL'].tolist(), dtype=object)
    y_scores = df['SCORE'].values.astype(np.float32)

    # EDA on full dataset
    eda_label_balance_table(y_labels.tolist(), "full_dataset", os.path.join(out_dir, "eda_full_label_balance.csv"))

    # K-Fold with stratification on labels
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    fold_counts = {}
    oof_lr_expected = np.zeros(len(X), dtype=np.float32)
    oof_tr_score = np.zeros(len(X), dtype=np.float32)

    # We will store per-fold metrics
    fold_metrics = []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X, y_labels), start=1):
        X_tr, X_te = X[tr_idx].tolist(), X[te_idx].tolist()
        ylab_tr, ylab_te = y_labels[tr_idx].tolist(), y_labels[te_idx].tolist()
        ysc_tr, ysc_te = y_scores[tr_idx], y_scores[te_idx]

        # EDA per fold (label counts on val)
        fold_counts[fold] = Counter(ylab_te)

        # Train LR on fold
        logreg_pipe = build_logreg_pipeline()
        logreg_pipe.fit(X_tr, ylab_tr)
        proba_te = logreg_pipe.predict_proba(X_te)
        class_list = list(logreg_pipe.classes_)
        exp_scores_lr = np.array([expected_score_from_probs(class_list, p) for p in proba_te], dtype=np.float32)
        oof_lr_expected[te_idx] = exp_scores_lr

        # Transformer per fold (optional, slower). Otherwise train once on whole data at end.
        if cv_transformer:
            model, tokenizer = train_transformer_regression(X_tr, ysc_tr, X_te, ysc_te, epochs=transformer_epochs)
            pred_scores_tr = predict_transformer_scores(model, tokenizer, X_te)
            oof_tr_score[te_idx] = pred_scores_tr

            # Ensemble & metrics for this fold
            final_scores = np.clip(ensemble_alpha * pred_scores_tr + (1.0 - ensemble_alpha) * exp_scores_lr, 0.0, 1.0)
            final_labels = [map_score_to_bucket(s) for s in final_scores]

            cls_report = evaluate_classification(ylab_te, final_labels, class_names)
            reg_logreg = evaluate_regression(ysc_te, exp_scores_lr)
            reg_transformer = evaluate_regression(ysc_te, pred_scores_tr)
            reg_ensemble = evaluate_regression(ysc_te, final_scores)

            fold_metrics.append({
                'fold': fold,
                'classification_report': cls_report,
                'regression_logreg_expected': reg_logreg,
                'regression_transformer': reg_transformer,
                'regression_ensemble': reg_ensemble,
            })
        else:
            # Metrics without transformer per fold (we'll ensemble later using a single transformer model)
            reg_logreg = evaluate_regression(ysc_te, exp_scores_lr)
            fold_metrics.append({
                'fold': fold,
                'regression_logreg_expected': reg_logreg
            })

    # Plot EDA of per-fold label balance
    eda_plot_kfold_balance(fold_counts, class_names, os.path.join(out_dir, f"eda_k{len(fold_counts)}_label_balance.png"))
    # Save counts table
    counts_df = pd.DataFrame({f"fold_{k}": pd.Series(v) for k, v in fold_counts.items()}).reindex(class_names).fillna(0).astype(int)
    counts_df.to_csv(os.path.join(out_dir, f"eda_k{len(fold_counts)}_label_counts.csv"))

    # If transformer wasn't run in CV, train it once on full data and ensemble with OOF LR expected
    if not cv_transformer:
        # Use a simple 80/20 internal split for transformer
        stratify_arg = y_labels if all(c >= 2 for c in Counter(y_labels).values()) else None
        X_tr, X_te, ysc_tr, ysc_te = train_test_split(
            X.tolist(), y_scores, test_size=0.2, random_state=SEED, stratify=stratify_arg
        )
        model, tokenizer = train_transformer_regression(X_tr, ysc_tr, X_te, ysc_te, epochs=transformer_epochs)
        # Predict transformer on all samples for consistency
        tr_all_scores = predict_transformer_scores(model, tokenizer, X.tolist())
        oof_tr_score = tr_all_scores  # use the same transformer estimation for every sample

    # Final OOF ensemble scores
    oof_scores = np.clip(ensemble_alpha * oof_tr_score + (1.0 - ensemble_alpha) * oof_lr_expected, 0.0, 1.0)
    oof_labels = [map_score_to_bucket(s) for s in oof_scores]

    # EDA on OOF predictions vs true labels (balance)
    eda_label_balance_table(oof_labels, "oof_predicted_labels", os.path.join(out_dir, "eda_oof_predicted_labels.csv"))

    # Aggregate metrics on all OOF
    reg_lr = evaluate_regression(y_scores, oof_lr_expected)
    reg_tr = evaluate_regression(y_scores, oof_tr_score)
    reg_ens = evaluate_regression(y_scores, oof_scores)

    metrics = {
        'oof_regression_logreg_expected': reg_lr,
        'oof_regression_transformer': reg_tr,
        'oof_regression_ensemble': reg_ens,
        'k_folds': k_folds,
        'cv_transformer': cv_transformer,
        'fold_metrics': fold_metrics,
    }

    with open(os.path.join(out_dir, f'metrics_cv_k{k_folds}.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # Save LR pipeline trained on full data for serving
    full_lr = build_logreg_pipeline()
    full_lr.fit(X.tolist(), y_labels.tolist())
    dump(full_lr, os.path.join(out_dir, 'logreg_tfidf.joblib'))

    # Save transformer model trained on full data for serving (reuse the one we trained if cv_transformer=False)
    if cv_transformer:
        # train once on full data for artifacts
        stratify_arg = y_labels if all(c >= 2 for c in Counter(y_labels).values()) else None
        X_tr, X_te, ysc_tr, ysc_te = train_test_split(
            X.tolist(), y_scores, test_size=0.2, random_state=SEED, stratify=stratify_arg
        )
        model, tokenizer = train_transformer_regression(X_tr, ysc_tr, X_te, ysc_te, epochs=transformer_epochs)

    # Save artifacts
    model.save_pretrained(os.path.join(out_dir, 'transformer_reg'))
    tokenizer.save_pretrained(os.path.join(out_dir, 'transformer_reg'))
    with open(os.path.join(out_dir, 'bucket_config.json'), 'w', encoding='utf-8') as f:
        json.dump({'risk_buckets': RISK_BUCKETS, 'bucket_representatives': BUCKET_REP, 'ensemble_alpha': ensemble_alpha}, f, indent=2)

    print("=== CV Metrics (OOF) ===")
    print(f"OOF LogReg Expected Score -> MAE: {reg_lr['mae']:.4f}, MSE: {reg_lr['mse']:.4f}")
    print(f"OOF Transformer -> MAE: {reg_tr['mae']:.4f}, MSE: {reg_tr['mse']:.4f}")
    print(f"OOF Ensemble (alpha={ensemble_alpha:.2f}) -> MAE: {reg_ens['mae']:.4f}, MSE: {reg_ens['mse']:.4f}")
    return metrics

# ---------------------------
# Serving
# ---------------------------

@dataclass
class EnsembleService:
    logreg_pipe: Any
    transformer_model: Any
    tokenizer: Any
    class_names: List[str]
    ensemble_alpha: float

    @classmethod
    def load(cls, artifact_dir: str = ARTIFACT_DIR):
        logreg_pipe = load(os.path.join(artifact_dir, 'logreg_tfidf.joblib'))
        transformer_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(artifact_dir, 'transformer_reg'))
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(artifact_dir, 'transformer_reg'))
        with open(os.path.join(artifact_dir, 'bucket_config.json'), 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        class_names = list(logreg_pipe.classes_)
        alpha = float(cfg.get('ensemble_alpha', 0.5))
        return cls(logreg_pipe, transformer_model, tokenizer, class_names, alpha)

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        probs = self.logreg_pipe.predict_proba(texts)
        exp_scores = np.array([expected_score_from_probs(self.class_names, p) for p in probs], dtype=np.float32)
        scores_tr = predict_transformer_scores(self.transformer_model, self.tokenizer, texts)
        scores = np.clip(self.ensemble_alpha * scores_tr + (1.0 - self.ensemble_alpha) * exp_scores, 0.0, 1.0)
        labels = [map_score_to_bucket(float(s)) for s in scores]
        top3 = [sorted(list(zip(self.class_names, p)), key=lambda t: t[1], reverse=True)[:3] for p in probs]
        return [
            {'score': float(s), 'label': lbl, 'lr_expected_score': float(es), 'tr_score': float(ts),
             'top3_lr': [(c, float(p)) for c, p in t3]}
            for s, lbl, es, ts, t3 in zip(scores, labels, exp_scores, scores_tr, top3)
        ]
# ---------------------------
# Conversation test utility
# ---------------------------

def run_conversation_test(conversation: List[str], artifact_dir: str = ARTIFACT_DIR, alpha: float = 0.5):
    """
    Simulate a conversation: print scam-risk scores + labels for each message.
    """
    svc = EnsembleService.load(artifact_dir)
    results = svc.predict(conversation)

    print("\n=== Conversation Risk Test ===")
    for i, (msg, res) in enumerate(zip(conversation, results), start=1):
        print(f"[Step {i}]")
        print(f" Message: {msg}")
        print(f" → Score: {res['score']:.3f}")
        print(f" → Label: {res['label']}")
        print(f" → Transformer Score: {res['tr_score']:.3f}")
        print(f" → LR Expected Score: {res['lr_expected_score']:.3f}")
        print(f" → Top-3 LR classes: {res['top3_lr']}")
        print("-"*50)

# ---------------------------
# Conversation scoring utilities (avg + critical warning)
# ---------------------------

def run_conversation_scoring(
    conversation: List[str],
    artifact_dir: str = ARTIFACT_DIR,
    threshold: float = 0.70
) -> Dict[str, Any]:
    """
    Compute per-message scores for the (scammer-only) conversation and the average.
    Prints a full breakdown to console and returns a structured result.
    """
    svc = EnsembleService.load(artifact_dir)
    results = svc.predict(conversation)

    print("\n=== Conversation Risk Scoring (Scammer-only inputs) ===")
    print(f"Ensemble alpha (transformer weight): {svc.ensemble_alpha:.2f}")
    print(f"Critical warning threshold (average): {threshold:.2f}\n")

    per_steps = []
    for i, (msg, res) in enumerate(zip(conversation, results), start=1):
        # contribution breakdown (purely cosmetic/inspectable)
        contrib_tr = svc.ensemble_alpha * res["tr_score"]
        contrib_lr = (1.0 - svc.ensemble_alpha) * res["lr_expected_score"]

        print(f"[Step {i}]")
        print(f" Message: {msg}")
        print(f" Transformer score:      {res['tr_score']:.3f}   (alpha * tr = {contrib_tr:.3f})")
        print(f" LR expected score:      {res['lr_expected_score']:.3f}   ((1-alpha) * lr = {contrib_lr:.3f})")
        print(f"  Final ensemble score: {res['score']:.3f}   Label: {res['label']}")
        print(f" Top-3 LR classes:       {res['top3_lr']}")
        print("-" * 70)

        per_steps.append({
            "text": msg,
            "score": res["score"],
            "label": res["label"],
            "lr_expected_score": res["lr_expected_score"],
            "tr_score": res["tr_score"],
            "top3_lr": res["top3_lr"],
            "alpha": svc.ensemble_alpha,
            "contrib_transformer": contrib_tr,
            "contrib_logreg": contrib_lr,
        })

    scores = [s["score"] for s in per_steps]
    avg_score = float(np.mean(scores)) if scores else 0.0
    critical_warning = bool(avg_score >= threshold)

    print("\n=== Conversation Summary ===")
    print(f" Messages scored: {len(scores)}")
    print(f" Average score:   {avg_score:.3f}")
    print(f" Decision:    {'CRITICAL WARNING' if critical_warning else 'normal monitoring'} "
          f"(avg {'>=' if critical_warning else '<'} {threshold:.2f})")

    return {
        "average_score": avg_score,
        "critical_warning": critical_warning,
        "threshold": threshold,
        "steps": per_steps,
    }
def print_conversation_scoring(conversation: List[str], artifact_dir: str = ARTIFACT_DIR, threshold: float = 0.70):
    """
    Thin wrapper that only prints; ignores the returned dict.
    """
    _ = run_conversation_scoring(conversation, artifact_dir=artifact_dir, threshold=threshold)

def run_service(host: str = '0.0.0.0', port: int = 8000, artifact_dir: str = ARTIFACT_DIR):
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    svc = EnsembleService.load(artifact_dir)
    app = FastAPI(title="Scam Risk Ensemble API")

    class PredictIn(BaseModel):
        texts: List[str]

    @app.post("/predict")
    def predict(inp: PredictIn):
        res = svc.predict(inp.texts)
        return {'results': res}

    uvicorn.run(app, host=host, port=port)

# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Train (single or CV) and serve Scam Risk Ensemble")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Single-split training (parity with original)")
    p_train.add_argument("--data", type=str, default=DEFAULT_DATA_PATH)
    p_train.add_argument("--out", type=str, default=ARTIFACT_DIR)
    p_train.add_argument("--epochs", type=int, default=2)
    p_train.add_argument("--alpha", type=float, default=0.5)

    p_traincv = sub.add_parser("train-cv", help="Cross-validated training (Stratified K-Fold)")
    p_traincv.add_argument("--data", type=str, default=DEFAULT_DATA_PATH)
    p_traincv.add_argument("--out", type=str, default=ARTIFACT_DIR)
    p_traincv.add_argument("--epochs", type=int, default=2, help="Transformer epochs per fold or for single transformer")
    p_traincv.add_argument("--alpha", type=float, default=0.5, help="Ensemble alpha (transformer weight)")
    p_traincv.add_argument("--k", type=int, default=3, help="Number of folds (keep small if classes are tiny)")
    p_traincv.add_argument("--cv-transformer", action="store_true", help="Also CV the transformer (slower)")

    p_serve = sub.add_parser("serve", help="Start FastAPI service using saved artifacts")
    p_serve.add_argument("--host", type=str, default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--artifacts", type=str, default=ARTIFACT_DIR)

    p_score = sub.add_parser("score-convo", help="Score a scammer-only conversation from a text file (one line per message)")
    p_score.add_argument("--file", type=str, required=True, help="Path to .txt with one message per line")
    p_score.add_argument("--artifacts", type=str, default=ARTIFACT_DIR)
    p_score.add_argument("--threshold", type=float, default=0.70)

    args = parser.parse_args()

    if args.cmd == "train":
        df = load_data(args.data)
        train_single_split(df, args.out, args.epochs, args.alpha)
    elif args.cmd == "train-cv":
        df = load_data(args.data)
        train_cv(df, args.out, args.k, args.epochs, args.alpha, args.cv_transformer)
    elif args.cmd == "serve":
        run_service(host=args.host, port=args.port, artifact_dir=args.artifacts)
    elif args.cmd == "score-convo":
        with open(args.file, "r", encoding="utf-8") as f:
            convo = [line.strip() for line in f.readlines() if line.strip()]
        print_conversation_scoring(convo, artifact_dir=args.artifacts, threshold=args.threshold)

if __name__ == "__main__":
    main()
