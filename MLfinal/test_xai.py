# test_xai.py — per mesaj + summary final (media + decizie)
import argparse, json, re, string, sys, os
from typing import List, Tuple, Iterable
from scipy.sparse import issparse
from math import isfinite
import numpy as np
import pandas as pd
import joblib
import torch

# ===== Bucket representatives (adjust if you use a different class set) =====
BUCKET_REP = {
    'legitimate': 0.00,
    'neutral': (0.01 + 0.20) / 2.0,
    'slightly_suspicious': (0.20 + 0.40) / 2.0,
    'suspicious': (0.40 + 0.55) / 2.0,
    'highly_suspicious': (0.55 + 0.70) / 2.0,
    'potential_scam': (0.70 + 0.80) / 2.0,
    'scam/scam_response': (0.80 + 1.00) / 2.0,
}

# ============== helpers ==============
def clean_text(t: str) -> str:
    """Normalize text: lowercase, mask URLs/emails/phones/numbers, strip punctuation and extra spaces."""
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"http[s]?://\S+", " URL ", t)
    t = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " EMAIL ", t)
    t = re.sub(r"\b(?:\+?\d[\d\-\s]{7,}\d)\b", " PHONE ", t)
    t = re.sub(r"\b\d+\b", " NUM ", t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", t).strip()

def _clean_line(s: str) -> str:
    """Curăță ghilimele/virgula de la capete + spațiile, ca să eviți dublurile la print."""
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    if s.endswith(","):
        s = s[:-1].strip()
    return s

def expected_score_from_probs(class_names: List[str], probs: np.ndarray) -> float:
    """Map class probabilities to a scalar risk via bucket representatives."""
    reps = np.array([BUCKET_REP[str(c)] for c in class_names], dtype=np.float32)
    return float(np.dot(reps, probs))

def load_transformer(artifacts_dir: str):
    """Load optional transformer regression head (tokenizer + model)."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    try:
        tdir = os.path.join(artifacts_dir, "transformer_reg")
        tok = AutoTokenizer.from_pretrained(tdir, local_files_only=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(tdir, local_files_only=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mdl.to(device)
        return tok, mdl, device
    except Exception:
        return None, None, None

def transformer_score(tok, mdl, device, text: str) -> float:
    """Score text with the transformer model; clip to [0, 1]."""
    enc = tok([text], truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mdl(**enc)
        val = out.logits.detach().cpu().numpy().reshape(-1)[0]
    return float(np.clip(val, 0.0, 1.0))

def text_bar(p: float, width: int = 24) -> str:
    """ASCII bar used elsewhere if needed."""
    k = int(round(p * width))
    return "[" + "█" * k + " " * (width - k) + "]"

# -------- term contributions for predicted class (LR + TFIDF) --------
def explain_logreg(pipe, raw_text: str, top_k: int = 8):
    """Explain LR decision on cleaned text; return prediction, internals, and feature contributions."""
    text = clean_text(raw_text)
    probs = pipe.predict_proba([text])[0]
    classes = [str(c) for c in pipe.classes_]
    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    pred_prob = float(probs[pred_idx])

    clf = pipe.named_steps['clf']
    feats = pipe.named_steps['features']
    X = feats.transform([text])              # may be sparse or dense
    coef_vec = clf.coef_[pred_idx]          # (n_features,)
    bias = float(clf.intercept_[pred_idx])

    # logit and contribution sum depend on X type
    if issparse(X):
        contrib_val = float((X @ coef_vec).ravel()[0])
    else:
        contrib_val = float((X * coef_vec).sum())
    logit = contrib_val + bias

    # feature names from FeatureUnion
    names = []
    for name, trans in feats.transformer_list:
        if hasattr(trans, 'get_feature_names_out'):
            nms = list(trans.get_feature_names_out())
            names.extend(nms)
        else:
            names.append(name)

    # per-feature contributions w_i = x_i * coef_i (top +/-)
    pairs = []
    if issparse(X):
        Xcoo = X.tocoo()
        for col, val in zip(Xcoo.col, Xcoo.data):
            if col < len(coef_vec):
                w = float(val * coef_vec[col])
                term = names[col] if col < len(names) else f"f{col}"
                pairs.append((term, w))
    else:
        row = X[0]
        nz_idx = np.where(row != 0)[0]
        for col in nz_idx:
            w = float(row[col] * coef_vec[col])
            term = names[col] if col < len(names) else f"f{col}"
            pairs.append((term, w))

    pairs.sort(key=lambda t: t[1], reverse=True)
    top_pos = [(term, w) for term, w in pairs if w > 0][:top_k]
    top_neg = [(term, w) for term, w in sorted(pairs, key=lambda t: t[1]) if w < 0][:top_k]

    probs_dict = {cls: float(p) for cls, p in zip(classes, probs)}
    return pred_label, pred_prob, probs_dict, float(logit), float(bias), top_pos, top_neg, pairs

# --- Pretty printing helpers ---
def _pretty_token(term: str) -> str:
    """Make feature names readable (trim, compress spaces, drop tiny fragments)."""
    t = term.replace("\u00A0", " ")
    t = re.sub(r"\s+", " ", t.strip())
    if len(t) <= 2 and not re.match(r"^[A-Za-z0-9]+$", t):
        return ""
    return t

def _collapse_contribs(pairs: Iterable[tuple[str, float]], top_k: int = 8):
    """Collapse duplicates after prettifying; return top_k positives and negatives."""
    agg: dict[str, float] = {}
    for term, w in pairs:
        pt = _pretty_token(term)
        if not pt:
            continue
        agg[pt] = agg.get(pt, 0.0) + float(w)
    pos = sorted([(t, v) for t, v in agg.items() if v > 0], key=lambda x: x[1], reverse=True)[:top_k]
    neg = sorted([(t, v) for t, v in agg.items() if v < 0], key=lambda x: x[1])[:top_k]
    return pos, neg

def _ascii_bar(p: float, width: int = 20) -> str:
    """Compact ASCII probability bar."""
    p = max(0.0, min(1.0, float(p)))
    n = int(round(p * width))
    return "[" + "#" * n + "-" * (width - n) + f"] {p*100:5.1f}%"

def _print_probs_block(probs_dict: dict[str, float], title="Class probabilities"):
    print(title + ":")
    for cls, pv in sorted(probs_dict.items(), key=lambda t: t[1], reverse=True):
        print(f"  - {cls:>20}: {_ascii_bar(pv)}")
    print("")

def _print_contribs_block(top_pos, top_neg, title_pos="Top reasons (push UP)", title_neg="Top reasons (push DOWN)"):
    tot = sum(abs(v) for _, v in top_pos + top_neg) or 1e-9
    print(title_pos + ":")
    if not top_pos:
        print("  (none)")
    for term, val in top_pos:
        share = 100.0 * abs(val) / tot
        print(f"  + {term:<30} {val:+.4f}  ({share:4.1f}%)")
    print("")
    print(title_neg + ":")
    if not top_neg:
        print("  (none)")
    for term, val in top_neg:
        share = 100.0 * abs(val) / tot
        print(f"  - {term:<30} {val:+.4f}  ({share:4.1f}%)")
    print("")

def print_friendly_summary(
    raw_text: str,
    pred_label: str,
    pred_prob: float,
    probs_dict: dict[str, float],
    logit: float,
    bias: float,
    pairs: list[tuple[str, float]],
    expected_lr: float | None = None,
    ensemble_score: float | None = None,
):
    """Pretty, jury-friendly terminal summary."""
    print("\n" + "=" * 80)
    print("INPUT TEXT:")
    print(raw_text)
    print("-" * 80)

    # 1) Prediction
    print(f"Prediction: {pred_label}   |   Confidence: {_ascii_bar(pred_prob)}")
    if expected_lr is not None:
        print(f"Expected risk (LR-only): {expected_lr:.3f}")
    if ensemble_score is not None and isfinite(ensemble_score):
        print(f"Final ensemble score   : {ensemble_score:.3f}")
    print("")

    # 2) Class probabilities
    _print_probs_block(probs_dict)

    # 3) Short internals
    print(f"Model internals (for '{pred_label}'):  logit = {logit:.4f}   |   bias = {bias:.4f}\n")

    # 4) Reasons
    top_pos, top_neg = _collapse_contribs(pairs, top_k=8)
    _print_contribs_block(
        top_pos,
        top_neg,
        title_pos="Top words/phrases pushing UP (toward this class)",
        title_neg="Top words/phrases pushing DOWN",
    )

    # 5) Notes
    print("Notes:")
    print("  • “Confidence” is the model probability for the predicted class.")
    print("  • “Expected risk (LR-only)” is the TF-IDF+LogReg score mapped to [0..1].")
    print("  • “Final ensemble score” blends transformer regression with LR (if available).")
    print("=" * 80 + "\n")

def main():
    ap = argparse.ArgumentParser(description="Explain LR prediction + optional ensemble with transformer.")
    ap.add_argument("--model", type=str, required=True, help="Path to TF-IDF + LR pipeline (.joblib)")
    ap.add_argument("--text", type=str, default=None, help="Single text to score")
    ap.add_argument("--file", type=str, default=None, help="Path to a .txt with one sample per line")
    ap.add_argument("--as-one", action="store_true", help="Concatenate all lines into one text")
    ap.add_argument("--topk", type=int, default=10, help="Top K positive/negative contributors to display")
    ap.add_argument("--artifacts", type=str, default="artifacts_scams",
                    help="Dir containing transformer_reg/ and bucket_config.json for ensemble")
    ap.add_argument("--alpha", type=float, default=None,
                    help="Override ensemble alpha; otherwise use bucket_config.json (default 0.5).")
    ap.add_argument("--threshold", type=float, default=0.60,
                    help="Decision threshold on average final score for conversation summary.")
    args = ap.parse_args()

    pipe = joblib.load(args.model)

    # Optional transformer
    tok, mdl, device = load_transformer(args.artifacts)

    # ensemble alpha: din config sau suprascris din CLI
    cfg_alpha = 0.5
    try:
        with open(os.path.join(args.artifacts, "bucket_config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
            cfg_alpha = float(cfg.get("ensemble_alpha", 0.5))
    except Exception:
        pass
    alpha = args.alpha if args.alpha is not None else cfg_alpha

    def process_one(sample: str):
        pred_label, pred_prob, probs_dict, logit, bias, top_pos, top_neg, pairs = explain_logreg(
            pipe, sample, top_k=args.topk
        )
        classes = list(map(str, pipe.classes_))
        proba_vec = np.array([probs_dict[c] for c in classes], dtype=np.float32)
        exp_score = expected_score_from_probs(classes, proba_vec)

        # Try ensemble if transformer is available
        tr_score = None
        ensemble = None
        if tok is not None and mdl is not None:
            try:
                tr_score = transformer_score(tok, mdl, device, sample)
                ensemble = float(np.clip(alpha * tr_score + (1.0 - alpha) * exp_score, 0.0, 1.0))
            except Exception:
                tr_score, ensemble = None, None

        print_friendly_summary(
            raw_text=sample,
            pred_label=pred_label,
            pred_prob=pred_prob,
            probs_dict=probs_dict,
            logit=logit,
            bias=bias,
            pairs=pairs,
            expected_lr=exp_score,
            ensemble_score=ensemble,
        )

        final_score = ensemble if ensemble is not None else exp_score
        return {
            "text": sample,
            "lr_expected": exp_score,
            "tr_score": tr_score,
            "final_score": final_score,
        }

    # Inputs
    texts = []
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            raw = [ln.strip() for ln in f if ln.strip()]
        lines = [_clean_line(x) for x in raw]
        lines = [x for x in lines if x]  
        texts = [" ".join(lines)] if args.as_one else lines
    elif args.text:
        texts = [args.text]
    else:
        print("Provide --text or --file")
        sys.exit(1)

    # Per-line + colectare pentru summary
    results = []
    for t in texts:
        r = process_one(t)
        results.append(r)

    # === Conversation Summary ===
    thr = float(args.threshold)
    finals = [r["final_score"] if r["final_score"] is not None else r["lr_expected"] for r in results]
    avg = sum(finals) / len(finals) if finals else 0.0

    print("\n=== Conversation Summary ===")
    print(f" Messages scored: {len(results)}")
    print(f" Ensemble alpha (transformer weight): {alpha:.2f}")
    print(f" Critical warning threshold (average): {thr:.2f}\n")
    print(f" Average score:   {avg:.3f}")
    dec = "CRITICAL WARNING" if avg >= thr else "OK"
    sign = "≥" if avg >= thr else "<"
    print(f" Decision:        {dec} (avg {sign} {thr:.2f})")

    if avg >= thr:
        print(" Reason: average risk is above threshold; multiple messages scored high.")
    else:
        print(" Reason: average risk stayed below threshold at current alpha/threshold settings.")

    summary = {
        "average_score": avg,
        "critical_warning": avg >= thr,
        "threshold": thr,
        "alpha": alpha,
        "steps": [
            {
                "text": r["text"],
                "score": r["final_score"],
                "lr_expected_score": r["lr_expected"],
                "tr_score": r["tr_score"],
            }
            for r in results
        ],
    }
    print("\n[App-usable summary]")
    print(summary)

if __name__ == "__main__":
    main()
