# xai_utils.py
# XAI helpers for your current pipeline: TF-IDF+LogReg and DistilRoBERTa regression.

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional

# --- Optional deps (soft-fail) ---
_ELI5 = None
_CAPTUM = None
_TORCH = None

def _lazy_imports():
    global _ELI5, _CAPTUM, _TORCH
    if _ELI5 is None:
        try:
            import eli5
            from eli5.sklearn import explain_prediction_linear_classifier as _exp_lin
            _ELI5 = (eli5, _exp_lin)
        except Exception:
            _ELI5 = None
    if _CAPTUM is None or _TORCH is None:
        try:
            import torch
            from captum.attr import IntegratedGradients
            _CAPTUM = IntegratedGradients
            _TORCH = torch
        except Exception:
            _CAPTUM = None
            _TORCH = None

# -----------------------------------------
#  A) EXPLAIN TF-IDF + LogisticRegression
# -----------------------------------------
def explain_logreg_text(
    pipeline,              # your fitted ImbPipeline/Pipeline [features -> clf]
    text: str,
    top_k: int = 15
) -> Dict[str, Any]:
    """
    Returns per-token contributions for the predicted class using ELI5.
    Works best for linear models over sparse TF-IDF.
    """
    _lazy_imports()
    if _ELI5 is None:
        return {
            "ok": False,
            "why": "eli5 not installed (pip install eli5).",
            "pred_text": text
        }

    eli5, exp_linear = _ELI5
    clf = pipeline.named_steps.get("clf", None)
    feats = pipeline.named_steps.get("features", None)
    if clf is None or feats is None:
        return {"ok": False, "why": "pipeline must have 'features' and 'clf' steps"}

    # ELI5 explanation object
    try:
        expl = exp_linear(clf, doc=text, vec=feats,
                          top=top_k, target_names=list(getattr(clf, "classes_", [])))
    except Exception as e:
        return {"ok": False, "why": f"eli5 explain failed: {e}"}

    # parse weights into a compact structure
    parsed = []
    for sec in expl.targets:
        cls_name = sec.target
        weights = []
        for w in sec.feature_weights.pos + sec.feature_weights.neg:
            weights.append({
                "feature": w.feature,      # token/ngram
                "weight": float(w.weight), # signed contribution
            })
        parsed.append({"class": cls_name, "weights": weights[:top_k]})

    # predicted class/probabilities
    proba = pipeline.predict_proba([text])[0]
    classes = list(getattr(clf, "classes_", []))
    pred_idx = int(np.argmax(proba))
    return {
        "ok": True,
        "type": "logreg_eli5",
        "text": text,
        "pred_class": classes[pred_idx] if classes else None,
        "proba": {c: float(p) for c, p in zip(classes, proba)},
        "per_class_top_features": parsed
    }

# -------------------------------------------------
#  B) EXPLAIN DistilRoBERTa regression (Captum IG)
# -------------------------------------------------
def explain_transformer_text(
    model, tokenizer, text: str, max_length: int = 256, n_steps: int = 64
) -> Dict[str, Any]:
    """
    Token-level attributions via Integrated Gradients for a regression head (num_labels=1).
    Requires torch + captum. Returns tokens + attribution scores (normalized).
    """
    _lazy_imports()
    if _CAPTUM is None or _TORCH is None:
        return {"ok": False, "why": "captum/torch not available (pip install captum torch)."}

    torch = _TORCH
    IntegratedGradients = _CAPTUM

    device = next(model.parameters()).device
    enc = tokenizer(
        text,
        truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    def forward_func(input_ids, attention_mask):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        # regression → single logit
        return out.logits.squeeze(-1)

    ig = IntegratedGradients(lambda ids: forward_func(ids, enc.get("attention_mask")))
    # baselines: all [PAD] (same shape), captum expects same device
    baseline_ids = torch.full_like(enc["input_ids"], fill_value=tokenizer.pad_token_id)

    attributions, _ = ig.attribute(
        inputs=enc["input_ids"],
        baselines=baseline_ids,
        n_steps=n_steps,
        return_convergence_delta=True
    )

    # aggregate subword attributions to tokens
    atts = attributions.detach().cpu().numpy()[0]  # (seq_len,)
    input_ids = enc["input_ids"][0].detach().cpu().numpy().tolist()
    toks = tokenizer.convert_ids_to_tokens(input_ids)

    # simple normalization for readability
    att_min, att_max = float(np.min(atts)), float(np.max(atts))
    denom = (att_max - att_min) or 1.0
    norm = ((atts - att_min) / denom).tolist()

    # pair tokens with scores and strip special tokens
    out = []
    for t, s in zip(toks, norm):
        if t in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token):
            continue
        out.append({"token": t, "importance": float(s)})

    with torch.no_grad():
        pred = forward_func(enc["input_ids"], enc.get("attention_mask")).detach().cpu().numpy().reshape(-1)[0]
        pred = float(np.clip(pred, 0.0, 1.0))

    return {
        "ok": True,
        "type": "transformer_captum_ig",
        "text": text,
        "predicted_score": pred,
        "token_importances": out
    }

# -------------------------------------------------
#  C) Convenience wrapper for entire conversation
# -------------------------------------------------
def explain_conversation(
    convo_texts: List[str],
    lr_pipeline = None,
    tr_model = None,
    tr_tokenizer = None,
    which: str = "auto",     # "lr" | "transformer" | "auto"
    top_k_lr: int = 10
) -> Dict[str, Any]:
    """
    Returns explanations per message. If which='auto':
      - use LR (fast) if available; else transformer if provided.
    """
    results = []
    for msg in convo_texts:
        if which == "lr" or (which == "auto" and lr_pipeline is not None):
            results.append(explain_logreg_text(lr_pipeline, msg, top_k=top_k_lr))
        elif which == "transformer" or (which == "auto" and tr_model is not None and tr_tokenizer is not None):
            results.append(explain_transformer_text(tr_model, tr_tokenizer, msg))
        else:
            results.append({"ok": False, "why": "no explainer available for this message", "text": msg})
    return {"items": results}
