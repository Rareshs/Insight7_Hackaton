import argparse
import pandas as pd
import joblib

from ml import clean_text, expected_score_from_proba, FOUR_CLASSES  # importi utilitare

def predict_texts(texts, model):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--as-one", action="store_true",
                        help="Concatenate all lines from --file into a single text")
    args = parser.parse_args()

    pipe = joblib.load(args.model)

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        texts = [" ".join(lines)] if args.as_one else lines
        predict_texts(texts, pipe)

    elif args.text:
        predict_texts(args.text, pipe)
    else:
        print("Please provide --text or --file")

if __name__ == "__main__":
    main()
