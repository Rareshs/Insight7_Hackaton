import argparse, json, sys
from pathlib import Path
import joblib

def load_model(path: str):
    obj = joblib.load(path)
    # suportă fie Pipeline sklearn, fie imblearn Pipeline
    if hasattr(obj, "predict_proba") and hasattr(obj, "classes_"):
        return obj
    if hasattr(obj, "get") and "model" in obj:   # cazul când ai salvat dict cu 'model'
        return obj["model"]
    raise ValueError("Model necunoscut: nu găsesc predict_proba/classes_")

def predict_texts(model, texts):
    proba = model.predict_proba(texts)
    classes = list(model.classes_)
    preds = proba.argmax(axis=1)
    out = []
    for t, i, row in zip(texts, preds, proba):
        out.append({
            "text": t,
            "label": classes[i],
            "probs": {c: float(p) for c, p in zip(classes, row)}
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Calea către .joblib")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str, help="Un singur sir de intrare")
    g.add_argument("--stdin", action="store_true", help="Citește tot din STDIN")
    g.add_argument("--file", type=str, help="Fișier cu mai multe linii")
    ap.add_argument("--as-one", action="store_true", help="Dacă e --file, unește toate liniile într-un singur exemplu")
    ap.add_argument("--format", choices=["json","plain"], default="json", help="Format output")
    args = ap.parse_args()

    model = load_model(args.model)

    if args.text:
        texts = [args.text.strip()]
    elif args.stdin:
        texts = [sys.stdin.read().strip()]
    else:
        p = Path(args.file)
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        texts = [" ".join(lines)] if args.as_one else lines

    results = predict_texts(model, texts)

    if args.format == "json":
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for r in results:
            print(f"[{r['label']}] {r['text'][:120]} ...")
            print("probs:", ", ".join(f"{k}={v:.3f}" for k,v in r["probs"].items()))
            print("-" * 60)

if __name__ == "__main__":
    main()