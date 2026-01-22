import argparse
import csv
import json
import logging
import math
import os
import re
from typing import Dict, List, Tuple


# ----------------------------
# Logging
# ----------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


# ----------------------------
# Stopwords (English) - LOCAL, no Spark/JVM
# (listă standard comună; suficient pentru demo/predict)
# ----------------------------
EN_STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at",
    "be","because","been","before","being","below","between","both","but","by",
    "can","can't","cannot","could","couldn't",
    "did","didn't","do","does","doesn't","doing","don't","down","during",
    "each",
    "few","for","from","further",
    "had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's",
    "hers","herself","him","himself","his","how","how's",
    "i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself",
    "let's",
    "me","more","most","mustn't","my","myself",
    "no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own",
    "same","she","she'd","she'll","she's","should","shouldn't","so","some","such",
    "than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd",
    "they'll","they're","they've","this","those","through","to","too",
    "under","until","up",
    "very",
    "was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where",
    "where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't",
    "you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"
}


# ----------------------------
# Helpers
# ----------------------------
def load_json(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_vocab_list(vocab_obj):
    if isinstance(vocab_obj, list):
        return vocab_obj
    if isinstance(vocab_obj, dict):
        sample_key = next(iter(vocab_obj.keys()))
        sample_val = vocab_obj[sample_key]

        if isinstance(sample_val, int):
            return [w for w, _ in sorted(vocab_obj.items(), key=lambda x: x[1])]

        if isinstance(sample_key, str) and sample_key.isdigit():
            return [w for _, w in sorted(vocab_obj.items(), key=lambda x: int(x[0]))]

    raise ValueError("cv_vocabulary.json must be list[str] or dict (word->idx).")


def stable_sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


# ----------------------------
# Text processing: RegexTokenizer(pattern=r"\W+", minTokenLength=2)
# ----------------------------
TOKEN_SPLIT_RE = re.compile(r"\W+", flags=re.UNICODE)

def tokenize(text: str) -> List[str]:
    if text is None:
        return []
    text = str(text).lower().strip()
    parts = TOKEN_SPLIT_RE.split(text)
    return [p for p in parts if len(p) >= 2]


def remove_stopwords(tokens: List[str], stopset: set) -> List[str]:
    return [t for t in tokens if t not in stopset]


# ----------------------------
# Vectorization + IDF
# ----------------------------
def build_vocab_index(vocab_list: List[str]) -> Dict[str, int]:
    return {w: i for i, w in enumerate(vocab_list)}


def featurize(tokens: List[str],
              vocab_index: Dict[str, int],
              idf: List[float],
              binary_tf: bool) -> Dict[int, float]:
    counts: Dict[int, float] = {}

    for t in tokens:
        idx = vocab_index.get(t)
        if idx is None:
            continue
        if binary_tf:
            counts[idx] = 1.0
        else:
            counts[idx] = counts.get(idx, 0.0) + 1.0

    feats: Dict[int, float] = {}
    for i, tfv in counts.items():
        feats[i] = float(tfv) * float(idf[i])
    return feats


def dot_sparse(feats: Dict[int, float], weights: List[float]) -> float:
    s = 0.0
    for i, v in feats.items():
        s += float(v) * float(weights[i])
    return s


# ----------------------------
# Predict
# ----------------------------
def predict_logreg(feats: Dict[int, float], coefs: List[float], intercept: float) -> Tuple[float, float]:
    z = float(intercept) + dot_sparse(feats, coefs)
    p = stable_sigmoid(z)
    pred = 1.0 if p >= 0.5 else 0.0
    return p, pred


def predict_linearsvc(feats: Dict[int, float], coefs: List[float], intercept: float) -> Tuple[float, float]:
    margin = float(intercept) + dot_sparse(feats, coefs)
    pred = 1.0 if margin >= 0.0 else 0.0
    return margin, pred


# ----------------------------
# CSV IO
# ----------------------------
def read_input_rows(input_csv: str, text_col: str) -> List[str]:
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    texts: List[str] = []
    with open(input_csv, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or text_col not in reader.fieldnames:
            raise ValueError(f"CSV missing column '{text_col}'. Columns: {reader.fieldnames}")
        for row in reader:
            txt = row.get(text_col)
            if txt is None:
                continue
            txt = str(txt).strip()
            if txt:
                texts.append(txt)
    return texts


def write_output_csv(out_path: str, rows: List[dict], fieldnames: List[str]):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logreg", "linearsvc"], required=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--input_csv", type=str, default=None)
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--reports_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default=None)

    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger("predict_local")

    artifacts_dir = os.path.join(args.reports_dir, "feature_artifacts")
    vocab_path = os.path.join(artifacts_dir, "cv_vocabulary.json")
    idf_path = os.path.join(artifacts_dir, "idf_values.json")
    meta_path = os.path.join(artifacts_dir, "features_meta.json")

    weights_dir = os.path.join(args.reports_dir, "model_weights")
    weights_path = os.path.join(weights_dir, f"{args.model}_weights.json")

    vocab = ensure_vocab_list(load_json(vocab_path))
    idf_list = list(map(float, load_json(idf_path)))
    meta = load_json(meta_path)
    weights = load_json(weights_path)

    binary_tf = bool(meta.get("binary_tf", False))

    coefs = weights.get("coefficients")
    intercept = weights.get("intercept")
    if coefs is None or intercept is None:
        raise ValueError(f"Bad weights file: {weights_path} (missing coefficients/intercept)")

    coefs = list(map(float, coefs))
    intercept = float(intercept)

    if len(vocab) != len(idf_list):
        raise ValueError(f"Mismatch vocab vs idf: vocab={len(vocab)} idf={len(idf_list)}")
    if len(coefs) != len(vocab):
        raise ValueError(f"Mismatch vocab vs coefficients: vocab={len(vocab)} coefs={len(coefs)}")

    logger.info(f"Loaded artifacts: vocab={len(vocab)} idf={len(idf_list)} binary_tf={binary_tf}")
    logger.info(f"Loaded weights: {weights.get('type')} from {weights_path}")

    vocab_index = build_vocab_index(vocab)

    if args.input_csv:
        texts = read_input_rows(args.input_csv, args.text_col)
    else:
        if not args.text:
            raise ValueError("Provide --text or --input_csv")
        texts = [args.text]

    logger.info(f"Predicting on {len(texts)} row(s)")

    out_rows: List[dict] = []
    for t in texts:
        tokens = tokenize(t)
        tokens = remove_stopwords(tokens, EN_STOPWORDS)

        feats = featurize(tokens, vocab_index, idf_list, binary_tf)

        if args.model == "logreg":
            prob, pred = predict_logreg(feats, coefs, intercept)
            out_rows.append({"text": t, "probability": prob, "prediction": pred})
        else:
            margin, pred = predict_linearsvc(feats, coefs, intercept)
            out_rows.append({"text": t, "margin": margin, "prediction": pred})

    # preview
    show_n = min(20, len(out_rows))
    for i in range(show_n):
        r = out_rows[i]
        if args.model == "logreg":
            logger.info(f"[{i}] pred={r['prediction']} prob={r['probability']:.4f} text={r['text'][:120]!r}")
        else:
            logger.info(f"[{i}] pred={r['prediction']} margin={r['margin']:.4f} text={r['text'][:120]!r}")

    if args.out_csv:
        if args.model == "logreg":
            fields = ["text", "probability", "prediction"]
        else:
            fields = ["text", "margin", "prediction"]
        write_output_csv(args.out_csv, out_rows, fields)
        logger.info(f"Saved predictions to: {args.out_csv}")


if __name__ == "__main__":
    main()
