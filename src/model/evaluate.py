import os
import json
from typing import Dict, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def _confusion_counts(pred_df: DataFrame, label_col: str, pred_col: str) -> Dict[str, int]:
    # TP: label=1 pred=1, TN: label=0 pred=0, FP: label=0 pred=1, FN: label=1 pred=0
    df = pred_df.select(
        F.col(label_col).cast("int").alias("label"),
        F.col(pred_col).cast("int").alias("pred")
    )

    tp = df.filter((F.col("label") == 1) & (F.col("pred") == 1)).count()
    tn = df.filter((F.col("label") == 0) & (F.col("pred") == 0)).count()
    fp = df.filter((F.col("label") == 0) & (F.col("pred") == 1)).count()
    fn = df.filter((F.col("label") == 1) & (F.col("pred") == 0)).count()

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compute_metrics(pred_df: DataFrame, *, label_col: str = "label", pred_col: str = "prediction") -> Dict:
    """
    Metrici manual (fără seaborn etc):
      Accuracy, Precision, Recall, F1 + confusion matrix
    """
    cm = _confusion_counts(pred_df, label_col, pred_col)

    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "total": total
    }


def save_metrics_json(metrics: Dict, out_dir: str, filename: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return out_path


def top_errors(
    pred_df: DataFrame,
    *,
    label_col: str = "label",
    pred_col: str = "prediction",
    text_col: str = "text",
    k: int = 5
) -> Tuple[DataFrame, DataFrame]:
    """
    Returnează top FP și top FN (sample) pentru analiză erori.
    """
    df = pred_df.select("title", text_col, F.col(label_col).alias("label"), F.col(pred_col).alias("pred"))

    fp = df.filter((F.col("label") == 0) & (F.col("pred") == 1)).limit(k)
    fn = df.filter((F.col("label") == 1) & (F.col("pred") == 0)).limit(k)

    return fp, fn
