import argparse
import json
import os
import sys
import time
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF
from pyspark.ml.classification import LogisticRegression, LinearSVC, RandomForestClassifier, GBTClassifier


def create_spark(app_name: str,
                 shuffle_partitions: int = 2,
                 driver_memory: str = "4g",
                 local_dir: str | None = None) -> SparkSession:
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    b = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.memory", driver_memory)

        # stability
        .config("spark.network.timeout", "800s")
        .config("spark.executor.heartbeatInterval", "60s")
        .config("spark.python.worker.reuse", "true")

        # reduce shuffles
        .config("spark.sql.adaptive.enabled", "false")

        # reduce memory pressure
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.shuffle.compress", "true")
        .config("spark.shuffle.spill.compress", "true")
        .config("spark.rdd.compress", "true")
        .config("spark.ui.enabled", "false")
    )
    if local_dir:
        b = b.config("spark.local.dir", local_dir)

    return b.getOrCreate()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def now_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_true_fake(spark: SparkSession, true_csv: str, fake_csv: str, text_col: str):
    df_true = spark.read.option("header", True).csv(true_csv)
    df_fake = spark.read.option("header", True).csv(fake_csv)

    if text_col not in df_true.columns:
        raise ValueError(f"[True.csv] missing '{text_col}'. Columns: {df_true.columns}")
    if text_col not in df_fake.columns:
        raise ValueError(f"[Fake.csv] missing '{text_col}'. Columns: {df_fake.columns}")

    df_true = df_true.select(F.col(text_col).alias("text")).withColumn("label", F.lit(1.0))
    df_fake = df_fake.select(F.col(text_col).alias("text")).withColumn("label", F.lit(0.0))
    df = df_true.unionByName(df_fake)

    df = df.na.drop(subset=["text", "label"])
    df = df.filter(F.length(F.trim(F.col("text"))) > 0)
    return df


def clip_text(df, max_chars: int):
    # taie textul ca să nu-ți umfle tokenizarea/memoria
    return df.withColumn("text", F.substring(F.col("text"), 1, max_chars))


def build_pipeline(model_name: str, num_features: int):
    tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern=r"\W+", minTokenLength=2)
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")

    # FARA IDF (stabil pe Windows)
    tf = HashingTF(inputCol="filtered_tokens", outputCol="features", numFeatures=num_features)

    if model_name == "logreg":
        clf = LogisticRegression(featuresCol="features", labelCol="label", maxIter=30, regParam=0.0)
    elif model_name == "linearsvc":
        clf = LinearSVC(featuresCol="features", labelCol="label", maxIter=30, regParam=0.0)
    elif model_name == "rf":
        clf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=10, maxDepth=6)
    elif model_name == "gbt":
        clf = GBTClassifier(featuresCol="features", labelCol="label", maxIter=10, maxDepth=4)
    else:
        raise ValueError("model must be one of: logreg, linearsvc, rf, gbt")

    return Pipeline(stages=[tokenizer, remover, tf, clf])


def evaluate_predictions(pred_df):
    p = pred_df.select(
        F.col("label").cast("int").alias("label"),
        F.col("prediction").cast("int").alias("prediction")
    ).na.drop()

    agg = p.agg(
        F.sum(F.when((F.col("label") == 1) & (F.col("prediction") == 1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("label") == 0) & (F.col("prediction") == 0), 1).otherwise(0)).alias("tn"),
        F.sum(F.when((F.col("label") == 0) & (F.col("prediction") == 1), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("label") == 1) & (F.col("prediction") == 0), 1).otherwise(0)).alias("fn"),
    ).collect()[0]

    tp = int(agg["tp"]); tn = int(agg["tn"]); fp = int(agg["fp"]); fn = int(agg["fn"])
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def save_report(out_dir: str, payload: dict):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{payload['model']}_{now_ts()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true_csv", required=True)
    ap.add_argument("--fake_csv", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--model", choices=["logreg", "linearsvc", "rf", "gbt"], required=True)

    ap.add_argument("--mode", choices=["debug", "full"], default="debug")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--driver_memory", default="4g")
    ap.add_argument("--shuffle_partitions", type=int, default=2)
    ap.add_argument("--spark_local_dir", default=None)

    # anti-crash knobs
    ap.add_argument("--debug_rows", type=int, default=2000)
    ap.add_argument("--max_chars", type=int, default=2000)
    ap.add_argument("--num_features", type=int, default=1 << 16)  # 65536 (mai mic = mai stabil)

    ap.add_argument("--out_dir", default="outputs/reports")

    args = ap.parse_args()

    spark = create_spark(
        "FakeNewsRunNoCrash",
        shuffle_partitions=args.shuffle_partitions,
        driver_memory=args.driver_memory,
        local_dir=args.spark_local_dir
    )
    spark.sparkContext.setLogLevel("WARN")

    print("\n[1/5] load")
    df = load_true_fake(spark, args.true_csv, args.fake_csv, args.text_col)
    df = clip_text(df, args.max_chars)

    if args.mode == "debug":
        df = df.orderBy(F.rand(args.seed)).limit(args.debug_rows)

    # keep partitions tiny
    df = df.repartition(2)

    print("[2/5] split")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=args.seed)

    print("[3/5] build pipeline")
    pipe = build_pipeline(args.model, num_features=args.num_features)

    print("[4/5] train")
    t1 = time.time()
    model = pipe.fit(train_df)
    train_sec = time.time() - t1
    print(f"  train_seconds={train_sec:.3f}")

    print("[5/5] score + eval")
    t2 = time.time()
    preds = model.transform(test_df).select("label", "prediction")
    # force one tiny action
    preds.limit(1).collect()
    score_sec = time.time() - t2
    print(f"  score_seconds={score_sec:.3f}")

    metrics = evaluate_predictions(preds)
    print(f"  accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}")
    print(f"  confusion_matrix={metrics['confusion_matrix']}")

    payload = {
        "model": args.model,
        "mode": args.mode,
        "driver_memory": args.driver_memory,
        "shuffle_partitions": args.shuffle_partitions,
        "debug_rows": args.debug_rows if args.mode == "debug" else None,
        "max_chars": args.max_chars,
        "num_features": args.num_features,
        "train_seconds": float(train_sec),
        "score_seconds": float(score_sec),
        "metrics": metrics,
    }
    report_path = save_report(args.out_dir, payload)
    print("\nSaved report:", report_path)

    spark.stop()


if __name__ == "__main__":
    main()
