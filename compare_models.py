import argparse
import json
import os
import sys
import time
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression, LinearSVC, RandomForestClassifier, GBTClassifier


# ---------------------------
# Spark + IO
# ---------------------------

def create_spark(app_name: str, shuffle_partitions: int = 16, driver_memory: str = "4g") -> SparkSession:
    # Windows-safe: force Spark to use current Python executable (avoids "Python was not found" MS Store alias)
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.memory", driver_memory)
        .getOrCreate()
    )


def load_true_fake(spark: SparkSession, true_csv: str, fake_csv: str, text_col: str = "text"):
    df_true = spark.read.option("header", True).csv(true_csv)
    df_fake = spark.read.option("header", True).csv(fake_csv)

    df_true = df_true.withColumn("label", F.lit(1.0))
    df_fake = df_fake.withColumn("label", F.lit(0.0))

    # normalize to internal columns
    df_true = df_true.select(F.col(text_col).alias("text"), F.col("label"))
    df_fake = df_fake.select(F.col(text_col).alias("text"), F.col("label"))

    df = df_true.unionByName(df_fake)

    # drop invalid text early
    df = df.na.drop(subset=["text", "label"])
    df = df.filter(F.length(F.trim(F.col("text"))) > 0)

    return df


# ---------------------------
# Features
# ---------------------------

def build_feature_pipeline(vocab_size: int = 20000, min_df: int = 2):
    tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern=r"\W+", minTokenLength=2)
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    cv = CountVectorizer(
        inputCol="filtered_tokens",
        outputCol="tf",
        vocabSize=vocab_size,
        minDF=min_df
    )
    idf = IDF(inputCol="tf", outputCol="features")
    return Pipeline(stages=[tokenizer, remover, cv, idf])


# ---------------------------
# Evaluation (NO RDD / NO MulticlassMetrics)
# ---------------------------

def evaluate_predictions(pred_df):
    p = (
        pred_df
        .select(
            F.col("label").cast("int").alias("label"),
            F.col("prediction").cast("int").alias("prediction")
        )
        .na.drop()
    )

    agg = p.agg(
        F.sum(F.when((F.col("label") == 1) & (F.col("prediction") == 1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("label") == 0) & (F.col("prediction") == 0), 1).otherwise(0)).alias("tn"),
        F.sum(F.when((F.col("label") == 0) & (F.col("prediction") == 1), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("label") == 1) & (F.col("prediction") == 0), 1).otherwise(0)).alias("fn"),
    ).collect()[0]

    tp = int(agg["tp"])
    tn = int(agg["tn"])
    fp = int(agg["fp"])
    fn = int(agg["fn"])

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    cm = [[tn, fp], [fn, tp]]  # [[tn, fp], [fn, tp]]

    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": cm,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ---------------------------
# Reporting
# ---------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_reports(out_dir: str, results: list):
    ensure_dir(out_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"model_comparison_{ts}.json")
    csv_path = os.path.join(out_dir, f"model_comparison_{ts}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    header = "model,features_fit_seconds,train_seconds,score_seconds,accuracy,f1,precision,recall,tp,tn,fp,fn\n"
    lines = [header]
    for r in results:
        m = r["metrics"]
        lines.append(
            f"{r['model']},{r['features_fit_seconds']:.3f},{r['train_seconds']:.3f},{r['score_seconds']:.3f},"
            f"{m['accuracy']:.6f},{m['f1']:.6f},{m['precision']:.6f},{m['recall']:.6f},"
            f"{m['tp']},{m['tn']},{m['fp']},{m['fn']}\n"
        )

    with open(csv_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return json_path, csv_path


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true_csv", required=True)
    ap.add_argument("--fake_csv", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--out_dir", default="outputs/reports")
    ap.add_argument("--mode", choices=["debug", "full"], default="full")
    ap.add_argument("--seed", type=int, default=42)

    # Spark params
    ap.add_argument("--shuffle_partitions", type=int, default=16)
    ap.add_argument("--driver_memory", default="4g")

    # Feature params
    ap.add_argument("--vocab_size", type=int, default=20000)
    ap.add_argument("--min_df", type=int, default=2)

    args = ap.parse_args()

    spark = create_spark(
        "FakeNewsModelComparison",
        shuffle_partitions=args.shuffle_partitions,
        driver_memory=args.driver_memory,
    )
    spark.sparkContext.setLogLevel("WARN")

    df = load_true_fake(spark, args.true_csv, args.fake_csv, text_col=args.text_col)

    # debug mode: random sample (NOT limit())
    if args.mode == "debug":
        df = df.orderBy(F.rand(args.seed)).limit(5000)

    # cache base df
    df = df.repartition(8).cache()
    _ = df.count()

    # quick label distribution check
    dist = df.groupBy("label").count().orderBy("label").collect()
    print("\nLabel distribution:")
    for row in dist:
        print(f"  label={row['label']}: {row['count']}")

    if len(dist) < 2:
        print("\nERROR: Only one class present after sampling/cleaning. "
              "Use mode=full or change debug sampling.")
        spark.stop()
        return

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=args.seed)
    train_df = train_df.cache()
    test_df = test_df.cache()
    _ = train_df.count()
    _ = test_df.count()

    # Fit feature pipeline ONCE
    feat_pipe = build_feature_pipeline(vocab_size=args.vocab_size, min_df=args.min_df)

    t0 = time.time()
    feat_model = feat_pipe.fit(train_df)
    feat_fit_sec = time.time() - t0

    train_feat = feat_model.transform(train_df).select("features", "label").cache()
    test_feat = feat_model.transform(test_df).select("features", "label").cache()
    _ = train_feat.count()
    _ = test_feat.count()

    # Models
    models = [
        ("logreg", LogisticRegression(featuresCol="features", labelCol="label", maxIter=50, regParam=0.0)),
        ("linearsvc", LinearSVC(featuresCol="features", labelCol="label", maxIter=50, regParam=0.0)),
        ("rf", RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50, maxDepth=10)),
        ("gbt", GBTClassifier(featuresCol="features", labelCol="label", maxIter=30, maxDepth=5)),
    ]

    results = []
    print("\n=== Feature pipeline fit time (seconds) ===")
    print(f"features_fit_seconds = {feat_fit_sec:.3f}\n")

    print("=== Model comparison ===")
    print("model     train_s   score_s   accuracy   f1        precision  recall")
    print("---------------------------------------------------------------------")

    for name, estimator in models:
        t1 = time.time()
        model = estimator.fit(train_feat)
        train_sec = time.time() - t1

        t2 = time.time()
        preds = model.transform(test_feat)
        # Force materialization to measure scoring properly
        _ = preds.select("prediction").count()
        score_sec = time.time() - t2

        metrics = evaluate_predictions(preds)

        results.append({
            "model": name,
            "features_fit_seconds": float(feat_fit_sec),
            "train_seconds": float(train_sec),
            "score_seconds": float(score_sec),
            "metrics": metrics,
        })

        print(
            f"{name:<8} {train_sec:>8.3f} {score_sec:>8.3f} "
            f"{metrics['accuracy']:>10.4f} {metrics['f1']:>9.4f} "
            f"{metrics['precision']:>10.4f} {metrics['recall']:>7.4f}"
        )

    json_path, csv_path = save_reports(args.out_dir, results)
    print("\nSaved comparison reports:")
    print(json_path)
    print(csv_path)

    spark.stop()


if __name__ == "__main__":
    main()
