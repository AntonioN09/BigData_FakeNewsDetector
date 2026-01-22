import argparse
import json
import logging
import os
import time

from pyspark import StorageLevel
from pyspark.sql import functions as F
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier

from src.utils.logging import setup_logging
from src.utils.spark import create_spark_session
from src.utils.timing import Timer
from src.utils.save_safe import export_feature_artifacts

from src.data.load import load_dataset1
from src.data.clean import clean_text_df
from src.features.text_features import build_text_feature_pipeline
from src.model.evaluate import compute_metrics, save_metrics_json


# 🔴 HARD-CODE PATHS (same style as run_pipeline.py)
RAW_DATA_DIR = r"C:\Users\carin\OneDrive\Desktop\ibd\fake-news-detection-spark\data\raw\News_dataset"
REPORTS_DIR = r"C:\Users\carin\OneDrive\Desktop\ibd\fake-news-detection-spark\outputs\reports"

def _write_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path


def export_tree_artifacts(tree_model, out_dir: str, model_name: str) -> str:
    """
    Tree models don't have linear "weights".
    Export:
      - featureImportances
      - only params that are actually set (no KeyError)
    """
    os.makedirs(out_dir, exist_ok=True)

    fi = tree_model.featureImportances.toArray().tolist()

    # only params that are set => safe
    try:
        pmap = tree_model.extractParamMap()
        params_safe = {p.name: str(v) for p, v in pmap.items()}
    except Exception:
        params_safe = {}

    payload = {
        "model": model_name,
        "spark_type": tree_model.__class__.__name__,
        "num_features": int(len(fi)),
        "feature_importances": [float(x) for x in fi],
        "params": params_safe,
    }

    out_path = os.path.join(out_dir, f"{model_name}_artifacts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["debug", "demo", "full"], default="demo")
    parser.add_argument("--limit_rows", type=int, default=None)          # ex: 8000
    parser.add_argument("--sample_fraction", type=float, default=None)   # ex: 0.15
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("trees_demo")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ---- tuned params by mode ----
    if args.mode == "debug":
        shuffle_partitions = 16
        npart_clean = 16
        sample_fraction = 0.10
        vocab_size = 12000
        min_df = 5
        rf_num_trees = 60
        rf_max_depth = 10
        gbt_max_iter = 35
        gbt_max_depth = 6
    elif args.mode == "full":
        shuffle_partitions = 64
        npart_clean = 64
        sample_fraction = None
        vocab_size = 20000
        min_df = 10
        rf_num_trees = 120
        rf_max_depth = 12
        gbt_max_iter = 60
        gbt_max_depth = 7
    else:  # demo
        shuffle_partitions = 16
        npart_clean = 16
        sample_fraction = 0.20
        vocab_size = 12000
        min_df = 5
        rf_num_trees = 50
        rf_max_depth = 9
        gbt_max_iter = 35
        gbt_max_depth = 6

    # override
    if args.sample_fraction is not None:
        sample_fraction = args.sample_fraction

    times = {}

    # SPARK (same pattern as run_pipeline.py)
    with Timer("Spark session init"):
        spark = create_spark_session({
            "spark": {
                "master": "local[*]",
                "app_name": "FakeNewsTreesDemo",
                "driver_memory": "4g",
                "executor_memory": "4g",
                "shuffle_partitions": shuffle_partitions,
                "configs": {
                    "spark.sql.adaptive.enabled": "true",
                    "spark.sql.adaptive.coalescePartitions.enabled": "true",
                    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
                    # optional if C: small:
                    # "spark.local.dir": r"D:\spark_tmp",
                }
            }
        })

    # LOAD
    t0 = time.time()
    with Timer("Load dataset"):
        df, meta = load_dataset1(spark, RAW_DATA_DIR)
    times["load_sec"] = round(time.time() - t0, 2)
    logger.info(f"Meta: {meta}")
    logger.info(f"Columns: {df.columns}")

    # CLEAN
    t0 = time.time()
    with Timer("Clean dataset"):
        df_clean, clean_meta = clean_text_df(df, min_text_length=30, dedup=True, cache=False)
    times["clean_sec"] = round(time.time() - t0, 2)
    logger.info(f"Clean meta: {clean_meta}")

    # SUBSET (for demo stability)
    if args.limit_rows is not None:
        df_clean = df_clean.limit(args.limit_rows)
        logger.info(f"LIMIT: using first {args.limit_rows} rows after clean.")
    elif sample_fraction is not None:
        df_clean = df_clean.sample(withReplacement=False, fraction=sample_fraction, seed=42)
        logger.info(f"SAMPLE: using {int(sample_fraction*100)}% after clean.")

    # persist key dataset
    df_clean = df_clean.repartition(npart_clean).persist(StorageLevel.MEMORY_AND_DISK)
    _ = df_clean.count()
    logger.info(f"df_clean partitions: {df_clean.rdd.getNumPartitions()}")

    # FEATURES
    t0 = time.time()
    with Timer("Build features (CountVectorizer + IDF)"):
        feat_pipe, feat_meta = build_text_feature_pipeline(
            input_col="text",
            output_col="features",
            vocab_size=vocab_size,
            min_df=min_df
        )
        feat_model = feat_pipe.fit(df_clean)
        df_feat = feat_model.transform(df_clean).select("label", "features", "text")
    df_feat = df_feat.persist(StorageLevel.MEMORY_AND_DISK)
    _ = df_feat.count()
    times["features_sec"] = round(time.time() - t0, 2)
    logger.info(f"Feature meta: {feat_meta}")

    # save feature artifacts (same as run_pipeline.py)
    feature_art_dir = os.path.join(REPORTS_DIR, "feature_artifacts_trees")
    art_paths = export_feature_artifacts(feat_model, feature_art_dir)
    logger.info(f"Saved feature artifacts (safe) to: {art_paths}")

    # SPLIT
    train_df, test_df = df_feat.randomSplit([0.8, 0.2], seed=42)
    train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
    test_df = test_df.persist(StorageLevel.MEMORY_AND_DISK)
    _ = train_df.count()
    _ = test_df.count()

    # TRAIN RF
    logger.info("Training RandomForest...")
    t0 = time.time()
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=rf_num_trees,
        maxDepth=rf_max_depth,
        seed=42,
        featureSubsetStrategy="auto",
    )
    rf_model = rf.fit(train_df)
    rf_train_sec = round(time.time() - t0, 2)

    # SCORE RF
    t0 = time.time()
    rf_pred = rf_model.transform(test_df)
    _ = rf_pred.select("prediction").count()
    rf_score_sec = round(time.time() - t0, 2)

    rf_metrics = compute_metrics(rf_pred, label_col="label", pred_col="prediction")
    rf_metrics["train_seconds"] = rf_train_sec
    rf_metrics["score_seconds"] = rf_score_sec
    logger.info(f"[randomforest] metrics: {rf_metrics}")
    save_metrics_json(rf_metrics, REPORTS_DIR, "metrics_randomforest.json")
    export_tree_artifacts(rf_model, os.path.join(REPORTS_DIR, "tree_artifacts"), "randomforest")

    # TRAIN GBT
    logger.info("Training GBT...")
    t0 = time.time()
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        maxIter=gbt_max_iter,
        maxDepth=gbt_max_depth,
        seed=42,
    )
    gbt_model = gbt.fit(train_df)
    gbt_train_sec = round(time.time() - t0, 2)

    # SCORE GBT
    t0 = time.time()
    gbt_pred = gbt_model.transform(test_df)
    _ = gbt_pred.select("prediction").count()
    gbt_score_sec = round(time.time() - t0, 2)

    gbt_metrics = compute_metrics(gbt_pred, label_col="label", pred_col="prediction")
    gbt_metrics["train_seconds"] = gbt_train_sec
    gbt_metrics["score_seconds"] = gbt_score_sec
    logger.info(f"[gbt] metrics: {gbt_metrics}")
    save_metrics_json(gbt_metrics, REPORTS_DIR, "metrics_gbt.json")
    export_tree_artifacts(gbt_model, os.path.join(REPORTS_DIR, "tree_artifacts"), "gbt")

    # OPTIONAL: save predictions on test split (CSV) for demo (NO HADOOP_HOME issue if you avoid Spark writer)
    # We'll do a lightweight JSONL with Python to avoid Hadoop FS writes.
    out_pred_dir = os.path.join(REPORTS_DIR, "trees_predictions")
    os.makedirs(out_pred_dir, exist_ok=True)

    # Keep it small (top N) to avoid huge driver collect
    N = 200
    rf_rows = (
        rf_pred.select("text", "label", F.col("prediction").cast("int").alias("prediction"))
        .limit(N)
        .toPandas()
        .to_dict(orient="records")
    )
    _write_json(os.path.join(out_pred_dir, "randomforest_pred_sample.json"), rf_rows)

    gbt_rows = (
        gbt_pred.select("text", "label", F.col("prediction").cast("int").alias("prediction"))
        .limit(N)
        .toPandas()
        .to_dict(orient="records")
    )
    _write_json(os.path.join(out_pred_dir, "gbt_pred_sample.json"), gbt_rows)

    # FINAL REPORT
    report = {
        "mode": args.mode,
        "dataset": {
            "raw_rows": meta.get("rows"),
            "clean_rows": clean_meta.get("rows_after"),
            "dropped": clean_meta.get("dropped"),
        },
        "features": feat_meta,
        "features_params": {"vocab_size": vocab_size, "min_df": min_df},
        "feature_artifacts_dir": feature_art_dir,
        "params": {
            "shuffle_partitions": shuffle_partitions,
            "npart_clean": npart_clean,
            "sample_fraction": sample_fraction,
            "limit_rows": args.limit_rows,
            "rf_num_trees": rf_num_trees,
            "rf_max_depth": rf_max_depth,
            "gbt_max_iter": gbt_max_iter,
            "gbt_max_depth": gbt_max_depth,
        },
        "times_sec": times,
        "models": {
            "randomforest": rf_metrics,
            "gbt": gbt_metrics,
        },
        "artifacts": {
            "tree_artifacts_dir": os.path.join(REPORTS_DIR, "tree_artifacts"),
            "pred_sample_dir": out_pred_dir,
        }
    }

    _write_json(os.path.join(REPORTS_DIR, "run_report_trees_demo.json"), report)
    logger.info(f"Saved trees demo report to: {os.path.join(REPORTS_DIR, 'run_report_trees_demo.json')}")

    # cleanup
    try:
        df_feat.unpersist()
        df_clean.unpersist()
        train_df.unpersist()
        test_df.unpersist()
    except Exception:
        pass

    spark.stop()


if __name__ == "__main__":
    main()