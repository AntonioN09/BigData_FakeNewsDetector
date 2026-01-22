import argparse
import json
import logging
import os
import time

from pyspark import StorageLevel

from src.utils.logging import setup_logging
from src.utils.spark import create_spark_session
from src.utils.timing import Timer
from src.utils.save_safe import export_feature_artifacts

from src.data.load import load_dataset, load_label
from src.data.clean import clean_text_df
from src.features.text_features import build_text_feature_pipeline
from src.model.train import train_models
from src.model.evaluate import compute_metrics, save_metrics_json, top_errors
from src.utils.export_weights import export_model_weights



# 🔴 HARD-CODE PATHS
RAW_DATA_DIR = r"C:\Users\carin\OneDrive\Desktop\ibd\fake-news-detection-spark\data\raw\News_dataset"
MODELS_DIR = r"C:\Users\carin\OneDrive\Desktop\ibd\fake-news-detection-spark\outputs\models"
REPORTS_DIR = r"C:\Users\carin\OneDrive\Desktop\ibd\fake-news-detection-spark\outputs\reports"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["debug", "full"], default="debug")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("pipeline")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    logger.info(f"Running in {args.mode.upper()} mode")
    logger.info(f"RAW_DATA_DIR = {RAW_DATA_DIR}")
    logger.info(f"Exists? {os.path.isdir(RAW_DATA_DIR)}")

    # ---- tuned params by mode ----
    if args.mode == "debug":
        shuffle_partitions = 16
        npart_clean = 16
        sample_fraction = 0.10
        vocab_size = 12000
        min_df = 5
        models_to_train = ["linearsvc"]   # ✅ debug = 1 model
        lr_max_iter = 12
        svc_max_iter = 20
    else:
        shuffle_partitions = 64
        npart_clean = 64
        sample_fraction = None
        vocab_size = 20000
        min_df = 10
        models_to_train = ["logreg", "linearsvc"]
        lr_max_iter = 30
        svc_max_iter = 30
    # --------------------------------

    times = {}

    # SPARK
    with Timer("Spark session init"):
        spark = create_spark_session({
            "spark": {
                "master": "local[*]",
                "app_name": "FakeNewsSpark",
                "driver_memory": "4g",
                "executor_memory": "4g",
                "shuffle_partitions": shuffle_partitions,
                # dacă ai patch în create_spark_session pentru extra configs:
                "configs": {
                    "spark.sql.adaptive.enabled": "true",
                    "spark.sql.adaptive.coalescePartitions.enabled": "true",
                    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
                }
            }
        })

    # dump conf minimal
    try:
        spark_conf_dump = {
            "spark.sql.shuffle.partitions": spark.conf.get("spark.sql.shuffle.partitions"),
            "spark.sql.adaptive.enabled": spark.conf.get("spark.sql.adaptive.enabled"),
            "spark.defaultParallelism": spark.sparkContext.defaultParallelism,
        }
    except Exception:
        spark_conf_dump = {}

    # LOAD
    t0 = time.time()
    with Timer("Load dataset"):
        dataframes = load_dataset(spark, RAW_DATA_DIR)
        df, meta = load_label(dataframes.df_true, dataframes.df_fake)




    times["load_dataset_sec"] = round(time.time() - t0, 2)

    logger.info(f"Meta: {meta}")
    logger.info(f"Columns: {df.columns}")

    if args.mode == "debug":
        df.select("title", "text", "label").show(3, truncate=80)

    # CLEAN (NU cache aici)
    t0 = time.time()
    with Timer("Clean dataset"):
        df_clean, clean_meta = clean_text_df(
            df,
            min_text_length=30,
            dedup=True,
            cache=False
        )
    times["clean_sec"] = round(time.time() - t0, 2)

    logger.info(f"Clean meta: {clean_meta}")
    if args.mode == "debug":
        df_clean.select("title", "text", "label").show(3, truncate=80)

    # DEBUG sample (reduce enorm timpul)
    if sample_fraction is not None:
        df_clean = df_clean.sample(withReplacement=False, fraction=sample_fraction, seed=42)
        logger.info(f"DEBUG mode: using {int(sample_fraction * 100)}% sample after cleaning.")

    # repartition + persist înainte de feature stage (stage greu)
    df_clean = df_clean.repartition(npart_clean).persist(StorageLevel.MEMORY_AND_DISK)
    _ = df_clean.count()  # materialize o singură dată
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
        df_feat = feat_model.transform(df_clean)

    # persist pentru training (se folosește de 1-2 modele)
    df_feat = df_feat.persist(StorageLevel.MEMORY_AND_DISK)
    _ = df_feat.count()
    times["features_sec"] = round(time.time() - t0, 2)

    logger.info(f"Feature meta: {feat_meta}")
    logger.info(f"df_feat partitions: {df_feat.rdd.getNumPartitions()}")
    if args.mode == "debug":
        df_feat.select("label", "features").show(3, truncate=80)

    # SAFE SAVE: vocab + idf
    feature_art_dir = os.path.join(REPORTS_DIR, "feature_artifacts")
    art_paths = export_feature_artifacts(feat_model, feature_art_dir)
    logger.info(f"Saved feature artifacts (safe) to: {art_paths}")

    # TRAIN
    t0 = time.time()
    with Timer("Train models"):
        results = train_models(
            df_feat,
            label_col="label",
            features_col="features",
            seed=42,
            models_to_train=models_to_train,
            lr_max_iter=lr_max_iter,
            svc_max_iter=svc_max_iter,
            reg_param=0.1,
        )
    times["train_sec"] = round(time.time() - t0, 2)

    # EVAL + SAVE
    all_metrics = {}

    logger.info(f"Trained models: {list(results.keys())}")

    for name, pack in results.items():
    # --- unpack ---
        pred = pack["pred"]
        model = pack["model"]

        # --- metrics ---
        metrics = compute_metrics(pred, label_col="label", pred_col="prediction")
        all_metrics[name] = metrics
        logger.info(f"[{name}] metrics: {metrics}")

        out_file = save_metrics_json(
            metrics,
            REPORTS_DIR,
            f"metrics_{name}.json"
        )
        logger.info(f"[{name}] saved metrics to: {out_file}")

        # --- export weights (PENTRU predict.py) ---
        weights_dir = os.path.join(REPORTS_DIR, "model_weights")
        os.makedirs(weights_dir, exist_ok=True)

        w_path = export_model_weights(
            model=model,
            out_dir=weights_dir,
            model_name=name
        )
        logger.info(f"[{name}] saved model weights to: {w_path}")

        # --- debug samples ---
        if args.mode == "debug":
            fp_df, fn_df = top_errors(pred, k=5)
            logger.info(f"[{name}] False Positives sample:")
            fp_df.show(5, truncate=80)
            logger.info(f"[{name}] False Negatives sample:")
            fn_df.show(5, truncate=80)

        # --- cleanup ---
    try:
        pred.unpersist()
    except Exception:
        pass

    # FINAL REPORT
    final_report = {
        "mode": args.mode,
        "paths": {
            "raw_data_dir": RAW_DATA_DIR,
            "models_dir": MODELS_DIR,
            "reports_dir": REPORTS_DIR
        },
        "spark_conf": spark_conf_dump,
        "partitions": {
            "shuffle_partitions": shuffle_partitions,
            "npart_clean": npart_clean,
        },
        "dataset": {
            "raw_rows": meta.get("rows"),
            "clean_rows": clean_meta.get("rows_after"),
            "dropped": clean_meta.get("dropped"),
        },
        "clean_params": {
            "min_text_length": 30,
            "dedup": True,
        },
        "features": feat_meta,
        "features_params": {
            "vocab_size": vocab_size,
            "min_df": min_df
        },
        "feature_artifacts_dir": feature_art_dir,
        "times_sec": times,
        "models": all_metrics
    }

    final_report_path = os.path.join(REPORTS_DIR, "run_report.json")
    with open(final_report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    logger.info(f"Saved final report to: {final_report_path}")

    # cleanup cache
    try:
        df_feat.unpersist()
    except Exception:
        pass
    try:
        df_clean.unpersist()
    except Exception:
        pass

    spark.stop()


if __name__ == "__main__":
    main()
