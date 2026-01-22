import os
import glob
import logging
from typing import Dict, Tuple, Optional, List

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from types import SimpleNamespace


COLUMN_ALIASES = {
    "title": ["title", "headline", "head", "heading"],
    "text": ["text", "content", "article", "body", "story"],
    "subject": ["subject", "category", "topic", "section"],
    "date": ["date", "published", "publish_date", "timestamp", "time"],
}

logger = logging.getLogger("data.load")

def _find_first_existing_col(df: DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _normalize_columns(df: DataFrame) -> DataFrame:
    mapping = {}
    for std_col, aliases in COLUMN_ALIASES.items():
        found = _find_first_existing_col(df, aliases)
        if found:
            mapping[std_col] = found

    if "text" not in mapping:
        raise ValueError(f"No text column found. Columns: {df.columns}")

    df2 = df.select(
        F.col(mapping.get("title", mapping["text"])).alias("title"),
        F.col(mapping["text"]).alias("text"),
        F.col(mapping.get("subject")).alias("subject") if "subject" in mapping else F.lit(None).alias("subject"),
        F.col(mapping.get("date")).alias("date") if "date" in mapping else F.lit(None).alias("date"),
        F.col("label").alias("label")
    )

    return df2


def _read_csv(spark: SparkSession, path: str) -> DataFrame:
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("multiLine", True)
        .option("escape", "\"")
        .option("quote", "\"")
        .csv(path)
    )

def load_dataset(spark: SparkSession, raw_dir: str):

    logger.info(f"Loading dataset from: {raw_dir}")
    raw_dir = os.path.normpath(raw_dir)
    
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"RAW DIR DOES NOT EXIST: {raw_dir}")
    true_path=os.path.join(raw_dir, "True.csv")
    fake_path=os.path.join(raw_dir, "Fake.csv")
    logger.info("Detected True.csv / Fake.csv dataset")
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"Missing file: {true_path}")
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Missing file: {fake_path}")
    df_true = _read_csv(spark, true_path)
    df_fake = _read_csv(spark, fake_path)

    return SimpleNamespace(df_true=df_true, df_fake=df_fake)\
    

def load_label(df_true, df_fake) -> Tuple[DataFrame, Dict]:

    df_true_l = df_true.withColumn("label", F.lit(1).cast("int"))
    df_fake_l = df_fake.withColumn("label", F.lit(0).cast("int"))
    
    df = df_true_l.unionByName(df_fake_l, allowMissingColumns=True)
    df = _normalize_columns(df)
    df = df.withColumn("label", F.col("label").cast("int"))
    meta = {
        "source": "true_fake",
        "rows": df.count(),
       }

    return df, meta





def load_dataset1(spark: SparkSession, raw_dir: str) -> Tuple[DataFrame, Dict]:
    logger = logging.getLogger("data.load")

    raw_dir = os.path.normpath(raw_dir)
    logger.info(f"Loading dataset from: {raw_dir}")

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"RAW DIR DOES NOT EXIST: {raw_dir}")

    true_path = os.path.join(raw_dir, "True.csv")
    fake_path = os.path.join(raw_dir, "Fake.csv")

    if not os.path.exists(true_path):
        raise FileNotFoundError(f"Missing file: {true_path}")
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Missing file: {fake_path}")

    logger.info("Detected True.csv / Fake.csv dataset")

    df_true = _read_csv(spark, true_path).withColumn("label", F.lit(1))
    df_fake = _read_csv(spark, fake_path).withColumn("label", F.lit(0))

    df = df_true.unionByName(df_fake, allowMissingColumns=True)
    df = _normalize_columns(df)

    df = df.withColumn("label", F.col("label").cast("int"))

    meta = {
        "source": "true_fake",
        "files": [true_path, fake_path],
        "rows": df.count()
    }

    return df, meta