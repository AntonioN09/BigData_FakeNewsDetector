import logging
from typing import Dict, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel


def clean_text_df(
    df: DataFrame,
    *,
    min_text_length: int = 30,
    dedup: bool = True,
    cache: bool = True,
) -> Tuple[DataFrame, Dict]:
    """
    Curățare text:
      - drop null/empty pentru text
      - strip + lowercase
      - remove HTML
      - remove control chars
      - normalize whitespace
      - filtru min length
      - dedup opțional (pe hash(text))
      - persist/caching opțional

    Return: (df_curat, meta)
    """
    logger = logging.getLogger("data.clean")

    before = df.count()

    # 1) drop null/empty text
    df2 = df.filter(F.col("text").isNotNull())
    df2 = df2.withColumn("text", F.trim(F.col("text")))
    df2 = df2.filter(F.length(F.col("text")) > 0)

    # 2) lowercase
    df2 = df2.withColumn("text", F.lower(F.col("text")))
    df2 = df2.withColumn("title", F.lower(F.trim(F.col("title"))))

    # 3) remove HTML tags
    df2 = df2.withColumn("text", F.regexp_replace(F.col("text"), r"<[^>]+>", " "))

    # 4) remove control chars (tabs, newlines, etc.)
    df2 = df2.withColumn("text", F.regexp_replace(F.col("text"), r"[\r\n\t]", " "))

    # 5) keep reasonable chars (optional: keep punctuation). Aici doar curățăm caractere non-printabile.
    df2 = df2.withColumn("text", F.regexp_replace(F.col("text"), r"[\x00-\x1F\x7F]", " "))

    # 6) normalize spaces
    df2 = df2.withColumn("text", F.regexp_replace(F.col("text"), r"\s+", " "))
    df2 = df2.withColumn("title", F.regexp_replace(F.col("title"), r"\s+", " "))

    # 7) min length
    df2 = df2.filter(F.length(F.col("text")) >= F.lit(min_text_length))

    # 8) dedup by hash(text)
    if dedup:
        df2 = df2.withColumn("_text_hash", F.sha2(F.col("text"), 256))
        df2 = df2.dropDuplicates(["_text_hash"]).drop("_text_hash")

    after = df2.count()

    meta = {
        "rows_before": before,
        "rows_after": after,
        "dropped": before - after,
        "min_text_length": min_text_length,
        "dedup": dedup,
        "cache": cache,
    }

    logger.info(f"Clean meta: {meta}")

    if cache:
        # Persist pentru reuse (features + train/eval)
        df2 = df2.persist(StorageLevel.MEMORY_AND_DISK)
        # materialize cache
        _ = df2.count()
        logger.info("Cleaned DataFrame persisted (MEMORY_AND_DISK).")

    return df2, meta
