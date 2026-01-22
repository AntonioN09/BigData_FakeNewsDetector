from typing import Dict, Tuple

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
)


def build_text_feature_pipeline(
    *,
    input_col: str = "text",
    output_col: str = "features",
    vocab_size: int = 20000,
    min_df: int = 5,
    binary_tf: bool = True,   # ✅ mare boost pt viteză
) -> Tuple[Pipeline, Dict]:
    """
    Pipeline NLP Spark optimizat pentru Big Data:
      text
        -> tokens (lowercase, regex simplu)
        -> filtered_tokens (stopwords EN)
        -> tf (CountVectorizer)
        -> idf
        -> features
    """

    # 1️⃣ Tokenizer – regex simplu + lowercase
    tokenizer = RegexTokenizer(
        inputCol=input_col,
        outputCol="tokens",
        pattern=r"[^a-zA-Z]+",   # mult mai rapid decât \W+
        toLowercase=True,
        minTokenLength=2
    )

    # 2️⃣ Stopwords – EN explicit (fără overhead)
    remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="filtered_tokens",
        stopWords=StopWordsRemover.loadDefaultStopWords("english"),
        caseSensitive=False
    )

    # 3️⃣ CountVectorizer – TF
    cv = CountVectorizer(
        inputCol="filtered_tokens",
        outputCol="tf",
        vocabSize=vocab_size,
        minDF=min_df,
        binary=binary_tf   # 🔥 HUGE speedup pt clasificare
    )

    # 4️⃣ IDF
    idf = IDF(
        inputCol="tf",
        outputCol=output_col
    )

    pipeline = Pipeline(stages=[tokenizer, remover, cv, idf])

    meta = {
        "input_col": input_col,
        "output_col": output_col,
        "vocab_size": vocab_size,
        "min_df": min_df,
        "binary_tf": binary_tf,
        "stages": [
            "RegexTokenizer",
            "StopWordsRemover",
            "CountVectorizer",
            "IDF"
        ]
    }

    return pipeline, meta
