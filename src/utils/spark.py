import os
import sys
from pyspark.sql import SparkSession


def create_spark_session(config: dict) -> SparkSession:
    spark_cfg = config.get("spark", {})

    # 1) Python absolut (cel cu care rulezi scriptul)
    python_exe = sys.executable

    # 2) IMPORTANT: Spark workers citesc în primul rând aceste ENV-uri
    os.environ["PYSPARK_PYTHON"] = python_exe
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_exe

    builder = (
        SparkSession.builder
        .master(spark_cfg.get("master", "local[*]"))
        .appName(spark_cfg.get("app_name", "FakeNewsSpark"))
        .config("spark.driver.memory", spark_cfg.get("driver_memory", "4g"))
        .config("spark.executor.memory", spark_cfg.get("executor_memory", "4g"))
        .config("spark.sql.shuffle.partitions", spark_cfg.get("shuffle_partitions", 64))

        # 3) Și conf-urile Spark (în caz că ENV nu e suficient)
        .config("spark.pyspark.python", python_exe)
        .config("spark.pyspark.driver.python", python_exe)

        # 4) Stabilitate pe Windows
        .config("spark.python.worker.reuse", "true")
    )

    for k, v in spark_cfg.get("configs", {}).items():
        builder = builder.config(k, str(v))

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
