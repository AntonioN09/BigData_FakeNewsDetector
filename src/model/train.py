from typing import Dict, List, Optional

from pyspark import StorageLevel
from pyspark.sql import DataFrame
from pyspark.ml.classification import LogisticRegression, LinearSVC


def train_models(
    df_feat: DataFrame,
    *,
    label_col: str = "label",
    features_col: str = "features",
    seed: int = 42,
    models_to_train: Optional[List[str]] = None,
    lr_max_iter: int = 20,
    svc_max_iter: int = 30,
    reg_param: float = 0.1,
) -> Dict[str, Dict]:
    """
    Train 1-2 modele (LogReg, LinearSVC) optimizat pt viteză:
      - randomSplit o singură dată
      - persist train/test + materialize o singură dată
      - calcule train/test count o singură dată
      - opțional: rulezi doar 1 model în debug (models_to_train)

    Return:
      {
        "logreg": {"model": ..., "pred": ..., "train_rows": ..., "test_rows": ..., "params": {...}},
        "linearsvc": {...}
      }
    """
    if models_to_train is None:
        models_to_train = ["logreg", "linearsvc"]

    allowed = {"logreg", "linearsvc"}
    unknown = set(models_to_train) - allowed
    if unknown:
        raise ValueError(f"models_to_train contains unknown values: {sorted(unknown)}")

    # ensure label double
    df = df_feat.withColumn(label_col, df_feat[label_col].cast("double"))

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=seed)

    # persist to avoid recompute per model
    train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
    test_df = test_df.persist(StorageLevel.MEMORY_AND_DISK)

    train_rows = train_df.count()
    test_rows = test_df.count()

    results: Dict[str, Dict] = {}

    if "logreg" in models_to_train:
        lr = LogisticRegression(
            labelCol=label_col,
            featuresCol=features_col,
            maxIter=lr_max_iter,
            regParam=reg_param,
            elasticNetParam=0.0
        )
        lr_model = lr.fit(train_df)
        lr_pred = lr_model.transform(test_df).persist(StorageLevel.MEMORY_AND_DISK)
        _ = lr_pred.count()  # materialize pred once

        results["logreg"] = {
            "model": lr_model,
            "pred": lr_pred,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "params": {"maxIter": lr_max_iter, "regParam": reg_param},
        }

    if "linearsvc" in models_to_train:
        svc = LinearSVC(
            labelCol=label_col,
            featuresCol=features_col,
            maxIter=svc_max_iter,
            regParam=reg_param
        )
        svc_model = svc.fit(train_df)
        svc_pred = svc_model.transform(test_df).persist(StorageLevel.MEMORY_AND_DISK)
        _ = svc_pred.count()  # materialize pred once

        results["linearsvc"] = {
            "model": svc_model,
            "pred": svc_pred,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "params": {"maxIter": svc_max_iter, "regParam": reg_param},
        }

    # free train/test cache
    train_df.unpersist()
    test_df.unpersist()

    return results
