import os
import json


def export_feature_artifacts(feat_model, out_dir: str):
    """
    Salvează vocabularul CountVectorizer + valorile IDF în JSON
    (safe pe Windows, fără winutils / HDFS)
    """
    os.makedirs(out_dir, exist_ok=True)

    cv_model = feat_model.stages[-2]   # CountVectorizerModel
    idf_model = feat_model.stages[-1]  # IDFModel

    vocab_path = os.path.join(out_dir, "cv_vocabulary.json")
    idf_path = os.path.join(out_dir, "idf_values.json")
    meta_path = os.path.join(out_dir, "features_meta.json")

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(cv_model.vocabulary, f, indent=2)

    with open(idf_path, "w", encoding="utf-8") as f:
        json.dump(idf_model.idf.toArray().tolist(), f, indent=2)

    meta = {
        "num_features": len(cv_model.vocabulary),
        "min_df": cv_model.getMinDF(),
        "input_col": cv_model.getInputCol(),
        "output_col": idf_model.getOutputCol(),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "vocab_json": vocab_path,
        "idf_json": idf_path,
        "meta_json": meta_path,
    }
