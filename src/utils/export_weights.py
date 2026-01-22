import json
import os


def export_model_weights(model, out_dir: str, model_name: str) -> str:
    """
    Salvează weights pentru predict fără MLWriter (Windows friendly).
    Output:
      outputs/reports/model_weights/<model_name>_weights.json
    Suport:
      - LogisticRegressionModel
      - LinearSVCModel
    """
    os.makedirs(out_dir, exist_ok=True)

    # LogisticRegressionModel
    if hasattr(model, "coefficients") and hasattr(model, "intercept"):
        coef = model.coefficients.toArray().tolist()
        # intercept poate fi float sau vector (multiclass). La binary e float.
        intercept = float(model.intercept) if not hasattr(model.intercept, "__len__") else float(model.intercept[0])

        payload = {
            "type": model.__class__.__name__,
            "model": model_name,
            "coefficients": coef,
            "intercept": intercept
        }

    # LinearSVCModel
    elif hasattr(model, "coefficients") and hasattr(model, "intercept"):
        coef = model.coefficients.toArray().tolist()
        intercept = float(model.intercept)

        payload = {
            "type": model.__class__.__name__,
            "model": model_name,
            "coefficients": coef,
            "intercept": intercept
        }

    else:
        raise ValueError(f"Unsupported model type for weight export: {type(model)}")

    out_path = os.path.join(out_dir, f"{model_name}_weights.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_path
