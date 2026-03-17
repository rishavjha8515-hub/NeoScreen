import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)

CLASS_NAMES = ["Low", "Medium", "High"]


def load_test_data(test_dir: str, img_size=(224, 224)):
    """Load all images from test_dir/Low, /Medium, /High subfolders."""
    import cv2

    images, labels = [], []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        folder = os.path.join(test_dir, class_name)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            images.append(img.astype(np.float32) / 255.0)
            labels.append(class_idx)

    return np.array(images), np.array(labels)


def evaluate(model_path: str, test_dir: str, output_json: str = "eval_results.json"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    X, y_true = load_test_data(test_dir)
    print(f"Loaded {len(X)} test images")

    probs = []
    for img in X:
        interpreter.set_tensor(inp[0]["index"], np.expand_dims(img, axis=0))
        interpreter.invoke()
        probs.append(interpreter.get_tensor(out[0]["index"])[0])

    probs = np.array(probs)
    y_pred = np.argmax(probs, axis=1)

    # Apply NeoScreen thresholds
    y_thresh = []
    for p in probs:
        if p[2] >= 0.35:
            y_thresh.append(2)  # HIGH
        elif p[0] >= 0.60:
            y_thresh.append(0)  # LOW
        else:
            y_thresh.append(1)  # MEDIUM
    y_thresh = np.array(y_thresh)

    cm = confusion_matrix(y_true, y_thresh)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    # High-Risk sensitivity (most critical)
    high_idx = 2
    tp_high = cm[high_idx, high_idx]
    fn_high = cm[high_idx, :].sum() - tp_high
    sensitivity_high = tp_high / (tp_high + fn_high + 1e-9)

    tn_high = np.delete(np.delete(cm, high_idx, 0), high_idx, 1).sum()
    fp_high = cm[:, high_idx].sum() - tp_high
    specificity_high = tn_high / (tn_high + fp_high + 1e-9)

    kappa = cohen_kappa_score(y_true, y_thresh)

    # One-vs-rest AUC
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    auc = roc_auc_score(y_bin, probs, multi_class="ovr", average="macro")

    results = {
        "n_samples": len(X),
        "sensitivity_high_risk": round(float(sensitivity_high), 4),
        "specificity_high_risk": round(float(specificity_high), 4),
        "cohen_kappa": round(float(kappa), 4),
        "auc_roc_macro": round(float(auc), 4),
        "targets": {
            "sensitivity": "≥ 0.95",
            "kappa": "> 0.80",
            "auc": "> 0.92",
        },
    }

    print("\n── NeoScreen Evaluation Results ──")
    for k, v in results.items():
        print(f"  {k}: {v}")

    print("\n── Classification Report ──")
    print(classification_report(y_true, y_thresh, target_names=CLASS_NAMES))

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {output_json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="neoscreen_v1.tflite")
    parser.add_argument("--test_dir", default="./dataset/test")
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()
    evaluate(args.model, args.test_dir, args.output)