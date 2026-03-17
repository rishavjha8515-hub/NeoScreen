import time
from typing import Tuple

import numpy as np
import tensorflow as tf


RISK_MESSAGES = {
    "HIGH": {
        "hi": "Shishu ko turant PHC le jaayein",
        "en": "Take baby to PHC immediately",
    },
    "MEDIUM": {
        "hi": "Kal dobara check karein",
        "en": "Recheck tomorrow",
    },
    "LOW": {
        "hi": "Koi khatra nahi",
        "en": "No immediate risk",
    },
}

# Safety-first thresholds (deliberately low High threshold)
THRESHOLD_HIGH = 0.35
THRESHOLD_LOW = 0.60


def load_interpreter(model_path: str = "neoscreen_v1.tflite") -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def run_inference(interpreter: tf.lite.Interpreter, input_image: np.ndarray) -> np.ndarray:
    """
    Args:
        input_image: float32 array of shape (1, 224, 224, 3), values in [0, 1]
    Returns:
        softmax probabilities [P(Low), P(Medium), P(High)]
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_image)

    t0 = time.time()
    interpreter.invoke()
    elapsed_ms = (time.time() - t0) * 1000
    print(f"Inference time: {elapsed_ms:.1f}ms")

    output = interpreter.get_tensor(output_details[0]["index"])[0]
    return output  # [P(Low), P(Medium), P(High)]


def classify_risk(output: np.ndarray) -> str:
    """Apply threshold logic. Uncertain cases always escalate to High."""
    p_low, p_med, p_high = output[0], output[1], output[2]

    if p_high >= THRESHOLD_HIGH:
        return "HIGH"
    elif p_low >= THRESHOLD_LOW:
        return "LOW"
    else:
        return "MEDIUM"


def get_risk_message(risk: str, lang: str = "hi") -> str:
    lang = lang if lang in ("hi", "en") else "en"
    return RISK_MESSAGES[risk][lang]


def classify_jaundice(
    sclera_image: np.ndarray,
    model_path: str = "neoscreen_v1.tflite",
    lang: str = "hi",
) -> Tuple[str, str, np.ndarray]:
    """
    End-to-end convenience wrapper.

    Args:
        sclera_image: BGR uint8 numpy array of shape (224, 224, 3)
        model_path:   path to .tflite model file
        lang:         'hi' or 'en'

    Returns:
        (risk_label, message, raw_probabilities)
    """
    import cv2

    # Normalise
    rgb = cv2.cvtColor(sclera_image, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)

    interpreter = load_interpreter(model_path)
    output = run_inference(interpreter, img)
    risk = classify_risk(output)
    message = get_risk_message(risk, lang)

    print(f"Probabilities → Low: {output[0]:.3f}  Medium: {output[1]:.3f}  High: {output[2]:.3f}")
    print(f"Risk: {risk} | Message: {message}")

    return risk, message, output


if __name__ == "__main__":
    import sys, cv2
    from sclera_detection import detect_sclera

    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    sclera = detect_sclera(path)
    if sclera is None:
        print("Sclera not detected. Retake photo.")
        sys.exit(1)

    risk, message, probs = classify_jaundice(sclera)
    print(f"\nFINAL → {risk}: {message}")