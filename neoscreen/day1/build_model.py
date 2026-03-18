"""
Day 1 — Task 3 (Afternoon, 2 hrs)
Build and verify the MobileNetV3 3-class head architecture.
Run this BEFORE training to confirm the model structure looks correct.

Usage:
    python day1/build_model.py [--summary] [--plot]
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV3Small


def build_and_verify(show_summary: bool = True, plot: bool = False):
    print("=" * 54)
    print("NeoScreen — Day 1 Task 3: Model Architecture")
    print("=" * 54)

    # ── 1. Build model ────────────────────────────────────────────────────────
    print("\n[1/5] Loading MobileNetV3Small (ImageNet weights)...")
    base = MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False
    frozen_layers = sum(1 for l in base.layers if not l.trainable)
    print(f"  ✓ Base loaded — {len(base.layers)} layers, {frozen_layers} frozen")

    # ── 2. Attach 3-class head ────────────────────────────────────────────────
    print("\n[2/5] Attaching 3-class classification head...")
    inputs = keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(3, activation="softmax", name="risk_output")(x)
    model = keras.Model(inputs, outputs, name="NeoScreen_v1")
    print("  ✓ Head attached — 3 output classes: [Low, Medium, High]")

    # ── 3. Compile ────────────────────────────────────────────────────────────
    print("\n[3/5] Compiling...")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("  ✓ Compiled — Adam lr=1e-3, categorical_crossentropy")

    # ── 4. Dry run ────────────────────────────────────────────────────────────
    print("\n[4/5] Dry-run inference (random input)...")
    dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
    output = model.predict(dummy, verbose=0)
    assert output.shape == (1, 3), f"Expected (1,3), got {output.shape}"
    assert abs(output.sum() - 1.0) < 1e-5, "Softmax probabilities don't sum to 1"
    print(f"  ✓ Output shape: {output.shape}  Probabilities: {output[0].round(3)}")

    # ── 5. Trainable params ───────────────────────────────────────────────────
    print("\n[5/5] Parameter count:")
    total = model.count_params()
    trainable = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}  (head only during warm-up)")
    print(f"  Frozen params:    {total - trainable:,}  (MobileNetV3 base)")

    if show_summary:
        print()
        model.summary(line_length=72)

    if plot:
        try:
            keras.utils.plot_model(model, to_file="day1/model_architecture.png",
                                   show_shapes=True, dpi=96)
            print("\nArchitecture diagram saved → day1/model_architecture.png")
        except Exception as e:
            print(f"  (plot skipped: {e})")

    print("\n" + "=" * 54)
    print("Model architecture OK. Ready for training.")
    print("Run: python ml/train.py --data_dir ./dataset --epochs 25")
    print("=" * 54)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true", default=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    build_and_verify(show_summary=args.summary, plot=args.plot)
