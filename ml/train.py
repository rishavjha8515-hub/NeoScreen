import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 3  # Low, Medium, High
CLASS_NAMES = ["Low", "Medium", "High"]


def build_model(num_classes: int = NUM_CLASSES) -> keras.Model:
    """
    Freeze MobileNetV3Small base, attach 3-class classification head.
    Fine-tuning-friendly: base unfreezes after initial warm-up training.
    """
    base = MobileNetV3Small(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False  # freeze during warm-up

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="NeoScreen")


def make_data_generators(data_dir: str):
    """Augmentation during training only. Val/test: normalise only."""
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
        validation_split=0.15,
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.15)

    train = train_gen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        classes=CLASS_NAMES,
    )
    val = val_gen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        classes=CLASS_NAMES,
    )
    return train, val


def train(data_dir: str, epochs: int, output_path: str):
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    train_ds, val_ds = make_data_generators(data_dir)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        keras.callbacks.ModelCheckpoint("best_neoscreen.keras", save_best_only=True),
    ]

    # Phase 1: warm-up (frozen base)
    print("\n── Phase 1: Warm-up (frozen base) ──")
    model.fit(train_ds, validation_data=val_ds, epochs=min(10, epochs), callbacks=callbacks)

    # Phase 2: fine-tune top layers of base
    print("\n── Phase 2: Fine-tune (top 30 layers unfrozen) ──")
    base = model.layers[1]
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),  # lower LR for fine-tuning
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=min(10, epochs),
        callbacks=callbacks,
    )

    # Export to TF Lite with float16 quantisation (~4MB)
    print("\n── Exporting to TF Lite ──")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved {output_path}  ({size_mb:.2f} MB)")
    assert size_mb < 5, f"Model too large ({size_mb:.2f} MB) — check quantisation"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeoScreen MobileNetV3")
    parser.add_argument("--data_dir", default="./dataset", help="Dataset root (Low/Medium/High subdirs)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--output", default="neoscreen_v1.tflite")
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.output)