"""
Handwritten Digit Recognizer - CNN on the Kaggle "Digit Recognizer" (MNIST) dataset.

Dataset: https://www.kaggle.com/c/digit-recognizer  (train.csv, test.csv)

Run locally:   put train.csv and test.csv next to this file, then `python mnist_cnn.py`
Run on Kaggle: add the "Digit Recognizer" competition data to the notebook and paste this in.

Requirements: pip install tensorflow pandas numpy matplotlib scikit-learn
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models

# Use a non-interactive backend when there is no display (plots are saved to disk anyway)
if not os.environ.get("DISPLAY") and os.name != "nt" and "KAGGLE_KERNEL_RUN_TYPE" not in os.environ:
    matplotlib.use("Agg")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# 1. LOAD
# --------------------------------------------------------------------------
def find_file(name):
    candidates = [
        name,
        os.path.join("data", name),
        os.path.join("/kaggle/input/digit-recognizer", name),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {name}. Download it from "
        "https://www.kaggle.com/c/digit-recognizer and place it next to this script "
        "or in a ./data folder."
    )


train_df = pd.read_csv(find_file("train.csv"))
test_df = pd.read_csv(find_file("test.csv"))
print("Train:", train_df.shape, "| Test:", test_df.shape)

y = train_df["label"].values
X = train_df.drop(columns="label").values
X_test = test_df.values


# --------------------------------------------------------------------------
# 2. PREPROCESS
# --------------------------------------------------------------------------
# Scale pixels to 0-1 and reshape to (N, 28, 28, 1)
X = (X / 255.0).astype("float32").reshape(-1, 28, 28, 1)
X_test = (X_test / 255.0).astype("float32").reshape(-1, 28, 28, 1)

# 90/10 stratified split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=SEED
)
print("Train split:", X_train.shape, "| Validation split:", X_val.shape)


# --------------------------------------------------------------------------
# 3. EXPLORE
# --------------------------------------------------------------------------
# Class distribution
plt.figure(figsize=(6, 3.5))
pd.Series(y).value_counts().sort_index().plot(kind="bar", color="steelblue")
plt.title("Class distribution")
plt.xlabel("Digit")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/class_distribution.png", dpi=120)
plt.show()

# 10 sample digits
fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
for ax, img, label in zip(axes.ravel(), X[:10], y[:10]):
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(f"Label: {label}")
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/sample_digits.png", dpi=120)
plt.show()


# --------------------------------------------------------------------------
# 4. MODEL
# --------------------------------------------------------------------------
def build_model():
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = build_model()
model.summary()


# --------------------------------------------------------------------------
# 5. TRAIN
# --------------------------------------------------------------------------
cbs = [
    callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=3,
    batch_size=256,
    callbacks=cbs,
    verbose=2,
)


# --------------------------------------------------------------------------
# 6. EVALUATE
# --------------------------------------------------------------------------
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation accuracy: {val_acc:.4f} | Validation loss: {val_loss:.4f}")

# Loss / accuracy curves
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history.history["loss"], label="train")
axes[0].plot(history.history["val_loss"], label="val")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(history.history["accuracy"], label="train")
axes[1].plot(history.history["val_accuracy"], label="val")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/training_curves.png", dpi=120)
plt.show()

# Predictions on validation set
val_probs = model.predict(X_val, verbose=0)
val_pred = val_probs.argmax(axis=1)

print("\nClassification report:\n", classification_report(y_val, val_pred, digits=4))

# Confusion matrix
cm = confusion_matrix(y_val, val_pred)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion matrix (validation)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=120)
plt.show()

# Misclassified digits
wrong = np.where(val_pred != y_val)[0]
print(f"Misclassified: {len(wrong)} of {len(y_val)}")
n_show = min(15, len(wrong))
if n_show > 0:
    fig, axes = plt.subplots(3, 5, figsize=(10, 6.5))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), wrong[:n_show]):
        ax.imshow(X_val[idx].squeeze(), cmap="gray")
        ax.set_title(f"True {y_val[idx]} / Pred {val_pred[idx]}", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/misclassified.png", dpi=120)
    plt.show()


# --------------------------------------------------------------------------
# 7. PREDICT & EXPORT
# --------------------------------------------------------------------------
test_pred = model.predict(X_test, verbose=0).argmax(axis=1)
submission = pd.DataFrame({"ImageId": np.arange(1, len(test_pred) + 1), "Label": test_pred})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv (upload this to the Kaggle competition)")

model.save("mnist_cnn.keras")
print("Saved model to mnist_cnn.keras")
