# Fashion Product Classification using Fashion-MNIST (CNN)
# Team 2 | Deep Learning Case Study (BCA701)
#
# HOW TO RUN (Google Colab):
#   1. colab.research.google.com -> New notebook
#   2. Runtime -> Change runtime type -> T4 GPU (optional, makes it faster)
#   3. Paste this ENTIRE file into ONE cell and press Run (Shift+Enter)
#   4. It saves 4 pictures + results.json (open the folder icon on the left to see them)
#
# Settings match the PPT: Adam lr=0.001, up to 15 epochs, batch 64, dropout 0.3, early stopping.

import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import (confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, precision_recall_fscore_support)

tf.random.set_seed(42)
np.random.seed(42)

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# ---------------------------------------------------------------
# 1. Load data (60,000 train + 10,000 test, 28x28 grayscale)
# ---------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print("Train:", x_train.shape, "Test:", x_test.shape)

# Sample grid -> dataset_samples.png (goes on the Dataset slide)
plt.figure(figsize=(10, 4))
for i in range(10):
    idx = np.where(y_train == i)[0][0]
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[idx], cmap="gray")
    plt.title(CLASS_NAMES[i], fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.savefig("dataset_samples.png", dpi=200)
plt.show()

# ---------------------------------------------------------------
# 2. Preprocess: scale to 0-1, reshape to 28x28x1
# ---------------------------------------------------------------
x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

# ---------------------------------------------------------------
# 3. Build the CNN (225,034 parameters)
# ---------------------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])
model.summary()

# ---------------------------------------------------------------
# 4. Compile and train (10% of train data used for validation)
# ---------------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=2,
)

# ---------------------------------------------------------------
# 5. Evaluate on the 10,000 test images
# ---------------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTEST ACCURACY: {test_acc * 100:.2f}%   TEST LOSS: {test_loss:.4f}")

# Accuracy and loss curves -> training_curves.png
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history.history["accuracy"], label="Train")
ax[0].plot(history.history["val_accuracy"], label="Validation")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(history.history["loss"], label="Train")
ax[1].plot(history.history["val_loss"], label="Validation")
ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=200)
plt.show()

# Confusion matrix + precision / recall / F1 per class -> confusion_matrix.png
y_prob = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
    ax=ax, cmap="Oranges", xticks_rotation=45, colorbar=False
)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.show()

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3))

per_class = cm.diagonal() / cm.sum(axis=1)
print("Per-class accuracy:")
for name, acc in sorted(zip(CLASS_NAMES, per_class), key=lambda t: -t[1]):
    print(f"  {name:12s} {acc * 100:5.1f}%")

# ---------------------------------------------------------------
# 6. Prediction samples (green = correct, red = wrong) -> prediction_samples.png
# ---------------------------------------------------------------
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")
    ok = y_pred[i] == y_test[i]
    plt.title(f"Pred: {CLASS_NAMES[y_pred[i]]}\nTrue: {CLASS_NAMES[y_test[i]]}",
              fontsize=9, color="green" if ok else "red")
    plt.axis("off")
plt.tight_layout()
plt.savefig("prediction_samples.png", dpi=200)
plt.show()

# ---------------------------------------------------------------
# 7. Save the model, the numbers (results.json) + live demo helper
# ---------------------------------------------------------------
model.save("fashion_cnn.keras")

prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=range(10), zero_division=0)
results = {
    "test_acc": float(test_acc),
    "test_loss": float(test_loss),
    "epochs_run": len(history.history["loss"]),
    "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
    "best_val_acc": float(max(history.history["val_accuracy"])),
    "final_train_acc": float(history.history["accuracy"][-1]),
    "final_val_acc": float(history.history["val_accuracy"][-1]),
    "class_names": CLASS_NAMES,
    "precision": [float(v) for v in prec],
    "recall": [float(v) for v in rec],
    "f1": [float(v) for v in f1],
    "confusion_matrix": cm.tolist(),
    "correct_in_first_10": int(np.sum(y_pred[:10] == y_test[:10])),
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: dataset_samples.png, training_curves.png, confusion_matrix.png, "
      "prediction_samples.png, results.json, fashion_cnn.keras")


def predict_test_image(index):
    """Live demo: pick any index 0-9999 from the test set."""
    probs = model.predict(x_test[index:index + 1], verbose=0)[0]
    pred = int(np.argmax(probs))
    plt.imshow(x_test[index].squeeze(), cmap="gray")
    plt.title(f"Predicted: {CLASS_NAMES[pred]} ({probs[pred] * 100:.1f}%)\n"
              f"Actual: {CLASS_NAMES[int(y_test[index])]}")
    plt.axis("off")
    plt.show()


predict_test_image(0)
predict_test_image(25)
