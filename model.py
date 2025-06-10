import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import io
import sys




sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


print("Loading dataset...")
df = pd.read_csv("vidhuk_reviews_cleaned_extra_5.csv", encoding="utf-8")


print("Filtering relevant reviews (1, 4, 5 stars only)...")
df = df[df["stars"].isin([1, 2, 4, 5])]
df = df[["cleaned_text", "stars"]].dropna()


df["label"] = df["stars"].apply(lambda x: 0 if x in [1, 2] else 1)


print("Balancing the dataset...")
negatives = df[df["label"] == 0]
positives = df[df["label"] == 1]
min_size = min(len(negatives), len(positives))

balanced_df = pd.concat([
    negatives.sample(min_size, random_state=42),
    positives.sample(min_size, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Balanced dataset size: {len(balanced_df)} samples")


print("Splitting data into training and validation sets...")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    balanced_df["cleaned_text"].values,
    balanced_df["label"].values,
    test_size=0.2,
    stratify=balanced_df["label"],
    random_state=42
)


print("Creating TensorFlow datasets...")
batch_size = 32
train_ds = tf.data.Dataset.from_tensor_slices((train_texts, train_labels)).shuffle(10000).batch(batch_size)
val_ds = tf.data.Dataset.from_tensor_slices((val_texts, val_labels)).batch(batch_size)


print("Vectorizing text...")
max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length
)
vectorize_layer.adapt(train_ds.map(lambda text, label: text))


print("Building model...")
model = keras.Sequential([
    vectorize_layer,
    layers.Embedding(max_features + 1, 16),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


print("Training model...")
epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)


print("Evaluating model on validation set...")
loss, accuracy = model.evaluate(val_ds)
print(f"\n Validation Accuracy: {accuracy:.4f}")


model_path = "saved_model_vidhuk.h5"
print(f"Saving model to '{model_path}'...")
model.save(model_path)


history_df = pd.DataFrame(history.history)
history_file = "training_history.csv"
print(f"Saving training history to '{history_file}'...")
history_df.to_csv(history_file, index=False, encoding="utf-8")


print("Plotting training curves...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plot_file = "training_metrics.png"
plt.savefig(plot_file)
plt.show()
print(f"Training curves saved as '{plot_file}'")


print("Generating confusion matrix...")
val_text_tensor = tf.convert_to_tensor(val_texts)
pred_probs = model.predict(val_text_tensor, batch_size=batch_size)
pred_labels = (pred_probs > 0.5).astype("int32").flatten()

cm = confusion_matrix(val_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
conf_matrix_file = "confusion_matrix.png"
plt.savefig(conf_matrix_file)
plt.show()
print(f"Confusion matrix saved as '{conf_matrix_file}'")

print("All done.")
