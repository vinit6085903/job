# ===============================
# 1️⃣ IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

# ===============================
# 2️⃣ LOAD DATASET
# ===============================
csv_path = r"E:\job\data\fake_job_dataset\fake_job_postings.csv"
df = pd.read_csv(csv_path)

print("Dataset Shape:", df.shape)
print("Columns:", df.columns)

# ===============================
# 3️⃣ CLEAN DATA
# ===============================
df = df.fillna("")

# Combine text columns
df["text"] = (
    df["title"] + " " +
    df["company_profile"] + " " +
    df["description"] + " " +
    df["requirements"] + " " +
    df["benefits"]
)

texts = df["text"]
labels = df["fraudulent"]

# ===============================
# 4️⃣ TOKENIZATION
# ===============================
MAX_WORDS = 30000
MAX_LEN   = 300

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded_seq = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")

# ===============================
# 5️⃣ TRAIN–TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    padded_seq,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("Train size:", X_train.shape)
print("Test size :", X_test.shape)

# ===============================
# 6️⃣ BUILD LSTM MODEL
# ===============================
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),

    LSTM(128, return_sequences=True),
    Dropout(0.3),

    LSTM(64),
    Dropout(0.3),

    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# 7️⃣ TRAIN MODEL
# ===============================
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=64
)

# ===============================
# 8️⃣ EVALUATION
# ===============================
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc   = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTrain Accuracy: {train_acc*100:.2f}%")
print(f"Test  Accuracy: {test_acc*100:.2f}%")

# Classification report
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 9️⃣ SAVE MODEL & TOKENIZER
# ===============================
os.makedirs("saved_model", exist_ok=True)

#  Keras 3 compatible save (extension mandatory)
model.save("saved_model/fake_job_lstm_model.keras")

# Save tokenizer
with open("saved_model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("\n Model & Tokenizer saved successfully")
