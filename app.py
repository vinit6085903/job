from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# CONFIG
# =========================
MAX_LEN = 300
FAKE_THRESHOLD = 0.50
SUSPICIOUS_THRESHOLD = 0.15

# =========================
# LOAD MODEL & TOKENIZER
# =========================
model = tf.keras.models.load_model(
    "saved_model/fake_job_lstm_model.keras"
)

with open("saved_model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

# 🔹 Serve SPA HTML
@app.route("/")
def home():
    return render_template("index.html")

# 🔹 API
@app.route("/predict", methods=["POST"])
def predict_job():
    data = request.get_json()

    text = " ".join([
        data.get("title", "").strip(),
        data.get("company_profile", "").strip(),
        data.get("description", "").strip(),
        data.get("requirements", "").strip(),
        data.get("benefits", "").strip()
    ])

    if len(text) < 30:
        return jsonify({
            "status": "INSUFFICIENT DATA ❗",
            "message": "Please provide proper job details for analysis."
        }), 400

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    fake_prob = float(model.predict(padded, verbose=0)[0][0])

    if fake_prob >= FAKE_THRESHOLD:
        label = "FAKE JOB ❌"
        reason = "High similarity with known scam patterns"
    elif fake_prob >= SUSPICIOUS_THRESHOLD:
        label = "SUSPICIOUS ⚠️"
        reason = "Generic language / easy-money keywords detected"
    else:
        label = "LIKELY REAL ✅"
        reason = "Job details appear professional and structured"

    return jsonify({
        "prediction": label,
        "fake_probability": round(fake_prob, 4),
        "decision_reason": reason
    })

if __name__ == "__main__":
    app.run(debug=True)
