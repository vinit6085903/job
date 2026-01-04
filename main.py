from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# CONFIG
# =========================
MAX_LEN = 300

FAKE_THRESHOLD = 0.50        # clear scam
SUSPICIOUS_THRESHOLD = 0.15  # warning zone

# =========================
# LOAD MODEL & TOKENIZER
# =========================
model = tf.keras.models.load_model(
    "saved_model/fake_job_lstm_model.keras"
)

with open("saved_model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Fake Job Detection API")

class JobRequest(BaseModel):
    title: str = ""
    company_profile: str = ""
    description: str = ""
    requirements: str = ""
    benefits: str = ""

@app.get("/")
def home():
    return {"message": "Fake Job Detection API is running 🚀"}

@app.post("/predict")
def predict_job(job: JobRequest):

    # 🔹 Combine text safely
    text = " ".join([
        job.title.strip(),
        job.company_profile.strip(),
        job.description.strip(),
        job.requirements.strip(),
        job.benefits.strip()
    ])

    # 🔴 Validation (very important)
    if len(text) < 30:
        return {
            "status": "INSUFFICIENT DATA ❗",
            "message": "Please provide proper job title and description for analysis."
        }

    # 🔹 Tokenize
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    # 🔹 Predict probability
    fake_prob = float(model.predict(padded, verbose=0)[0][0])

    # =========================
    # DECISION LOGIC (REAL WORLD)
    # =========================
    if fake_prob >= FAKE_THRESHOLD:
        label = "FAKE JOB ❌"
        reason = "High similarity with known scam patterns"
    elif fake_prob >= SUSPICIOUS_THRESHOLD:
        label = "SUSPICIOUS ⚠️"
        reason = "Generic language / easy-money keywords detected"
    else:
        label = "LIKELY REAL ✅"
        reason = "Job details appear professional and structured"

    return {
        "prediction": label,
        "fake_probability": round(fake_prob, 4),
        "decision_reason": reason,
        "thresholds": {
            "fake": FAKE_THRESHOLD,
            "suspicious": SUSPICIOUS_THRESHOLD
        }
    }
