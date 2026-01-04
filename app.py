import streamlit as st
import requests

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🕵️",
    layout="centered"
)

st.title("🕵️ Fake Job Detection System")
st.write("AI-based system to check whether a job posting is **Real, Suspicious, or Fake**")

st.markdown("---")

# =========================
# INPUT FIELDS
# =========================
title = st.text_input("📌 Job Title")
company_profile = st.text_area("🏢 Company Profile")
description = st.text_area("📝 Job Description")
requirements = st.text_area("📋 Requirements")
benefits = st.text_area("🎁 Benefits")

# =========================
# BUTTON
# =========================
if st.button("🔍 Check Job"):

    payload = {
        "title": title,
        "company_profile": company_profile,
        "description": description,
        "requirements": requirements,
        "benefits": benefits
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()

            # =========================
            # HANDLE VALIDATION ERROR
            # =========================
            if "status" in result:
                st.warning(result["message"])
            else:
                prediction = result["prediction"]
                fake_prob = result["fake_probability"]
                reason = result["decision_reason"]

                st.markdown("## 🧾 Result")

                # =========================
                # DISPLAY RESULT
                # =========================
                if "FAKE" in prediction:
                    st.error(f"🚨 {prediction}")
                elif "SUSPICIOUS" in prediction:
                    st.warning(f"⚠️ {prediction}")
                else:
                    st.success(f"✅ {prediction}")

                st.markdown(f"**Fake Probability:** `{fake_prob}`")
                st.markdown(f"**Reason:** {reason}")

                st.markdown("---")
                st.info(
                    "ℹ️ **Tip:** If a job is marked *Suspicious*, verify the company website, "
                    "LinkedIn page, and never pay any registration fees."
                )

        else:
            st.error("❌ Server error. Please try again.")

    except Exception as e:
        st.error("🚫 FastAPI server not running. Please start backend first.")
