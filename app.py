import streamlit as st
import requests

# =========================
# CONFIG
# =========================
API_URL = "https://job-kd84.onrender.com/predict"

st.set_page_config(
    page_title="Fake Job Detection | EquinoxSphere",
    page_icon="🕵️‍♂️",
    layout="centered"
)

# =========================
# CUSTOM CSS (PRO LOOK)
# =========================
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1, h2, h3, h4 {
    color: #f5f5f5;
}
label {
    color: #cfcfcf !important;
}
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 16px;
    font-weight: 600;
}
.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}
.metric {
    font-size: 22px;
    font-weight: bold;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 13px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown(
    "<h1 style='text-align:center;'>🕵️ Fake Job Detection System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:#bdbdbd;'>AI-powered platform to identify <b>Real, Suspicious, or Fake</b> job postings</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# INPUT FORM
# =========================
with st.form("job_form"):
    title = st.text_input("📌 Job Title")
    company_profile = st.text_area("🏢 Company Profile")
    description = st.text_area("📝 Job Description")
    requirements = st.text_area("📋 Requirements")
    benefits = st.text_area("🎁 Benefits")

    submit = st.form_submit_button("🔍 Analyze Job")

# =========================
# PREDICTION
# =========================
if submit:
    payload = {
        "title": title,
        "company_profile": company_profile,
        "description": description,
        "requirements": requirements,
        "benefits": benefits
    }

    with st.spinner("Analyzing job posting..."):
        try:
            res = requests.post(API_URL, json=payload, timeout=20)

            if res.status_code == 200:
                result = res.json()

                prediction = result["prediction"]
                fake_prob = result["fake_probability"]
                reason = result["decision_reason"]

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("## 🧾 Analysis Result")

                if "FAKE" in prediction:
                    st.error(f"🚨 {prediction}")
                elif "SUSPICIOUS" in prediction:
                    st.warning(f"⚠️ {prediction}")
                else:
                    st.success(f"✅ {prediction}")

                st.markdown(
                    f"<p class='metric'>Fake Probability: {round(fake_prob*100,2)}%</p>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**Reason:** {reason}")

                st.info(
                    "ℹ️ Tip: Never pay registration fees. Always verify the company website and LinkedIn presence."
                )
                st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.error("❌ Server error. Please try again later.")

        except Exception:
            st.error("🚫 Backend server is unreachable.")

# =========================
# FOOTER
# =========================
st.markdown(
    "<div class='footer'>Built by <b>EquinoxSphere</b> • AI Systems & Automation</div>",
    unsafe_allow_html=True
)
