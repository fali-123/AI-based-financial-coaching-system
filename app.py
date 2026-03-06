
import os
import pandas as pd
import joblib
import streamlit as st
import requests

st.set_page_config(page_title="AI Spending Coach", page_icon="💳", layout="centered")

DECISION_THRESHOLD = 0.60  # balanced

@st.cache_resource
def load_artifacts():
    model = joblib.load("final_model.joblib")
    df = pd.read_csv("cleaned_transactions.csv")
    merchant_col = "merchant_name_clean" if "merchant_name_clean" in df.columns else ("merchant_name" if "merchant_name" in df.columns else None)
    return model, df, merchant_col

def risk_bucket(p: float) -> str:
    if p < 0.35:
        return "Low"
    if p < 0.65:
        return "Moderate"
    return "High"

def is_flagged(p: float, threshold: float = DECISION_THRESHOLD) -> int:
    return int(p >= threshold)

def build_prompt(age, gender, category, merchant, risk_prob, flagged):
    bucket = risk_bucket(risk_prob)
    flag_text = "YES" if flagged == 1 else "NO"
    return f"""
You are a friendly financial behavior coach. Explain patterns in clear, supportive language.

Rules:
- Educational insights only.
- Do NOT give regulated financial advice.
- Do NOT promise outcomes.
- Keep it concise and actionable.
- Use a supportive, non-judgmental tone.

Context:
- Age: {age}
- Gender: {gender}
- Category: {category}
- Merchant: {merchant}
- High-spender probability (model): {risk_prob:.2f}
- Risk bucket: {bucket}
- Flagged as high spender at threshold 0.60? {flag_text}

Write:
1) A 2–3 sentence plain-English summary of what this suggests.
2) Three bullet suggestions starting with "Consider..."
3) One reflective question.
4) End with: "Educational only, not financial advice."
""".strip()

def hf_inference_generate(prompt: str, model_id: str = "google/flan-t5-large", timeout: int = 60) -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        return "LLM not configured. Add HF_TOKEN in Hugging Face Space Secrets."

    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 220, "do_sample": False}}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            return data[0]["generated_text"]
        return str(data)[:1500]
    except Exception as e:
        return f"LLM request failed: {e}"

# -------------------- UI --------------------
st.title("💳 AI Personal Spending Coach")
st.caption("Educational insights only — not financial advice.")

model, df, merchant_col = load_artifacts()

categories = sorted(df["category"].dropna().unique().tolist())
genders = sorted(df["gender"].dropna().unique().tolist())
merchants = sorted(df[merchant_col].dropna().unique().tolist()) if merchant_col else ["N/A"]

st.subheader("Enter your context")
age = st.slider("Age", min_value=18, max_value=85, value=30)

default_gender = "Unknown" if "Unknown" in genders else genders[0]
gender = st.selectbox("Gender", options=genders, index=genders.index(default_gender))

category = st.selectbox("Category", options=categories, index=0)
merchant = st.selectbox("Merchant", options=merchants, index=0)

st.markdown(f"**Decision threshold:** `{DECISION_THRESHOLD}` (balanced)")

if st.button("Generate my coaching insight"):
    row = {"age": age, "gender": gender, "category": category}
    if merchant_col:
        row[merchant_col] = merchant

    X_one = pd.DataFrame([row])
    risk_prob = float(model.predict_proba(X_one)[:, 1][0])
    flagged = is_flagged(risk_prob, DECISION_THRESHOLD)
    bucket = risk_bucket(risk_prob)

    st.markdown("### Prediction")
    st.metric("High-spender probability", f"{risk_prob:.2f}")
    st.write(f"Risk bucket: **{bucket}**")
    st.write(f"Flagged as high spender @ 0.60? **{'Yes' if flagged==1 else 'No'}**")

    prompt = build_prompt(age, gender, category, merchant, risk_prob, flagged)

    st.markdown("### AI Coach Message")
    coaching_text = hf_inference_generate(prompt, model_id="google/flan-t5-large")
    st.write(coaching_text)

    with st.expander("Show prompt (transparency)"):
        st.code(prompt)
