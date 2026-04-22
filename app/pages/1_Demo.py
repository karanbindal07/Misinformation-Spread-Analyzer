# app/pages/1_Demo.py

import json
import os
import streamlit as st
import joblib
import numpy as np

# ---------- Page config ----------
st.set_page_config(page_title="Demo Tool", page_icon="🧪", layout="wide")

# ---------- Load model artifacts ----------
@st.cache_resource
def load_artifacts():
    # NOTE: these paths assume you run streamlit from repo root:
    # streamlit run app/Home.py
    pro_tfidf = "models/tfidf_model_pro.joblib"
    pro_model = "models/lr_model_pro.joblib"

    if os.path.exists(pro_tfidf) and os.path.exists(pro_model):
        tfidf = joblib.load(pro_tfidf)
        model = joblib.load(pro_model)
        return tfidf, model, "Professional TF-IDF pipeline"

    tfidf = joblib.load("models/tfidf_model.joblib")
    model = joblib.load("models/lr_model.joblib")
    return tfidf, model, "Legacy TF-IDF pipeline"


@st.cache_data
def load_manifest():
    path = "models/model_manifest_pro.json"
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- Helpers ----------
def risk_label(p: float) -> str:
    if p >= 0.70:
        return "High Risk"
    if p >= 0.40:
        return "Medium Risk"
    return "Low Risk"

def decision_label(p: float) -> str:
    return "Flag for Review" if p >= 0.40 else "Monitor"

def _safe_str(x) -> str:
    # feature_names can sometimes be dtype=object; force clean python str
    try:
        return str(x)
    except Exception:
        return ""

def explain_prediction(tfidf, model, X_row, top_k=8, min_token_len=3):
    """
    For a logistic regression text model:
    contribution_i = tfidf_weight_i * coef_i  (log-odds contribution)
    Positive -> pushes toward class 1 (misinformation)
    Negative -> pushes toward class 0 (real)

    Returns:
      top_fake: list[(term, contribution)]
      top_real: list[(term, contribution)]
    """
    feature_names = tfidf.get_feature_names_out()
    coefs = model.coef_.ravel()

    # X_row: (1, n_features) sparse
    x = X_row.toarray().ravel()
    present_idx = np.where(x > 0)[0]
    if present_idx.size == 0:
        return [], []

    present_names = [_safe_str(feature_names[i]) for i in present_idx]
    present_x = x[present_idx]
    present_coefs = coefs[present_idx]
    contrib = present_x * present_coefs

    # light cleanup for readability (avoid 1-2 char junk like "tv" if you want)
    keep = []
    for i, term in enumerate(present_names):
        term_clean = term.strip()
        if len(term_clean) >= min_token_len and term_clean.isascii():
            keep.append(i)

    if keep:
        present_names = [present_names[i] for i in keep]
        contrib = contrib[keep]

    contrib = np.asarray(contrib)
    if contrib.size == 0:
        return [], []

    order = np.argsort(contrib)

    # Most positive contributions => misinformation
    fake_idx = order[::-1][:top_k]
    # Most negative contributions => real
    real_idx = order[:top_k]

    top_fake = [(present_names[i], float(contrib[i])) for i in fake_idx if contrib[i] > 0]
    top_real = [(present_names[i], float(contrib[i])) for i in real_idx if contrib[i] < 0]

    return top_fake, top_real

def render_contrib_bars(items, positive=True):
    """
    items: list[(term, contribution)]
    positive=True means contributions are + (misinfo side), we show red-ish bars.
    positive=False means contributions are - (real side), we show green-ish bars.
    """
    if not items:
        st.write("No strong signals found in this text.")
        return

    # scale bars to max absolute within this side
    vals = np.array([abs(v) for _, v in items], dtype=float)
    vmax = float(vals.max()) if vals.size else 1.0

    for term, v in items:
        width_pct = 100.0 * (abs(v) / vmax) if vmax > 0 else 0.0
        sign = "+" if v >= 0 else "−"
        num = f"{sign}{abs(v):.3f}"

        # 3 columns: term | bar | value
        c1, c2, c3 = st.columns([2.2, 4.2, 1.3])
        with c1:
            st.write(term)

        with c2:
            # simple HTML bar; Streamlit theme stays dark
            color = "#ff4b4b" if positive else "#2ecc71"
            track = "rgba(255,255,255,0.12)"
            bar_html = f"""
            <div style="width:100%; height:14px; background:{track}; border-radius:10px; overflow:hidden;">
              <div style="width:{width_pct:.1f}%; height:14px; background:{color}; border-radius:10px;"></div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)

        with c3:
            st.write(num)

# ---------- UI ----------
st.title("🧪 Demo Tool")
st.caption("Paste article text and evaluate misinformation risk using the trained model.")

tfidf, model, model_version = load_artifacts()
manifest = load_manifest()

text = st.text_area(
    "Article text",
    height=220,
    placeholder="Paste an article paragraph or headline here..."
)

col1, col2 = st.columns([1, 1])
with col1:
    run = st.button("Analyze", use_container_width=True)
with col2:
    st.markdown(f"**Model:** TF-IDF + Logistic Regression ({model_version})")

if manifest:
    st.caption(
        f"Split policy: `{manifest.get('split_strategy', 'unknown')}` | "
        f"Validation-selected threshold: `{manifest.get('selected_threshold', 'n/a')}`"
    )

# Optional: keep thresholds visible to user (still simple)
with st.expander("Advanced (thresholds)", expanded=False):
    st.write("These thresholds convert probability → risk level / decision.")
    t_medium = st.slider("Medium Risk threshold", 0.05, 0.95, 0.40, 0.01)
    t_high = st.slider("High Risk threshold", 0.05, 0.99, 0.70, 0.01)

def risk_label_custom(p: float) -> str:
    if p >= t_high:
        return "High Risk"
    if p >= t_medium:
        return "Medium Risk"
    return "Low Risk"

def decision_label_custom(p: float) -> str:
    return "Flag for Review" if p >= t_medium else "Monitor"

# ---------- Run ----------
if run:
    if not text.strip():
        st.warning("Please paste some text first.")
    else:
        X = tfidf.transform([text])
        p = float(model.predict_proba(X)[0, 1])

        top_fake, top_real = explain_prediction(
            tfidf=tfidf,
            model=model,
            X_row=X,
            top_k=7,
            min_token_len=3,   # set to 2 if you want to allow "tv", "us", etc.
        )

        st.subheader("Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Misinformation Probability", f"{p:.2f}")
        c2.metric("Risk Level", risk_label_custom(p))
        c3.metric("Decision", decision_label_custom(p))

        st.subheader("Why this prediction?")
        left, right = st.columns(2)

        with left:
            st.markdown("**Signals pushing toward misinformation**")
            render_contrib_bars(top_fake, positive=True)

        with right:
            st.markdown("**Signals pushing toward real news**")
            render_contrib_bars(top_real, positive=False)

        st.caption(
            "Contributions are TF-IDF weights × logistic regression coefficients (log-odds). "
            "Only terms present in your input are shown."
        )
