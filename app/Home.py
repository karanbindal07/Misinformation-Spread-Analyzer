import streamlit as st
import os

st.set_page_config(
    page_title="Misinformation Risk Triage",
    page_icon="🛰️",
    layout="wide",
)

# HERO SECTION
st.title("🛰️ Misinformation Risk Triage")
st.subheader(
    "Understanding signal sources in misinformation detection: text vs diffusion features"
)

st.markdown(
"""
This project investigates **where predictive signal originates** in misinformation detection.

We compare three modeling strategies:

• **Metadata Models** – social diffusion signals  
• **Text Models** – linguistic content patterns  
• **Hybrid Models** – combined features

The goal is to understand **which signals actually drive prediction performance.**
"""
)

st.divider()

# FEATURE CARDS
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🧪 Demo Tool")
    st.write(
        "Paste article text and evaluate misinformation risk using the trained model."
    )

with col2:
    st.markdown("### 📊 Methodology")
    st.write(
        "Explore the datasets, feature engineering pipeline, and modeling decisions."
    )

with col3:
    st.markdown("### 📈 Results")
    st.write(
        "Compare performance of metadata, text, and hybrid models."
    )

st.divider()

# PIPELINE SECTION
st.markdown("## Modeling Pipeline")

st.markdown(
"""
Data Sources  
↓  
Feature Engineering  
↓  
Text Model | Metadata Model  
↓  
Hybrid Model  
↓  
Evaluation & Threshold Optimization  
↓  
Interactive Demo
"""
)

st.info("Use the sidebar to explore the Demo, Methodology, and Results pages.")

if os.path.exists("models/model_manifest_pro.json"):
    st.success(
        "Professional pipeline artifacts detected. Demo and Results pages now reflect the upgraded backend."
    )
