import streamlit as st
import json
import os
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="Results",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Results")
st.caption("Real model comparison results from the misinformation detection experiments.")
st.info(
    "Evaluation emphasis: discrimination and triage utility under held-out testing. "
    "These metrics do not imply autonomous fact-checking reliability."
)

pro_manifest_path = "models/model_manifest_pro.json"
pro_metrics_path = "outputs/professional_eval/test_metrics.json"
meta_metrics_path = "outputs/professional_eval/metadata_test_metrics.json"
hybrid_metrics_path = "outputs/professional_eval/hybrid_test_metrics.json"
if os.path.exists(pro_manifest_path) and os.path.exists(pro_metrics_path):
    with open(pro_manifest_path, "r", encoding="utf-8") as f:
        pro_manifest = json.load(f)
    with open(pro_metrics_path, "r", encoding="utf-8") as f:
        pro_metrics = json.load(f)

    st.markdown("## 0. Professional Pipeline Run (Latest)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Professional Accuracy", f"{pro_metrics.get('accuracy', 0):.3f}")
    c2.metric("Professional Precision", f"{pro_metrics.get('precision', 0):.3f}")
    c3.metric("Professional Recall", f"{pro_metrics.get('recall', 0):.3f}")
    c4.metric("Professional F1", f"{pro_metrics.get('f1', 0):.3f}")

    st.caption(
        f"Split strategy: `{pro_manifest.get('split_strategy', 'unknown')}` | "
        f"Validation-selected threshold: `{pro_manifest.get('selected_threshold', 'n/a')}`"
    )
    st.markdown("---")

if os.path.exists(pro_metrics_path) and os.path.exists(meta_metrics_path) and os.path.exists(hybrid_metrics_path):
    with open(pro_metrics_path, "r", encoding="utf-8") as f:
        text_metrics = json.load(f)
    with open(meta_metrics_path, "r", encoding="utf-8") as f:
        meta_metrics = json.load(f)
    with open(hybrid_metrics_path, "r", encoding="utf-8") as f:
        hybrid_metrics = json.load(f)

    dynamic_rows = [
        {"Model": "Text-only (Pro)", "Accuracy": text_metrics["accuracy"], "Precision": text_metrics["precision"], "Recall": text_metrics["recall"], "F1": text_metrics["f1"]},
        {"Model": "Metadata-only (Pro)", "Accuracy": meta_metrics["accuracy"], "Precision": meta_metrics["precision"], "Recall": meta_metrics["recall"], "F1": meta_metrics["f1"]},
        {"Model": "Hybrid (Pro)", "Accuracy": hybrid_metrics["accuracy"], "Precision": hybrid_metrics["precision"], "Recall": hybrid_metrics["recall"], "F1": hybrid_metrics["f1"]},
    ]
    st.markdown("## 1. Professional Pipeline Comparison (Current)")
    st.dataframe(pd.DataFrame(dynamic_rows), use_container_width=True)
    st.caption("These values are loaded from the latest script-generated test metric artifacts.")
    st.markdown("---")
st.markdown("## 2. Historical Controlled Benchmark Comparison")

comparison_df = pd.read_csv("outputs/model_comparison/best_threshold_summary.csv")

st.dataframe(
    comparison_df,
    use_container_width=True
)

st.markdown(
    """
This table shows the **best threshold-selected performance** for each model in the controlled
FakeNewsNet PolitiFact comparison.

The comparison focuses on three approaches:

- **Text-only model**
- **Metadata-only model**
- **Hybrid model**
"""
)

st.markdown("---")

st.markdown("## 3. Model Comparison Chart")

comparison_chart = Image.open("outputs/model_comparison/model_comparison_line.png")
st.image(comparison_chart, caption="Comparison of model metrics across the controlled benchmark.", use_container_width=True)

st.markdown("---")

st.markdown("## 4. Precision–Recall Curves")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Text Model")
    st.image(
        "outputs/fakenewsnet_text/pr_curve.png",
        caption="Precision–Recall curve for the text-only model.",
        use_container_width=True
    )

with col2:
    st.markdown("### Metadata Model")
    st.image(
        "outputs/fakenewsnet_metadata/pr_curve.png",
        caption="Precision–Recall curve for the metadata-only model.",
        use_container_width=True
    )

with col3:
    st.markdown("### Hybrid Model")
    st.image(
        "outputs/fakenewsnet_hybrid/pr_curve.png",
        caption="Precision–Recall curve for the hybrid model.",
        use_container_width=True
    )

st.markdown("---")

st.markdown("## 5. Interpretation")

st.markdown(
    """
These results suggest several important conclusions:

### Text features carry the strongest predictive signal
The **text-only model** achieved the highest **F1 score** among the three approaches.
This indicates that **linguistic patterns in article content are highly informative** for misinformation detection.

### Metadata alone is weaker
The **metadata-only model** achieved the weakest performance overall.
Although it captured some signal, its lower precision and F1 indicate that **propagation-style metadata alone is not sufficient** for strong classification performance.

### Hybrid modeling improves some metrics but not all
The **hybrid model** achieved the best **accuracy** and strongest **precision**, but did not surpass the text-only model on F1.
This suggests that metadata can add useful complementary information, but **text remains the dominant signal source** in this benchmark.
"""
)

st.markdown("---")

st.markdown("## 6. Research Takeaway")

st.success(
    "The controlled benchmark suggests that misinformation detection depends primarily on textual content, while metadata provides secondary but potentially complementary signal."
)

st.info("Next: the Limitations page explains what this model still cannot capture and where the project can be improved.")
