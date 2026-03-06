import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="Results",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Results")
st.caption("Real model comparison results from the misinformation detection experiments.")

st.markdown("## 1. Best Model Comparison")

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

st.markdown("## 2. Model Comparison Chart")

comparison_chart = Image.open("outputs/model_comparison/model_comparison_line.png")
st.image(comparison_chart, caption="Comparison of model metrics across the controlled benchmark.", use_container_width=True)

st.markdown("---")

st.markdown("## 3. Precision–Recall Curves")

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

st.markdown("## 4. Key Quantitative Results")

st.markdown(
    """
Using the controlled FakeNewsNet PolitiFact benchmark, the best-performing thresholds produced the following results:

- **Text-only model**
  - Accuracy: **0.8246**
  - Precision: **0.7207**
  - Recall: **0.9302**
  - F1: **0.8122**

- **Hybrid model**
  - Accuracy: **0.8294**
  - Precision: **0.8125**
  - Recall: **0.7558**
  - F1: **0.7831**

- **Metadata-only model**
  - Accuracy: **0.5991**
  - Precision: **0.5067**
  - Recall: **0.8736**
  - F1: **0.6414**
"""
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