# Misinformation Spread Analyzer

This project studies whether misinformation exhibits distinguishable early signals compared to factual content, and whether those signals can be used to predict misinformation risk.

The work is structured as a **multi-stage modeling pipeline**, progressing from simple metadata baselines to text-based and hybrid models. Each stage isolates the contribution of a different signal type (behavioral, linguistic, and social-contextual).

---

## Tech Stack
Python · pandas · numpy · scikit-learn · matplotlib · seaborn · scipy · nltk

---

## Repository Structure
- `data/` — dataset documentation and metadata (raw data ignored)
- `notebooks/` — modeling and evaluation notebooks
- `src/` — reusable data processing and feature engineering scripts
- `outputs/` — saved figures and evaluation artifacts
- `models/` — trained models (ignored if large)

---

## Stage 1 — Metadata-Only Baseline (COVID-19 Twitter Dataset)

**Question:**  
Can misinformation be detected using *only early behavioral metadata*, without reading content?

### Features
- Follower count
- Friend count
- Temporal indicators
- Engineered features:
  - log-transformed counts
  - follower–friend ratios
  - interaction terms
  - coarse time bins

### Model
- Logistic Regression  
- Stratified 80/20 train–test split

### Evaluation
- Precision, recall, F1-score
- Threshold analysis
- Precision–Recall curve

### Key Findings
- Dataset is highly imbalanced (~80/20).
- Precision and recall remain low across thresholds.
- Metadata alone provides limited discriminatory power.

### Conclusion
Early behavioral metadata is insufficient for reliable misinformation detection when content is excluded. This model serves as a **lower-bound baseline** for evaluating the added value of text and hybrid approaches.

---

## Stage 2 — Text-Only Model (ISOT Fake News Dataset)

**Question:**  
How much misinformation detection performance can be achieved using *content alone*?

### Dataset
- ISOT Fake News Dataset
- Two sources:
  - `True.csv`
  - `Fake.csv`
- Cleaned and processed using `src/features_text.py`

### Model
- TF-IDF vectorization
- Logistic Regression classifier

### Evaluation
- Classification report
- Threshold analysis
- Precision–Recall curve

### Key Findings
- Text features provide a strong predictive signal.
- Precision and recall improve substantially over metadata-only models.
- Language patterns alone can distinguish misinformation in this dataset.

### Conclusion
Text-based models significantly outperform behavioral metadata baselines, demonstrating that linguistic features carry strong misinformation signals. This establishes a **content-based benchmark** for later hybrid comparisons.

---

## Stage 3 — Hybrid V0 Model (Text + Text-Derived Metadata, ISOT)

**Question:**  
Do simple non-semantic structural cues add value beyond text alone?

### Dataset
- ISOT Fake News Dataset
- Same processed text as Stage 2 (controlled comparison)

### Features
- TF-IDF text features
- Text-derived structural metadata:
  - document length
  - token counts
  - basic structural statistics

### Model
- Logistic Regression
- Combined sparse TF-IDF features with standardized numeric features

### Evaluation
- Classification report
- Threshold analysis
- Precision–Recall curve

### Key Findings
- Hybrid V0 shows marginal but consistent improvements over text-only.
- Structural cues complement semantic information at higher thresholds.
- Gains are limited without true behavioral metadata.

### Conclusion
Text-derived structural features can modestly enhance content-based detection, but richer diffusion and user-level metadata is required for larger improvements.

---

## Stage 3B — FakeNewsNet (PolitiFact): Controlled Hybrid Experiments

This stage uses a **single dataset containing both text and social-context metadata**, enabling fair comparison between text-only, metadata-only, and hybrid models.

---

### Stage 3B.1 — Metadata-Only Model (FakeNewsNet PolitiFact)

**Question:**  
What signal is contained in social-context metadata alone?

#### Features
- Tweet count
- Engagement indicators
- Basic propagation statistics
- No text features

#### Model
- Logistic Regression on standardized metadata features

#### Key Findings
- Metadata provides a meaningful but weaker signal than text.
- Performance drops relative to text-only models.
- Metadata captures complementary information not present in language.

#### Conclusion
Social-context metadata alone is insufficient, but captures aspects of *how misinformation spreads*, motivating hybrid modeling.

---

### Stage 3B.2 — Text-Only Model (FakeNewsNet PolitiFact)

**Question:**  
How well does content perform in a realistic, expert-labeled setting?

#### Features
- TF-IDF features from article text only

#### Model
- Logistic Regression on sparse TF-IDF representations

#### Key Findings
- Text provides a strong but imperfect signal.
- Performance is lower than ISOT-based models, reflecting a more realistic dataset.
- Establishes a necessary baseline for hybrid comparison.

#### Conclusion
This model quantifies the predictive power of language alone before introducing diffusion and engagement metadata.

---

### Stage 3B.3 — Hybrid Model (Text + Metadata, FakeNewsNet PolitiFact)

**Question:**  
Do social-context signals improve detection beyond content alone?

#### Features
- TF-IDF text features
- Metadata features including:
  - tweet count
  - presence of tweets
  - title length
  - URL length

#### Model
- Logistic Regression on concatenated TF-IDF + standardized metadata features

#### Evaluation
- Classification report
- Threshold analysis
- Precision–Recall curve

#### Key Findings
- Hybrid model improves over metadata-only baselines.
- Gains over text-only models are modest but consistent.
- Diffusion signals add complementary information to content.

#### Conclusion
The hybrid FakeNewsNet model demonstrates that social-context metadata can enhance text-based misinformation detection. While improvements are limited, results motivate future work incorporating richer network-level and user-level diffusion features.

---

## Summary

This project demonstrates, through controlled experiments, that:

- Behavioral metadata alone is weak
- Text provides the strongest single signal
- Hybrid models offer incremental gains
- Diffusion metadata adds complementary but limited information

The staged design emphasizes **interpretability, comparison, and honest evaluation**, reflecting how real-world misinformation systems are built and analyzed.
