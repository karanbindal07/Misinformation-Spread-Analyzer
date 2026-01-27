# Misinformation Spread Analyzer

This project investigates whether misinformation posts exhibit different early spread characteristics than factual posts, and whether early signals can be used to predict misinformation risk.

The work is structured as a multi-stage data science pipeline, progressing from metadata-only baselines to content-based and hybrid models.

---

## Tech Stack
Python, pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, nltk

---

## Repository Structure
- `data/`: dataset metadata and documentation (raw data ignored)
- `notebooks/`: exploratory analysis and modeling notebooks
- `src/`: reusable data processing and feature engineering scripts
- `models/`: saved trained models (ignored if large)
- `outputs/`: figures and evaluation artifacts

---

## Stage 1 — Metadata-Only Baseline Model

This stage establishes a baseline for misinformation detection using **only tweet level metadata**, without incorporating text content.

### Features Used
- `followers`: number of followers of the tweet author  
- `friends`: number of accounts the author follows  
- `time`: temporal indicator of posting time  
- Engineered features:
  - log-transformed counts
  - follower to friend ratios
  - interaction terms
  - coarse time bins

### Target Variable
- `label`
  - `0` = factual post  
  - `1` = misinformation  

### Model
- Logistic Regression  
- Stratified 80/20 train–test split  

### Evaluation
- Precision, recall, F1-score
- Threshold analysis
- Precision–Recall curve

### Key Findings
- The dataset is highly imbalanced (80/20).
- Precision and recall for misinformation remain low across thresholds.
- Metadata alone provides limited predictive signal.

### Conclusion
The metadata only model demonstrates that early behavioral signals are insufficient for reliable misinformation detection when content is excluded. This model serves as a **comparison** for evaluating the added value of text based and hybrid approaches.

---

## Stage 2 — Text-Only Model (ISOT Fake News Dataset)

This stage evaluates how well misinformation can be detected using **content alone**, independent of diffusion or user metadata.

### Dataset
- ISOT Fake News Dataset
- Two files:
  - `True.csv`
  - `Fake.csv`
- Text cleaned and combined using `src/features_text.py`

### Model
- TF-IDF vectorization
- Logistic Regression classifier

### Evaluation
- Classification report
- Threshold analysis
- Precision–Recall curve

### Key Findings
- Text based features provide a strong predictive signal.
- Precision and recall improve substantially compared to metadata-only models.
- Language patterns alone can distinguish misinformation in this dataset.

### Conclusion
Text based models significantly outperform metadata only baselines, indicating that linguistic features carry strong misinformation signals. This model establishes a second benchmark for comparison with the final hybrid model.

---

## Stage 3 — Hybrid V0 Model (Text + Text Derived Metadata)

This stage extends the text-only model by incorporating simple structural features derived from the text itself, testing whether non-semantic cues improve performance beyond language alone.

## Dataset
- ISOT Fake News Dataset
- Same processed text used in Stage 2 to ensure a controlled comparison

## Features
- TF-IDF text features
- Text-derived metadata, including:
- Document length
- Token counts
- Basic structural statistics

## Model
- Logistic Regression classifier
- Combined sparse text features with standardized numeric features

## Evaluation
- Classification report
- Threshold analysis
- Precision–Recall curve

## Key Findings
- Hybrid v0 shows marginal but consistent improvements over the text-only model.
- Structural cues complement semantic information, improving precision at higher thresholds.
- Gains are limited, suggesting that richer behavioral metadata is needed for larger improvements.

## Conclusion
This hybrid model demonstrates that simple text-derived structural features can enhance content-based misinformation detection. While improvements over the text-only model are modest, the results motivate the incorporation of true diffusion and user metadata in subsequent hybrid models.


--- 

## Stage 3B — Metadata-Only Model (FakeNewsNet PolitiFact)

This stage evaluates misinformation detection using metadata signals only, isolating the predictive value of social and engagement metadata without any language information.

## Dataset
- FakeNewsNet (PolitiFact subset)
- Expert verified news articles labeled as fake or real
- Uses metadata derived from social context (e.g., tweet activity)

## Features
- Metadata features only, including:
  - Tweet count
  - Engagement indicators
  - Basic propagation statistics
- No textual content is used in this model

## Model
- Logistic Regression classifier
- Trained solely on standardized metadata features

## Evaluation
- Classification report
- Threshold analysis
- Precision–Recall curve

## Key Findings
- Metadata alone provides a meaningful but weaker signal compared to text-based models.
- Performance drops relative to text-only baselines, indicating limited discriminative power from engagement statistics alone.
- However, metadata captures complementary information not present in language features.

## Conclusion

The metadata-only model establishes a lower-bound baseline for misinformation detection using social context signals. While insufficient on its own, these features capture aspects of how misinformation spreads and gains attention. This motivates the final hybrid model, which combines both textual content and metadata to assess whether diffusion dynamics enhance predictive performance beyond content alone.