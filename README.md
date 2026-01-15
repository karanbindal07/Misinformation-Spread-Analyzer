# Misinformation Spread Analyzer

An end-to-end data science project that analyzes early indicators of misinformation spread on social media.
The pipeline covers data cleaning, feature engineering, baseline modeling, evaluation, and visualization.

## Tech Stack
Python, pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, nltk

## Repository Structure
- `data/raw/`: raw dataset
- `data/processed/`: cleaned datasets for modeling
- `notebooks/`: EDA and modeling notebooks
- `src/`: reusable pipeline scripts
- `models/`: saved trained models
- `outputs/`: figures and metrics


## Baseline Metadata Model 
This step creates a baseline for misinformation detection using only tweet-level metadata, without including text content.

Features used:
followers: number of followers of the tweet author
friends: number of accounts the author follows
time: timestamp related feature

Target variable:
label

0 = factual tweet
1 = misinformation

Model:
Logistic Regression
Stratified 80/20 train–test split

Evaluation:
Accuracy, precision, recall, and F1-score
Threshold analysis to examine precision and recall

Key findings:
The dataset is is very impoloced (80/20)
Precision and recall for misinformation remain low across thresholds
Metadata alone provides limited predictive signal

Conclusion:
This baseline demonstrates that user and metadata are insufficient for reliable misinformation detection. Future work will incorporate tweet text and network-based features to improve performance. This baseline model is used to find out how metadata alone contributes to misinformation detection. It will also be used to compare once to models that do take in texts. 