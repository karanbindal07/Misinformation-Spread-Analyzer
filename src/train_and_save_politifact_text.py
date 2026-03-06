import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# load processed dataset
df = pd.read_csv("data_fakenewsnet/processed/politifact_text_processed.csv")

X = df["clean_text"]
y = df["label"]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF vectorizer
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1,2)
)

X_train_tfidf = tfidf.fit_transform(X_train)

# logistic regression model
model = LogisticRegression(max_iter=3000)
model.fit(X_train_tfidf, y_train)

# save models
joblib.dump(tfidf, "models/tfidf_model.joblib")
joblib.dump(model, "models/lr_model.joblib")

print("Models saved successfully.")