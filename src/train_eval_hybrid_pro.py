#!/usr/bin/env python3
"""Train and evaluate a hybrid (text + metadata) PolitiFact baseline with deployment-safe evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, Tuple
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler


REAL_PATH = "data_fakenewsnet/raw/politifact/politifact_real.csv"
FAKE_PATH = "data_fakenewsnet/raw/politifact/politifact_fake.csv"


def clean_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def source_domain(url: str) -> str:
    candidate = str(url or "").strip().lower()
    if not candidate:
        return "unknown"
    try:
        netloc = urlparse(candidate).netloc
    except ValueError:
        return "unknown"
    if not netloc:
        return "unknown"
    return netloc.replace("www.", "")


def tweet_count(value) -> int:
    if pd.isna(value):
        return 0
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return 0
    return len([token for token in s.split("\t") if token.strip()])


def load_hybrid_dataset(real_path: str, fake_path: str) -> pd.DataFrame:
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Missing file: {real_path}")
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Missing file: {fake_path}")

    real_df = pd.read_csv(real_path)
    fake_df = pd.read_csv(fake_path)
    real_df["label"] = 0
    fake_df["label"] = 1
    df = pd.concat([real_df, fake_df], ignore_index=True)

    df["clean_text"] = df.get("title", "").fillna("").astype(str).apply(clean_text)
    df["tweet_count"] = df.get("tweet_ids", "").apply(tweet_count)
    df["has_tweets"] = (df["tweet_count"] > 0).astype(int)
    df["log_tweet_count"] = np.log1p(df["tweet_count"])
    df["title_len"] = df.get("title", "").fillna("").astype(str).str.len()
    df["url_len"] = df.get("news_url", "").fillna("").astype(str).str.len()
    df["source_domain"] = df.get("news_url", "").apply(source_domain)

    out_cols = [
        "clean_text",
        "tweet_count",
        "has_tweets",
        "log_tweet_count",
        "title_len",
        "url_len",
        "label",
        "source_domain",
    ]
    out = df[out_cols].copy()
    out = out[(out["clean_text"].str.len() > 0) & out["label"].isin([0, 1])].reset_index(drop=True)

    for col in ["tweet_count", "has_tweets", "log_tweet_count", "title_len", "url_len"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out


@dataclass
class SplitOutput:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    split_strategy: str


def _group_split_indices(
    y: pd.Series,
    groups: pd.Series,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    return next(gss.split(X=np.zeros(len(y)), y=y, groups=groups))


def split_train_val_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> SplitOutput:
    y = df["label"]
    groups = df["source_domain"].fillna("unknown")

    usable_groups = groups.nunique() >= 10 and (groups != "unknown").mean() > 0.3

    if usable_groups:
        train_val_idx, test_idx = _group_split_indices(
            y=y,
            groups=groups,
            test_size=test_size,
            random_state=random_state,
        )

        sub_df = df.iloc[train_val_idx].reset_index(drop=True)
        sub_y = sub_df["label"]
        sub_groups = sub_df["source_domain"].fillna("unknown")

        sub_train_idx, sub_val_idx = _group_split_indices(
            y=sub_y,
            groups=sub_groups,
            test_size=val_size,
            random_state=random_state + 1,
        )

        return SplitOutput(
            train_idx=train_val_idx[sub_train_idx],
            val_idx=train_val_idx[sub_val_idx],
            test_idx=test_idx,
            split_strategy="group_by_source_domain",
        )

    train_val_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        random_state=random_state + 1,
        stratify=df.iloc[train_val_idx]["label"],
    )

    return SplitOutput(
        train_idx=np.array(train_idx),
        val_idx=np.array(val_idx),
        test_idx=np.array(test_idx),
        split_strategy="stratified_random_fallback",
    )


def evaluate_at_threshold(y_true: Iterable[int], y_prob: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(list(y_true))
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def choose_threshold_by_f1(y_true: Iterable[int], y_prob: np.ndarray) -> Tuple[float, pd.DataFrame]:
    thresholds = np.round(np.arange(0.1, 0.95, 0.05), 2)
    rows = [evaluate_at_threshold(y_true, y_prob, t) for t in thresholds]
    table = pd.DataFrame(rows).sort_values(by=["f1", "recall"], ascending=[False, False])
    return float(table.iloc[0]["threshold"]), table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate hybrid professional baseline")
    parser.add_argument("--real-path", default=REAL_PATH)
    parser.add_argument("--fake-path", default=FAKE_PATH)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--outputs-dir", default="outputs/professional_eval")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--class-weight", default="balanced")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.outputs_dir, exist_ok=True)

    df = load_hybrid_dataset(args.real_path, args.fake_path)
    split = split_train_val_test(df=df, test_size=0.2, val_size=0.2, random_state=args.random_state)

    meta_cols = ["tweet_count", "has_tweets", "log_tweet_count", "title_len", "url_len"]
    train_df = df.iloc[split.train_idx]
    val_df = df.iloc[split.val_idx]
    test_df = df.iloc[split.test_idx]

    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=args.max_features,
        min_df=args.min_df,
    )
    X_text_train = tfidf.fit_transform(train_df["clean_text"])
    X_text_val = tfidf.transform(val_df["clean_text"])
    X_text_test = tfidf.transform(test_df["clean_text"])

    scaler = StandardScaler()
    X_meta_train = scaler.fit_transform(train_df[meta_cols])
    X_meta_val = scaler.transform(val_df[meta_cols])
    X_meta_test = scaler.transform(test_df[meta_cols])

    X_train = hstack([X_text_train, csr_matrix(X_meta_train)])
    X_val = hstack([X_text_val, csr_matrix(X_meta_val)])
    X_test = hstack([X_text_test, csr_matrix(X_meta_test)])

    class_weight = None if args.class_weight.lower() == "none" else args.class_weight
    model = LogisticRegression(max_iter=args.max_iter, class_weight=class_weight, random_state=args.random_state)
    model.fit(X_train, train_df["label"])

    y_val_prob = model.predict_proba(X_val)[:, 1]
    best_threshold, val_table = choose_threshold_by_f1(val_df["label"], y_val_prob)

    y_test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_at_threshold(test_df["label"], y_test_prob, best_threshold)

    tfidf_path = os.path.join(args.models_dir, "hybrid_tfidf_pro.joblib")
    scaler_path = os.path.join(args.models_dir, "hybrid_scaler_pro.joblib")
    model_path = os.path.join(args.models_dir, "hybrid_lr_model_pro.joblib")
    manifest_path = os.path.join(args.models_dir, "model_manifest_hybrid_pro.json")
    val_table_path = os.path.join(args.outputs_dir, "hybrid_validation_threshold_table.csv")
    test_metrics_path = os.path.join(args.outputs_dir, "hybrid_test_metrics.json")

    joblib.dump(tfidf, tfidf_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(model, model_path)
    val_table.to_csv(val_table_path, index=False)

    manifest = {
        "dataset": "FakeNewsNet PolitiFact CSV",
        "pipeline": "hybrid_text_plus_metadata_pro",
        "metadata_feature_columns": meta_cols,
        "n_rows": int(len(df)),
        "split_strategy": split.split_strategy,
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "vectorizer": {
            "max_features": args.max_features,
            "min_df": args.min_df,
            "ngram_range": [1, 2],
            "stop_words": "english",
        },
        "model": {
            "type": "LogisticRegression",
            "max_iter": args.max_iter,
            "class_weight": class_weight,
            "random_state": args.random_state,
        },
        "selected_threshold": best_threshold,
        "test_metrics": test_metrics,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print("Saved artifacts:")
    print(f"- {tfidf_path}")
    print(f"- {scaler_path}")
    print(f"- {model_path}")
    print(f"- {manifest_path}")
    print(f"- {val_table_path}")
    print(f"- {test_metrics_path}")
    print(f"Split strategy: {split.split_strategy}")
    print(f"Selected threshold (from validation): {best_threshold}")
    print(f"Test F1: {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
