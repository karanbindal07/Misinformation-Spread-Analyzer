#!/usr/bin/env python3
"""Train and evaluate a PolitiFact TF-IDF baseline with deployment-safe evaluation choices.

Key improvements over notebook-only workflow:
- group-aware splitting by source domain to reduce source leakage
- explicit train/validation/test protocol
- threshold selection on validation only
- saved metrics + manifest for reproducibility
"""

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split


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


def load_politifact(real_path: str, fake_path: str) -> pd.DataFrame:
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Missing file: {real_path}")
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Missing file: {fake_path}")

    real_df = pd.read_csv(real_path)
    fake_df = pd.read_csv(fake_path)
    real_df["label"] = 0
    fake_df["label"] = 1

    df = pd.concat([real_df, fake_df], ignore_index=True)
    if "title" not in df.columns:
        raise ValueError("Expected column 'title' not found in PolitiFact CSV files.")

    df["title"] = df["title"].fillna("").astype(str)
    df["clean_text"] = df["title"].apply(clean_text)
    df["source_domain"] = df.get("news_url", "").apply(source_domain)

    # Keep only rows with usable signal.
    df = df[df["clean_text"].str.len() > 0].copy()
    df = df[df["label"].isin([0, 1])].copy()

    # If a split has only unknown domains, fallback logic will use stratified splits.
    if "source_domain" not in df:
        df["source_domain"] = "unknown"

    return df[["clean_text", "label", "source_domain"]].reset_index(drop=True)


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

        train_idx = train_val_idx[sub_train_idx]
        val_idx = train_val_idx[sub_val_idx]
        return SplitOutput(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            split_strategy="group_by_source_domain",
        )

    # Fallback when domain grouping is too sparse/noisy.
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
    best = float(table.iloc[0]["threshold"])
    return best, table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate PolitiFact TF-IDF baseline")
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

    df = load_politifact(args.real_path, args.fake_path)
    split = split_train_val_test(
        df,
        test_size=0.2,
        val_size=0.2,
        random_state=args.random_state,
    )

    train_df = df.iloc[split.train_idx]
    val_df = df.iloc[split.val_idx]
    test_df = df.iloc[split.test_idx]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=args.max_features,
        min_df=args.min_df,
    )

    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_val = vectorizer.transform(val_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])

    class_weight = None if args.class_weight.lower() == "none" else args.class_weight
    clf = LogisticRegression(
        max_iter=args.max_iter,
        class_weight=class_weight,
        random_state=args.random_state,
    )
    clf.fit(X_train, train_df["label"])

    y_val_prob = clf.predict_proba(X_val)[:, 1]
    best_threshold, val_table = choose_threshold_by_f1(val_df["label"], y_val_prob)

    y_test_prob = clf.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_at_threshold(test_df["label"], y_test_prob, best_threshold)

    # Persist artifacts + metrics.
    tfidf_path = os.path.join(args.models_dir, "tfidf_model_pro.joblib")
    model_path = os.path.join(args.models_dir, "lr_model_pro.joblib")
    manifest_path = os.path.join(args.models_dir, "model_manifest_pro.json")
    val_table_path = os.path.join(args.outputs_dir, "validation_threshold_table.csv")
    test_metrics_path = os.path.join(args.outputs_dir, "test_metrics.json")

    joblib.dump(vectorizer, tfidf_path)
    joblib.dump(clf, model_path)
    val_table.to_csv(val_table_path, index=False)

    manifest = {
        "dataset": "FakeNewsNet PolitiFact CSV",
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
    print(f"- {model_path}")
    print(f"- {manifest_path}")
    print(f"- {val_table_path}")
    print(f"- {test_metrics_path}")
    print(f"Split strategy: {split.split_strategy}")
    print(f"Selected threshold (from validation): {best_threshold}")
    print(f"Test F1: {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
