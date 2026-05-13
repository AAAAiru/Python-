"""Tiny English placeholder dataset so the pipeline runs before Kaggle files exist."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def write_placeholder_dataset(data_dir: Path, n_each: int = 1200, random_state: int = 42) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)

    dep_templates = [
        "i feel empty and exhausted every day and nothing helps anymore",
        "i cannot sleep and i keep thinking that i am a burden to everyone",
        "nothing makes sense i just want to disappear and stop hurting",
        "i cry all the time and i feel hopeless about the future",
    ]
    ok_templates = [
        "the weather is nice today and i enjoyed a walk in the park",
        "i finished my assignment early and treated myself to good food",
        "hanging out with friends tonight felt relaxing and fun",
        "i started a new hobby and it is challenging but exciting",
    ]

    texts: list[str] = []
    labels: list[str] = []
    for _ in range(n_each):
        t = str(dep_templates[int(rng.integers(0, len(dep_templates)))])
        texts.append(t + " " + "word" * int(rng.integers(0, 5)))
        labels.append("Depression")
    for _ in range(n_each):
        t = str(ok_templates[int(rng.integers(0, len(ok_templates)))])
        texts.append(t + " " + "note" * int(rng.integers(0, 5)))
        labels.append("Normal")

    df = pd.DataFrame({"text": texts, "status": labels})
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    tr, te = train_test_split(df, test_size=0.25, random_state=random_state, stratify=df["status"])
    train_path = data_dir / "synthetic_train.csv"
    test_path = data_dir / "synthetic_test.csv"
    tr.to_csv(train_path, index=False)
    te.to_csv(test_path, index=False)
    return train_path, test_path
