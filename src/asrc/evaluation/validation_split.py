from __future__ import annotations

import numpy as np
import pandas as pd


def leave_one_group_indices(df: pd.DataFrame, column: str) -> list[tuple[np.ndarray, np.ndarray, float]]:
    splits = []
    values = sorted(df[column].dropna().unique().tolist())
    for value in values:
        test = df[column].to_numpy() == value
        train = ~test
        if train.sum() > 0 and test.sum() > 0:
            splits.append((np.flatnonzero(train), np.flatnonzero(test), value))
    return splits
