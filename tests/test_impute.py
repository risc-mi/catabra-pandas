#  Copyright (c) 2025. RISC Software GmbH.
#  All rights reserved.

import numpy as np
import pandas as pd
import pytest

from catabra_pandas import impute

from .util import compare_dataframes, create_random_series


@pytest.mark.parametrize(
    argnames=["n_rows", "n_groups", "ordered", "seed"],
    argvalues=[
        (1000, 5, False, 42),
        (1000, 10, False, 1234),
        (1000, 20, True, 2904),
        (1000, 30, True, 98765),
        (1000, 50, False, 34),
        (1000, 100, True, 753159),
        (1000, 200, True, 37008),
        (5000, 4000, True, 115),
        (10000, 5000, True, 0),
        (50000, 2000, True, 6657),
        (100000, 5000, True, 90270),
    ],
)
def test_random(n_rows: int, n_groups: int, ordered: bool, seed: int):
    rng = np.random.RandomState(seed=seed)

    df = pd.DataFrame(
        data=dict(a=create_random_series(n_rows, "float", rng), b=create_random_series(n_rows, "float", rng))
    )
    df.index = rng.randint(0, n_groups, size=len(df))

    if ordered:
        df.sort_index(inplace=True)

    groups = df.index.unique()

    mask = (
        rng.uniform(0, 1, size=len(df))
        < pd.Series(index=groups, data=rng.uniform(0, 1, size=len(groups))).reindex(df.index).values
    )
    df.loc[mask, "a"] = np.nan

    mask = (
        rng.uniform(0, 1, size=len(df))
        < pd.Series(index=groups, data=rng.uniform(0, 0.5, size=len(groups))).reindex(df.index).values
    )
    df.loc[mask, "b"] = np.nan

    # variant 1: no limit
    res = impute(df, method="linear", group_by=0, limit=None, inplace=False)

    gt = df.groupby(level=0).transform(
        lambda g: g.interpolate(method="linear", axis=0, limit=None, limit_area="inside")
    )

    compare_dataframes(res, gt)

    # variant 2: random limit
    limit = rng.randint(1, max(2, n_rows // (5 * n_groups)))

    res = impute(df, method="linear", group_by=0, limit=limit, inplace=False)

    gt = df.groupby(level=0).transform(
        lambda g: g.interpolate(method="linear", axis=0, limit=limit, limit_area="inside")
    )

    compare_dataframes(res, gt)


def test_all_nan():
    gt = pd.DataFrame(
        index=[0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
        data=dict(a=[1.0, 1.0, 2.0, 3.0, 3.0, np.nan, np.nan, np.nan, 8.0, 9.0, 9.0, 10.0, 10.0]),
    )
    df = gt.copy()
    df["a"].values[[0, 2, 4, -2, -3]] = np.nan

    res = impute(df, method="lfill", group_by=0, limit=None, inplace=False)

    compare_dataframes(res, gt)
