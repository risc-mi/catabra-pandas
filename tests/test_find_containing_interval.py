#  Copyright (c) 2024. RISC Software GmbH.
#  All rights reserved.

import numpy as np
import pandas as pd
import pytest

from catabra_pandas import combine_intervals, find_containing_interval

from .util import create_random_data


@pytest.mark.parametrize(
    argnames=["n_rows", "n_groups", "seed"],
    argvalues=[
        (1000, 20, 2904),
        (1000, 50, 34),
        (1000, 200, 37008),
        (5000, 4000, 115),
        (10000, 5000, 0),
        (50000, 2000, 6657),
        (100000, 5000, 90270),
    ],
    ids=["small_1", "small_2", "small_3", "medium_1", "medium_2", "medium_3", "medium_4"],
)
def test_random(n_rows: int, n_groups: int, seed: int):
    intervals, _ = create_random_data(n_rows, 1, n_entities=n_groups, intervals=True, seed=seed)

    # make intervals pairwise disjoint
    # this was necessary with the old version of `find_containing_interval`, but is not necessary anymore
    # it doesn't do any harm either, so we just leave it that way
    intervals = combine_intervals(intervals[["entity", "start", "stop"]], group_by="entity", n_min=1, n_max=1)

    rng = np.random.RandomState(seed=seed)
    gt = pd.Series(rng.randint(0, len(intervals), size=len(intervals) * 10))

    points = intervals.iloc[gt.values].copy()
    points.reset_index(drop=True, inplace=True)
    points["point"] = points["start"] + rng.uniform(0, 1, len(points)) * (points["stop"] - points["start"])
    points.drop(["start", "stop"], axis=1, inplace=True)

    rand = rng.uniform(-1, 1, size=len(gt))
    gt.loc[np.abs(rand) >= 0.9] = -1
    points.loc[rand <= -0.9, "point"] = (
        intervals.groupby("entity")["start"].min().reindex(points.loc[rand <= -0.9, "entity"].values)
        - pd.Timedelta(1, unit="h")
    ).values
    points.loc[rand >= 0.9, "point"] = (
        intervals.groupby("entity")["stop"].max().reindex(points.loc[rand >= 0.9, "entity"].values)
        + pd.Timedelta(1, unit="m")
    ).values

    df = find_containing_interval(points, intervals, "point", which="both", include_stop=False, group_by="entity")

    assert df.shape == (len(gt), 2)
    assert (df.index == gt.index).all()
    assert (df[("first", "point")] == gt).all()
    assert (df[("last", "point")] == gt).all()


def test_infinite():
    intervals = pd.DataFrame(
        data=dict(
            group=[0, 1, 1, 0, 0, 1, 0, 0, 1],
            start=[5.7, 1.9, -np.inf, 478.0, -np.inf, 3.3, -66.1, 143.5, 132.4],
            stop=[6.3, 1.8, 0.5, np.inf, -87.2, 4.0, -56.7, 143.6, 157.7],
        )
    )
    points = pd.DataFrame(
        data=dict(
            group=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            point=[-58746.0, -0.1, 6.0, 143.5, 143.6, 209.9, 492064.523, 0.0, 0.5, 1.85, 144.4, 3079.51],
        )
    )

    df = find_containing_interval(points, intervals, ["point"], group_by="group")

    assert df.shape == (len(points), 1)
    assert (df.index == points.index).all()
    assert (df["point"].values == np.array([4, -1, 0, 7, 7, -1, 3, 2, 2, -1, 8, -1], dtype=np.int64)).all()


@pytest.mark.manual
def test_large():
    test_random(10000000, 1000000, 310056)
