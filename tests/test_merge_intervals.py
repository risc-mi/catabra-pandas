#  Copyright (c) 2024. RISC Software GmbH.
#  All rights reserved.

from typing import Optional

import numpy as np
import pandas as pd
import pytest

from catabra_pandas import merge_intervals

from .util import create_random_series


def _create_random(
    rng: np.random.RandomState,
    n: int,
    n_groups: int = None,
    interval_dtype: str = "float",
    group_dtype: str = "int",
    empty_frac: float = 0.1,
) -> pd.DataFrame:
    if n_groups is None:
        n_groups = 2 if n < 30 else rng.randint(n // 20, n // 10)
    groups = rng.randint(0, n_groups, size=n)
    if group_dtype == "float":
        groups = groups.astype(np.float64) * np.pi
    elif group_dtype == "category":
        groups = pd.Categorical.from_codes(groups, categories=["cat_" + str(i) for i in range(n_groups)])
    elif group_dtype == "str":
        groups = groups.astype(str)

    a = create_random_series(n, interval_dtype, rng)
    b = create_random_series(n, interval_dtype, rng)
    start = np.minimum(a, b)
    stop = np.maximum(a, b)

    df = pd.DataFrame(data=dict(group=groups, start=start, stop=stop, idx=np.arange(n, dtype=int)))

    empty_mask = rng.uniform(0, 1, size=len(df)) < empty_frac
    if empty_mask.any():
        x = df.loc[empty_mask, "start"]
        df.loc[empty_mask, "start"] = df.loc[empty_mask, "stop"]
        df.loc[empty_mask, "stop"] = x

    return df


def _merge_intervals_slow(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on=None,
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    left_start: Optional[str] = None,
    left_stop: Optional[str] = None,
    right_start: Optional[str] = None,
    right_stop: Optional[str] = None,
) -> pd.DataFrame:
    assert all(c in (None, "start", "stop") for c in (left_start, left_stop, right_start, right_stop))

    if left_start is None:
        mask = np.ones(len(left), dtype=bool)
    else:
        mask = left[left_start].notna().values
    if left_stop is not None:
        mask &= left[left_stop].notna().values
    if left_start is not None and left_stop is not None and left_start != left_stop:
        mask &= (left[left_start] <= left[left_stop]).values
    left = left[mask]

    if right_start is None:
        mask = np.ones(len(right), dtype=bool)
    else:
        mask = right[right_start].notna().values
    if right_stop is not None:
        mask &= right[right_stop].notna().values
    if right_start is not None and right_stop is not None and right_start != right_stop:
        mask &= (right[right_start] <= right[right_stop]).values
    right = right[mask]

    df = pd.merge(
        left,
        right,
        how="inner",
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
    )

    if (
        (left_start is None and left_stop is None)
        or (right_start is None and right_stop is None)
        or (left_start is None and right_start is None)
        or (left_stop is None and right_stop is None)
    ):
        pass
    elif left_start == left_stop and right_start == right_stop:
        df = df[df[left_start + "_x"] == df[right_start + "_y"]]
    elif right_stop is None:
        df = df[df[left_stop + "_x"] >= df[right_start + "_y"]]
    elif right_start is None:
        df = df[df[left_start + "_x"] <= df[right_stop + "_y"]]
    elif left_start == left_stop:
        df = df[df[left_start + "_x"].between(df[right_start + "_y"], df[right_stop + "_y"])]
    elif left_stop is None:
        df = df[df[right_stop + "_y"] >= df[left_start + "_x"]]
    elif left_start is None:
        df = df[df[right_start + "_y"] <= df[left_stop + "_x"]]
    elif right_start == right_stop:
        df = df[df[right_start + "_y"].between(df["start_x"], df["stop_x"])]
    else:
        # proper interval overlaps
        df = df[df["start_x"].between(df["start_y"], df["stop_y"]) | df["start_y"].between(df["start_x"], df["stop_x"])]

    return df


def test_open_closed(seed: int = 0):
    rng = np.random.RandomState(seed)

    left = (
        pd.MultiIndex.from_product([list(range(4))] * 2)
        .to_frame()
        .reset_index(drop=True)
        .set_axis(["start", "stop"], axis=1)
    )
    left["i"] = np.arange(len(left))
    right = left.copy()

    cross = pd.merge(left, right, how="cross")

    for include_left_start in (False, True):
        for include_left_stop in (False, True):
            for include_right_start in (False, True):
                for include_right_stop in (False, True):
                    lperm = rng.permutation(len(left))
                    rperm = rng.permutation(len(right))
                    kwargs = dict(
                        left=left.iloc[lperm],
                        right=right.iloc[rperm],
                        how="inner",
                        left_start="start",
                        left_stop="stop",
                        right_start="start",
                        right_stop="stop",
                        include_left_start=include_left_start,
                        include_left_stop=include_left_stop,
                        include_right_start=include_right_start,
                        include_right_stop=include_right_stop,
                        return_indexers=True,
                    )
                    indexers = merge_intervals(**kwargs)
                    assert (indexers[0] >= 0).all()
                    assert (indexers[0] < len(left)).all()
                    assert (indexers[1] >= 0).all()
                    assert (indexers[1] < len(right)).all()

                    # simple "factorize"
                    a = lperm[indexers[0]] * len(right) + rperm[indexers[1]]
                    assert len(np.unique(a)) == len(a)

                    # get all empty intervals
                    if include_left_start and include_left_stop:
                        mask = cross["start_x"] > cross["stop_x"]
                    else:
                        mask = cross["start_x"] >= cross["stop_x"]
                    if include_right_start and include_right_stop:
                        mask |= cross["start_y"] > cross["stop_y"]
                    else:
                        mask |= cross["start_y"] >= cross["stop_y"]

                    # get all *disjoint* intervals
                    if include_left_stop and include_right_start:
                        mask |= cross["stop_x"] < cross["start_y"]
                    else:
                        mask |= cross["stop_x"] <= cross["start_y"]
                    if include_right_stop and include_left_start:
                        mask |= cross["stop_y"] < cross["start_x"]
                    else:
                        mask |= cross["stop_y"] <= cross["start_x"]

                    # invert to get all overlapping intervals
                    mask = ~mask

                    b = cross.loc[mask, "i_x"].values * len(right) + cross.loc[mask, "i_y"].values

                    false_positives = np.setdiff1d(a, b)
                    false_negatives = np.setdiff1d(b, a)

                    assert len(false_positives) == 0
                    assert len(false_negatives) == 0


def test_how():
    left = pd.DataFrame(data=dict(start=[0, 7, 1, 8], stop=[2, 8, 5, 9]))
    right = pd.DataFrame(data=dict(start=[10, 4, 0], stop=[11, 5, 3]))
    kwargs = dict(left_start="start", left_stop="stop", right_start="start", right_stop="stop")

    # inner
    indexer = merge_intervals(left, right, how="inner", return_indexers=True, **kwargs)
    assert indexer.shape == (2, 3)
    assert (indexer == np.array([[0, 2, 2], [2, 1, 2]])).all()
    df = merge_intervals(left, right, how="inner", return_indexers=False, **kwargs)
    assert len(df) == 3

    # left
    indexer = merge_intervals(left, right, how="left", return_indexers=True, **kwargs)
    assert indexer.shape == (2, 5)
    assert (indexer == np.array([[0, 1, 2, 2, 3], [2, -1, 1, 2, -1]])).all()
    df = merge_intervals(left, right, how="left", return_indexers=False, **kwargs)
    assert len(df) == 5

    # right
    indexer = merge_intervals(left, right, how="right", return_indexers=True, **kwargs)
    assert indexer.shape == (2, 4)
    assert (indexer == np.array([[-1, 2, 0, 2], [0, 1, 2, 2]])).all()
    df = merge_intervals(left, right, how="right", return_indexers=False, **kwargs)
    assert len(df) == 4

    # outer
    indexer = merge_intervals(left, right, how="outer", return_indexers=True, **kwargs)
    assert indexer.shape == (2, 6)
    assert (indexer == np.array([[0, 1, 2, 2, 3, -1], [2, -1, 1, 2, -1, 0]])).all()
    df = merge_intervals(left, right, how="outer", return_indexers=False, **kwargs)
    assert len(df) == 6


@pytest.mark.parametrize(
    argnames=["n", "seed"],
    argvalues=[
        (1000, 42),
        (1000, 1234),
        (1000, 2904),
        (1000, 98765),
        (1000, 34),
        (1000, 753159),
        (1000, 37008),
        (1000, 115),
        (1000, 0),
        (1000, 6657),
        (1000, 90270),
    ],
)
def test_start_stop(n: int, seed: int):
    rng = np.random.RandomState(seed)

    for left_start, left_stop, right_start, right_stop in [
        (None, None, None, None),
        (None, None, None, "stop"),
        (None, "stop", "start", None),
        (None, None, "start", "stop"),
        (None, "stop", None, "stop"),
        ("start", None, None, "stop"),
        ("start", None, "start", "stop"),
        ("start", "start", "stop", "stop"),
        ("stop", "stop", "start", "stop"),
        ("start", "stop", "start", "stop"),
    ]:
        kwargs = dict(left_start=left_start, left_stop=left_stop, right_start=right_start, right_stop=right_stop)
        left = _create_random(rng, n)
        right = _create_random(rng, n)

        out = merge_intervals(left, right, on="group", how="inner", **kwargs)
        gt = _merge_intervals_slow(left, right, on="group", **kwargs)

        assert len(out) == len(gt), kwargs
        assert (out["idx_x"].values == gt["idx_x"].values).all(), kwargs
        assert (out["idx_y"].values == gt["idx_y"].values).all(), kwargs


@pytest.mark.parametrize(
    argnames=["n", "seed"],
    argvalues=[
        (1000, 24),
        (1000, 4321),
        (1000, 4092),
        (1000, 56789),
        (1000, 43),
        (1000, 953157),
        (1000, 80073),
        (1000, 511),
        (1000, 1),
        (1000, 7566),
        (1000, 7209),
    ],
)
def test_on(n: int, seed: int):
    rng = np.random.RandomState(seed)

    for left_index in (False, True):
        for right_index in (False, True):
            dtype = rng.choice(["int", "float", "category", "str"])
            left = _create_random(rng, n, group_dtype=dtype)
            right = _create_random(rng, n, group_dtype=dtype)
            kwargs = dict(left_start="start", left_stop="stop", right_start="start", right_stop="stop")

            if left_index:
                left.set_index("group", inplace=True)
                kwargs["left_index"] = True
            else:
                left.rename({"group": "lgroup"}, axis=1, inplace=True)
                kwargs["left_on"] = "lgroup" if rng.randint(2) == 0 else ["lgroup"]
            if right_index:
                right.set_index("group", inplace=True)
                kwargs["right_index"] = True
            else:
                right.rename({"group": "rgroup"}, axis=1, inplace=True)
                kwargs["right_on"] = "rgroup" if rng.randint(2) == 0 else ["rgroup"]

            out = merge_intervals(left, right, how="inner", **kwargs)
            gt = _merge_intervals_slow(left, right, **kwargs)

            assert len(out) == len(gt), kwargs
            assert (out["idx_x"].values == gt["idx_x"].values).all(), kwargs
            assert (out["idx_y"].values == gt["idx_y"].values).all(), kwargs


@pytest.mark.parametrize(
    argnames=["n", "seed"],
    argvalues=[
        (1000, 13579),
        (1000, 24680),
        (1000, 37195),
        (1000, 84206),
        (1000, 51397),
        (1000, 60482),
    ],
)
def test_dtypes(n: int, seed: int):
    rng = np.random.RandomState(seed)
    kwargs = dict(on="group", left_start="start", left_stop="stop", right_start="start", right_stop="stop")

    for dtype in ("float", "int", "timedelta", "timestamp"):
        left = _create_random(rng, n * 2, interval_dtype=dtype)
        right = _create_random(rng, n, interval_dtype=dtype)

        out = merge_intervals(left, right, how="inner", **kwargs)
        gt = _merge_intervals_slow(left, right, **kwargs)

        assert len(out) == len(gt), kwargs
        assert (out["idx_x"].values == gt["idx_x"].values).all(), kwargs
        assert (out["idx_y"].values == gt["idx_y"].values).all(), kwargs


@pytest.mark.parametrize(
    argnames=["n", "seed"],
    argvalues=[
        (100, 39469),
        (100, 7901665),
        (500, 1601641),
        (500, 352009),
        (1000, 701),
        (1000, 126947),
    ],
)
def test_nan_inf(n: int, seed: int):
    rng = np.random.RandomState(seed)
    kwargs = dict(on="group", left_start="start", left_stop="stop", right_start="start", right_stop="stop")
    left = _create_random(rng, n * 10)
    right = _create_random(rng, n)

    left.loc[rng.uniform(0, 1, len(left)) < 0.05, "start"] = np.nan
    left.loc[rng.uniform(0, 1, len(left)) < 0.05, "start"] = -np.inf
    left.loc[rng.uniform(0, 1, len(left)) < 0.05, "stop"] = np.nan
    left.loc[rng.uniform(0, 1, len(left)) < 0.05, "stop"] = np.inf
    right.loc[rng.uniform(0, 1, len(right)) < 0.05, "start"] = np.nan
    right.loc[rng.uniform(0, 1, len(right)) < 0.05, "start"] = -np.inf
    right.loc[rng.uniform(0, 1, len(right)) < 0.05, "stop"] = np.nan
    right.loc[rng.uniform(0, 1, len(right)) < 0.05, "stop"] = np.inf

    out = merge_intervals(left, right, how="inner", **kwargs)
    gt = _merge_intervals_slow(left, right, **kwargs)

    assert len(out) == len(gt), kwargs
    assert (out["idx_x"].values == gt["idx_x"].values).all(), kwargs
    assert (out["idx_y"].values == gt["idx_y"].values).all(), kwargs


def test_keep():
    left = pd.DataFrame(data=dict(start=[4, 4, 0, 2, -7, 50], stop=[5, 6, 0, 2, -4, 52]), dtype=np.int8)
    right = pd.DataFrame(
        data=dict(start=[1, 23, 3, -10, 10, 4, -1, -2, 3, 12, 2], stop=[2, 24, 4, -7, 15, 4, 0, 0, 4, 14, 2]),
        dtype=np.int8,
    )
    kwargs = dict(how="inner", left_start="start", right_stop="stop", return_indexers=True)

    for left_stop in ("start", "stop"):
        for right_start in ("start", "stop"):
            indexer = merge_intervals(left, right, keep="all", left_stop=left_stop, right_start=right_start, **kwargs)
            assert indexer.shape == (2, 11), (left_stop, right_start)
            assert (indexer == np.array([[0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4], [2, 5, 8, 2, 5, 8, 6, 7, 0, 10, 3]])).all()

            indexer = merge_intervals(left, right, keep="first", left_stop=left_stop, right_start=right_start, **kwargs)
            assert indexer.shape == (2, 5), (left_stop, right_start)
            assert (indexer == np.array([[0, 1, 2, 3, 4], [2, 2, 6, 0, 3]])).all()

            indexer = merge_intervals(left, right, keep="last", left_stop=left_stop, right_start=right_start, **kwargs)
            assert indexer.shape == (2, 5), (left_stop, right_start)
            assert (indexer == np.array([[0, 1, 2, 3, 4], [8, 8, 7, 10, 3]])).all()

            indexer = merge_intervals(left, right, keep="both", left_stop=left_stop, right_start=right_start, **kwargs)
            assert indexer.shape == (2, 9), (left_stop, right_start)
            assert (indexer == np.array([[0, 0, 1, 1, 2, 2, 3, 3, 4], [2, 8, 2, 8, 6, 7, 0, 10, 3]])).all()

    # `right` is already sorted
    left = pd.DataFrame(data=dict(start=[-1, -3, 1, 0, 5], stop=[7, -1, 9, 5, 7]), dtype=np.int8)
    right = pd.DataFrame(data=dict(a=np.arange(4, dtype=np.int8) * 2))
    kwargs = dict(how="inner", left_start="start", left_stop="stop", right_start="a", right_stop="a")

    indexer = merge_intervals(left, right, keep="first", return_indexers=True, **kwargs)
    assert indexer.shape == (2, 4)
    assert (indexer == np.array([[0, 2, 3, 4], [0, 1, 0, 3]])).all()

    indexer = merge_intervals(left, right, keep="last", return_indexers=True, **kwargs)
    assert indexer.shape == (2, 4)
    assert (indexer == np.array([[0, 2, 3, 4], [3, 3, 2, 3]])).all()

    indexer = merge_intervals(left, right, keep="both", return_indexers=True, **kwargs)
    assert indexer.shape == (2, 7)
    assert (indexer == np.array([[0, 0, 2, 2, 3, 3, 4], [0, 3, 1, 3, 0, 2, 3]])).all()

    # proper overlap
    left = pd.DataFrame(data=dict(start=[0, -3], stop=[4, -1]), dtype=np.int8)
    right = pd.DataFrame(data=dict(start=[1, -1, 2, -2, 1, -4], stop=[4, 2, 5, 3, 3, -1]))
    kwargs = dict(how="inner", left_start="start", left_stop="stop", right_start="start", right_stop="stop")

    indexer = merge_intervals(left, right, keep="all", return_indexers=True, **kwargs)
    assert indexer.shape == (2, 8)
    assert (indexer == np.array([[0, 0, 0, 0, 0, 1, 1, 1], [0, 1, 2, 3, 4, 1, 3, 5]])).all()

    indexer = merge_intervals(left, right, keep="first", return_indexers=True, **kwargs)
    assert indexer.shape == (2, 2)
    assert (indexer == np.array([[0, 1], [0, 1]])).all()

    indexer = merge_intervals(left, right, keep="last", return_indexers=True, **kwargs)
    assert indexer.shape == (2, 2)
    assert (indexer == np.array([[0, 1], [4, 5]])).all()

    indexer = merge_intervals(left, right, keep="both", return_indexers=True, **kwargs)
    assert indexer.shape == (2, 4)
    assert (indexer == np.array([[0, 0, 1, 1], [0, 4, 1, 5]])).all()


def test_exceptions():
    import warnings

    left = pd.DataFrame(data=dict(start=[0, 7, 1, 8], stop=[2, 8, 5, 9]))
    right = pd.DataFrame(data=dict(start=[10, 4, 0], stop=[11, 5, 3]))

    try:
        merge_intervals(left, right, how=True)
        assert False
    except ValueError:
        pass

    try:
        merge_intervals(left, right, keep="any")
        assert False
    except ValueError:
        pass

    try:
        merge_intervals(left, right, left_index=None)
        assert False
    except ValueError:
        pass

    try:
        merge_intervals(left, right, right_index=None)
        assert False
    except ValueError:
        pass

    try:
        merge_intervals(left, right, left_index=True, left_on="group")
        assert False
    except ValueError:
        pass

    try:
        merge_intervals(left, right, right_index=True, right_on="group")
        assert False
    except ValueError:
        pass

    try:
        merge_intervals(left, right, left_on="group", right_on=["group", "idx"])
        assert False
    except ValueError:
        pass

    try:
        merge_intervals(left, right, on="group", left_on="group")
        assert False
    except ValueError:
        pass

    try:
        merge_intervals(left, right, on="group", right_index=True)
        assert False
    except ValueError:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = merge_intervals(
            left,
            right,
            how="inner",
            left_start="start",
            left_stop="start",
            right_start="start",
            right_stop="stop",
            include_left_start=False,
        )
        assert len(out) == 0

    try:
        merge_intervals(left, right)
        assert False
    except ValueError:
        pass
