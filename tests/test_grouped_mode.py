#  Copyright (c) 2025. RISC Software GmbH.
#  All rights reserved.

import numpy as np
import pandas as pd

from catabra_pandas import grouped_mode


def test_series_single():
    s = pd.Series(data=[2, 0, 1, 1, 2, 1, 0, 3], index=[0, 0, 1, 1, 2, 1, 0, 2])
    df = grouped_mode(s)
    assert len(df) == 3
    assert df.loc[0, "mode"] == 0
    assert df.loc[0, "count"] == 2
    assert df.loc[1, "mode"] == 1
    assert df.loc[1, "count"] == 3
    assert df.loc[2, "mode"] in (2, 3)
    assert df.loc[2, "count"] == 1


def test_series_multi():
    s = pd.Series(
        data=[2, 0, 1, 1, 2, 1, 0, 3],
        index=pd.MultiIndex.from_tuples(
            [(0, "a"), (0, "b"), (1, "a"), (1, "a"), (2, "b"), (1, "b"), (0, "a"), (2, "a")]
        ),
    )

    df = grouped_mode(s)
    assert len(df) == 6
    assert df.loc[(0, "a"), "mode"] in (0, 2)
    assert df.loc[(0, "a"), "count"] == 1
    assert df.loc[(0, "b"), "mode"] == 0
    assert df.loc[(0, "b"), "count"] == 1
    assert df.loc[(1, "a"), "mode"] == 1
    assert df.loc[(1, "a"), "count"] == 2
    assert df.loc[(1, "b"), "mode"] == 1
    assert df.loc[(1, "b"), "count"] == 1
    assert df.loc[(2, "a"), "mode"] == 3
    assert df.loc[(2, "a"), "count"] == 1
    assert df.loc[(2, "b"), "mode"] == 2
    assert df.loc[(2, "b"), "count"] == 1

    df = grouped_mode(s, group_by=0)
    assert len(df) == 3
    assert df.loc[0, "mode"] == 0
    assert df.loc[0, "count"] == 2
    assert df.loc[1, "mode"] == 1
    assert df.loc[1, "count"] == 3
    assert df.loc[2, "mode"] in (2, 3)
    assert df.loc[2, "count"] == 1


def test_df_multi():
    arg = pd.DataFrame(
        data=dict(A=[2, 0, 1, 1, 2, 1, 0, 3]),
        index=pd.MultiIndex.from_tuples(
            [(0, "a"), (0, "b"), (1, "a"), (1, "a"), (2, "b"), (1, "b"), (0, "a"), (2, "a")]
        ),
    )

    # no `column` specified
    try:
        df = grouped_mode(arg)
    except ValueError:
        pass
    else:
        assert False

    df = grouped_mode(arg, column="A")
    assert len(df) == 6
    assert df.loc[(0, "a"), "mode"] in (0, 2)
    assert df.loc[(0, "a"), "count"] == 1
    assert df.loc[(0, "b"), "mode"] == 0
    assert df.loc[(0, "b"), "count"] == 1
    assert df.loc[(1, "a"), "mode"] == 1
    assert df.loc[(1, "a"), "count"] == 2
    assert df.loc[(1, "b"), "mode"] == 1
    assert df.loc[(1, "b"), "count"] == 1
    assert df.loc[(2, "a"), "mode"] == 3
    assert df.loc[(2, "a"), "count"] == 1
    assert df.loc[(2, "b"), "mode"] == 2
    assert df.loc[(2, "b"), "count"] == 1


def test_df_columns():
    arg = pd.DataFrame(data=dict(B=[0, 0, 1, 1, 2, 1, 0, 2], A=[2, 0, 1, 1, 2, 1, 0, 3], C=list("abaabbaa")))

    df = grouped_mode(arg, column="A", group_by=["B", "C"])
    assert len(df) == 6
    assert df.loc[(0, "a"), "mode"] in (0, 2)
    assert df.loc[(0, "a"), "count"] == 1
    assert df.loc[(0, "b"), "mode"] == 0
    assert df.loc[(0, "b"), "count"] == 1
    assert df.loc[(1, "a"), "mode"] == 1
    assert df.loc[(1, "a"), "count"] == 2
    assert df.loc[(1, "b"), "mode"] == 1
    assert df.loc[(1, "b"), "count"] == 1
    assert df.loc[(2, "a"), "mode"] == 3
    assert df.loc[(2, "a"), "count"] == 1
    assert df.loc[(2, "b"), "mode"] == 2
    assert df.loc[(2, "b"), "count"] == 1


def test_empty():
    dtype = pd.CategoricalDtype(categories=list("ABC"), ordered=True)

    df = grouped_mode(pd.Series(index=[], dtype=dtype))
    assert len(df) == 0
    assert df["mode"].dtype == dtype
    assert df["count"].dtype.kind == "i"

    arg = pd.DataFrame(columns=["B", "A", "C"], index=[])
    arg["A"] = arg["A"].astype(dtype)
    df = grouped_mode(arg, column="A", group_by=["B", "C"])
    assert len(df) == 0
    assert df["mode"].dtype == dtype
    assert df["count"].dtype.kind == "i"


def test_nan():
    s = pd.Series(data=[2, 0, None, 1, None, None, 0, None], index=[0, 0, 1, 1, 2, 1, 0, 2], dtype=float)

    df = grouped_mode(s, dropna=True)
    assert len(df) == 3
    assert df.loc[0, "mode"] == 0
    assert df.loc[0, "count"] == 2
    assert df.loc[1, "mode"] == 1
    assert df.loc[1, "count"] == 1
    assert np.isnan(df.loc[2, "mode"])
    assert df.loc[2, "count"] == 0

    df = grouped_mode(s, dropna=False)
    assert len(df) == 3
    assert df.loc[0, "mode"] == 0
    assert df.loc[0, "count"] == 2
    assert np.isnan(df.loc[1, "mode"])
    assert df.loc[1, "count"] == 2
    assert np.isnan(df.loc[2, "mode"])
    assert df.loc[2, "count"] == 2

    s.iloc[:] = None
    df = grouped_mode(s, dropna=True)
    assert len(df) == 3
    assert df["mode"].isna().all()
    assert (df["count"] == 0).all()

    df = grouped_mode(s, dropna=False)
    assert len(df) == 3
    assert df["mode"].isna().all()
    assert (df["count"] > 0).all()


def test_nan_groupby():
    s = pd.Series(data=[2, 0, None, 1, 4, None, 0, 4], index=[0, 0, 1, 1, None, 1, 0, None], dtype=float)

    df = grouped_mode(s, dropna=True)
    assert len(df) == 2
    assert df.loc[0, "mode"] == 0
    assert df.loc[0, "count"] == 2
    assert df.loc[1, "mode"] == 1
    assert df.loc[1, "count"] == 1

    df = grouped_mode(s, dropna=False)
    assert len(df) == 2
    assert df.loc[0, "mode"] == 0
    assert df.loc[0, "count"] == 2
    assert np.isnan(df.loc[1, "mode"])
    assert df.loc[1, "count"] == 2
