#  Copyright (c) 2024. RISC Software GmbH.
#  All rights reserved.

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd


def _find_contained_points(
    intervals: pd.DataFrame,
    points: pd.Series,
    intervals_on: Optional[np.ndarray],
    points_on: Optional[np.ndarray],
    interval_def: List[Tuple[Any, bool, Any, bool]],
) -> List[pd.DataFrame]:
    """Find points contained in one or more intervals.

    Parameters
    ----------
    intervals : pd.DataFrame
        DataFrame containing the intervals. See Notes for requirements `intervals` must satisfy.
    points : pd.Series
        Series containing the points. See Notes for requirements `points` must satisfy.
    intervals_on : array, optional
        If given, restrict matches to rows with identical values in `intervals_on` and `points_on`.
        Note that either both `intervals_on` and `points_on` must be given, or neither.
    points_on : array, optional
        If given, restrict matches to rows with identical values in `intervals_on` and `points_on`.
        Note that either both `intervals_on` and `points_on` must be given, or neither.
    interval_def : list of (str, bool, str, bool)
        Interval definitions, list of 4-tuples `(left, left_closed, right, right_closed)`. `left` and `right` must be
        names of columns in `intervals`, or None.

    Returns
    -------
    out : list of pd.DataFrame
        For each element in `interval_def` a DataFrame with columns "first" and "last", which refer to rows in `points`.
        The values in "first" are always less than or equal to the corresponding values in "last".
        The (unique) row index corresponds to the row index of `intervals`, but possibly with a different number of
        elements in a different order.

    Notes
    -----
    The input needs to satisfy the following requirements, which are tacitly assumed but not checked for performance
    reasons:
    * `intervals` has a unique, single-level integer row index; it does not need to be sorted, though
    * `points` has a single-level integer row index
    * `points` must be sorted for each `points_on`-group
    * `points` must not contain NaN values
    * `intervals` and `points` have exactly one column level
    * `intervals_on` and `points_on` are either None or monotonic increasing arrays with
        * the same length as `intervals` and `points`, respectively,
        * identical Numpy-native dtypes and
        * no NaN values
    * the various interval-defining columns refer to single columns in `intervals`

    On the other hand, this function can deal with NaN values and accepts views into other DataFrames/Series as input.
    """

    type_col = "__type__"
    idx_col = "__idx__"
    point_col = "__point__"
    on = "__on__"

    points_idx = pd.Series(index=points.index, data=np.arange(len(points), dtype=np.float64))

    out = []
    for a, a_closed, b, b_closed in interval_def:
        if a is None:
            if b is None:
                # corner case: this has nothing to do with intervals
                if intervals_on is None:
                    df = pd.DataFrame(
                        index=intervals.index,
                        data=dict(first=points_idx.min().astype(np.int64), last=points_idx.max().astype(np.int64)),
                    )
                else:
                    df = (
                        points_idx.groupby(points_on)
                        .agg(["first", "last"])
                        .astype(np.int64)
                        .reindex(intervals_on, fill_value=-1)
                    )
                    df.index = intervals.index
            else:
                if b_closed:
                    b_type = 1
                    p_type = 0
                else:
                    b_type = 0
                    p_type = 1

                mask = intervals[b].notna()

                if intervals_on is None:
                    lst = [(points, p_type, points_idx), (intervals.loc[mask, b], b_type, None)]
                    df = _build_df(lst if p_type < b_type else lst[::-1], point_col, type_col, idx_col)
                    df.sort_values(point_col, kind="stable", inplace=True)

                    last_idx = df[idx_col].ffill()[df[type_col] == b_type]
                    last_idx.dropna(inplace=True)
                    df = last_idx.to_frame("last")
                    df["first"] = points_idx.min().astype(np.int64)
                else:
                    # construct DataFrame with interval endpoints and points, and make sure it is sorted wrt. `on`
                    on_vals = intervals_on[mask]
                    type_vals = np.full(on_vals.shape, fill_value=b_type, dtype=np.int8)
                    idx_vals = np.full(on_vals.shape, fill_value=np.nan, dtype=np.float64)

                    insert_loc = np.searchsorted(on_vals, points_on, side="left")
                    df = pd.DataFrame(
                        index=np.insert(intervals[mask].index, insert_loc, points.index),
                        data={
                            on: np.insert(on_vals, insert_loc, points_on),
                            point_col: np.insert(intervals.loc[mask, b].values, insert_loc, points.values),
                            type_col: np.insert(
                                type_vals,
                                insert_loc,
                                np.full_like(len(points), fill_value=p_type, dtype=type_vals.dtype),
                            ),
                            idx_col: np.insert(idx_vals, insert_loc, points_idx.values),
                        },
                    )

                    # sort the partially-sorted (wrt. `on`) `df` additionally wrt. `point_col` and `type_col`
                    df = _grouped_lexsort(df, df[on].values, [point_col, type_col])

                    # find last point in each interval
                    last_idx = df.groupby(on)[idx_col].ffill()[df[type_col] == b_type]

                    # find first point in each interval
                    first_idx = df[df[type_col] == p_type].groupby(on)[idx_col].first()
                    first_idx = df.loc[df[type_col] == b_type, [on]].join(first_idx, on=on, how="left")[idx_col]

                    df = last_idx.to_frame("last")
                    df["first"] = first_idx.values
                    df.dropna(axis=0, inplace=True)
                    df["first"] = df["first"].astype(np.int64)

                df["last"] = df["last"].astype(np.int64)
        else:
            if b is None:
                if a_closed:
                    a_type = 0
                    p_type = 1
                else:
                    a_type = 1
                    p_type = 0

                mask = intervals[a].notna()

                if intervals_on is None:
                    lst = [(points, p_type, points_idx), (intervals.loc[mask, a], a_type, None)]
                    df = _build_df(lst if p_type < a_type else lst[::-1], point_col, type_col, idx_col)
                    df.sort_values(point_col, kind="stable", inplace=True)

                    first_idx = df[idx_col].bfill()[df[type_col] == a_type]
                    first_idx.dropna(inplace=True)
                    df = first_idx.to_frame("first")
                    df["last"] = points_idx.max().astype(np.int64)
                else:
                    # construct DataFrame with interval endpoints and points, and make sure it is sorted wrt. `on`
                    on_vals = intervals_on[mask]
                    type_vals = np.full(on_vals.shape, fill_value=a_type, dtype=np.int8)
                    idx_vals = np.full(on_vals.shape, fill_value=np.nan, dtype=np.float64)

                    insert_loc = np.searchsorted(on_vals, points_on, side="left")
                    df = pd.DataFrame(
                        index=np.insert(intervals[mask].index, insert_loc, points.index),
                        data={
                            on: np.insert(on_vals, insert_loc, points_on),
                            point_col: np.insert(intervals.loc[mask, a].values, insert_loc, points.values),
                            type_col: np.insert(
                                type_vals,
                                insert_loc,
                                np.full_like(len(points), fill_value=p_type, dtype=type_vals.dtype),
                            ),
                            idx_col: np.insert(idx_vals, insert_loc, points_idx.values),
                        },
                    )

                    # sort the partially-sorted (wrt. `on`) `df` additionally wrt. `point_col` and `type_col`
                    df = _grouped_lexsort(df, df[on].values, [point_col, type_col])

                    # find first point in each interval
                    first_idx = df.groupby(on)[idx_col].bfill()[df[type_col] == a_type]

                    # find last point in each interval
                    last_idx = df[df[type_col] == p_type].groupby(on)[idx_col].last()
                    last_idx = df.loc[df[type_col] == a_type, [on]].join(last_idx, on=on, how="left")[idx_col]

                    df = first_idx.to_frame("first")
                    df["last"] = last_idx.values
                    df.dropna(axis=0, inplace=True)
                    df["last"] = df["last"].astype(np.int64)

                df["first"] = df["first"].astype(np.int64)
            else:
                if a_closed:
                    if b_closed:
                        a_type = 0
                        b_type = 2
                        p_type = 1
                    else:
                        a_type = 1
                        b_type = 0
                        p_type = 2
                else:
                    if b_closed:
                        a_type = 2
                        b_type = 1
                        p_type = 0
                    else:
                        a_type = 2
                        b_type = 0
                        p_type = 1

                # this removes NaN endpoints
                mask = (intervals[a] <= intervals[b]) if a_closed and b_closed else (intervals[a] < intervals[b])

                if intervals_on is None:
                    # ensure correct order, to enable a stable sort afterward
                    lst = [
                        (intervals.loc[mask, a], a_type, None),
                        (intervals.loc[mask, b], b_type, None),
                        (points, p_type, points_idx),
                    ]
                    df = _build_df([lst[i] for i in np.argsort([a_type, b_type, p_type])], point_col, type_col, idx_col)
                    df.sort_values(point_col, kind="stable", inplace=True)

                    # find first point in each interval
                    first_idx = df[idx_col].bfill()[df[type_col] == a_type]

                    # find last point in each interval
                    last_idx = df[idx_col].ffill()[df[type_col] == b_type]
                else:
                    # construct DataFrame with interval endpoints and points, and make sure it is sorted wrt. `on`
                    index_vals = np.repeat(intervals.index[mask], 2)
                    on_vals = np.repeat(intervals_on[mask], 2)
                    point_vals = np.stack([intervals.loc[mask, a].values, intervals.loc[mask, b].values], 1).flatten()
                    type_vals = a_type + (b_type - a_type) * (np.arange(len(on_vals), dtype=np.int8) % 2)
                    idx_vals = np.full(on_vals.shape, fill_value=np.nan, dtype=np.float64)

                    # Note: searchsorted-insert-combos only give the correct result if *both* arrays passed to
                    #   searchsorted are sorted! This condition is fortunately satisified here.
                    #   Simple counterexample: np.searchsorted(np.array([0, 3]), np.array([2, 1]))
                    insert_loc = np.searchsorted(on_vals, points_on, side="left" if p_type < 2 else "right")
                    df = pd.DataFrame(
                        index=np.insert(index_vals, insert_loc, points.index),
                        data={
                            on: np.insert(on_vals, insert_loc, points_on),
                            point_col: np.insert(point_vals, insert_loc, points.values),
                            type_col: np.insert(
                                type_vals,
                                insert_loc,
                                np.full_like(len(points), fill_value=p_type, dtype=type_vals.dtype),
                            ),
                            idx_col: np.insert(idx_vals, insert_loc, points_idx.values),
                        },
                    )

                    # sort the partially-sorted (wrt. `on`) `df` additionally wrt. `point_col` and `type_col`
                    # strictly speaking, `df` is not necessarily stable wrt. `type_col`: if the right endpoint of one
                    # interval equals the left endpoint of another interval, then the corresponding rows might be in
                    # the wrong order; the relative order of these rows is irrelevant, though
                    df = _grouped_lexsort(
                        df,
                        df[on].values,
                        [point_col, type_col],
                        n_trailing_stable=(0 if p_type == 1 else 1),
                    )

                    # find first point in each interval
                    # super fast, because `groupby().bfill()` takes advantage of `df` being sorted by `on`
                    first_idx = df.groupby(on)[idx_col].bfill()[df[type_col] == a_type]

                    # find last point in each interval
                    # super fast, because `groupby().ffill()` takes advantage of `df` being sorted by `on`
                    last_idx = df.groupby(on)[idx_col].ffill()[df[type_col] == b_type]

                df = (
                    first_idx[first_idx.notna()]
                    .to_frame("first")
                    .join(last_idx[last_idx.notna()].to_frame("last"), how="inner")
                )
                df["first"] = df["first"].astype(np.int64)
                df["last"] = df["last"].astype(np.int64)

        # `df["first"] <= df["last"]` is necessary to capture cases where, for instance, the first point in
        # an interval comes after the interval's endpoint (and the interval is therefore actually empty)
        out.append(df[df["first"].between(0, df["last"])])

    return out


def _grouped_lexsort(
    df: pd.DataFrame,
    groups: Optional[np.ndarray],
    cols: list,
    return_indexer: bool = False,
    n_trailing_stable: int = 0,
    validate_groups: bool = False,
) -> pd.DataFrame:
    """Sort a DataFrame lexicographically by a list of columns, respecting a given group structure.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to sort.
    groups : array, optional
        Group IDs, array with the same length as `df`. If None, there is only one group encompassing all of `df`.
    cols : list
        List of columns to sort by.
    return_indexer : bool, default=False
        Return the sorting indexer instead of the sorted DataFrame. If True, the DataFrame can be sorted by
        `df.iloc[indexer]`.
    n_trailing_stable : int, default=0
        Number of trailing columns in `cols` that are already "stable". This means, whenever two rows belong to the
        same group and have the same values in all but the last `n_trailing_stable` columns in `cols`, then they are
        already ordered correctly wrt. the last `n_trailing_stable` columns as well.
        If `len(cols) <= n_trailing_stable + 1`, a more efficient sorting algorithm can be used.
    validate_groups : bool = False
        If False, `groups` must be monotonic increasing.
        If True, `groups` can be arbitrary.

    Returns
    -------
    pd.DataFrame | array
        Sorted DataFrame or sorting indexer, depending on `return_indexer`.

    Notes
    -----
    `n_trailing_stable` is not equivalent to being sorted by the last `n_trailing_stable` columns. For example, the
    following DataFrame is stable wrt. its last column "C", although the first two rows are not sorted by "C":

        A   B   C
        ---------
        0   1   5
        0   2   3
        1   0   0
        1   0   1
    """

    if len(cols) <= n_trailing_stable or len(df) <= 1:
        return np.arange(len(df)) if return_indexer else df
    else:
        keys = [df[c].values for c in cols[::-1]]

        if groups is None:
            # only one group => sort columns
            if len(keys) == n_trailing_stable + 1:
                indexer = np.argsort(keys[-1], kind=("stable" if n_trailing_stable > 0 else None))
            else:
                indexer = np.lexsort(keys)
        elif validate_groups and not (groups[:-1] <= groups[1:]).all():
            # groups are not monotonic increasing => sort everything
            indexer = np.lexsort(keys + [groups])
        else:
            # groups are monotonic increasing

            mask = groups != np.roll(groups, -1)
            mask[-1] = True
            n_groups = mask.sum()
            if n_groups == 1:
                # only one group => sort columns
                if len(keys) == n_trailing_stable + 1:
                    indexer = np.argsort(keys[-1], kind=("stable" if n_trailing_stable > 0 else None))
                else:
                    indexer = np.lexsort(keys)
            elif n_groups * 5 >= len(df):
                # if there are many small groups, `np.lexsort` is faster
                # group size 5 has been empirically found to be a good cut-off point, at least if all groups are
                # roughly equinumerous
                # if `np.lexsort` were known to be stable, a more efficient solution might be possible taking
                # `n_trailing_stable` into account
                indexer = np.lexsort(keys + [groups])
            else:
                idx = np.flatnonzero(mask)  # indices of last elements in each group
                sizes = np.empty(len(idx), dtype=np.int64)
                sizes[0] = idx[0] + 1
                sizes[1:] = np.diff(idx)
                indexer = np.empty(len(df), dtype=np.int64)
                i = 0
                # although we have a Python loop here, iterating over all unique values in `groups`, this solution is
                # *much* faster than blindly applying `np.lexsort` if the number of unique values is relatively small
                # compared to the total number of elements
                if len(keys) == n_trailing_stable + 1:
                    keys = keys[-1]  # `keys` has been reversed already
                    kind = "stable" if n_trailing_stable > 0 else None
                    for sz in sizes:
                        indexer[i : i + sz] = np.argsort(keys[i : i + sz], kind=kind) + i
                        i += sz
                else:
                    for sz in sizes:
                        indexer[i : i + sz] = np.lexsort([k[i : i + sz] for k in keys]) + i
                        i += sz

        return indexer if return_indexer else df.iloc[indexer]


def _build_df(
    specs: List[Tuple[pd.Series, int, Optional[pd.Series]]], point_col: str, type_col: str, idx_col: str
) -> pd.DataFrame:
    df = pd.concat([p for p, _, _ in specs], axis=0, ignore_index=False).to_frame(point_col)
    df[type_col] = np.full(len(df), fill_value=0, dtype=np.int8)
    df[idx_col] = np.full(len(df), fill_value=np.nan, dtype=np.float64)
    i = 0
    for p, t, s in specs:
        n = len(p)
        df[type_col].values[i : i + n] = t
        if s is not None:
            df[idx_col].values[i : i + n] = s.values
        i += n
    return df
