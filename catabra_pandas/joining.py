#  Copyright (c) 2024. RISC Software GmbH.
#  All rights reserved.

from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd


def _find_contained_points(
    intervals: pd.DataFrame,
    points: pd.DataFrame,
    point_col,
    intervals_on,
    points_on,
    interval_def: List[Tuple[Any, bool, Any, bool]],
    intersect: bool,
) -> Tuple[Union[pd.DataFrame, List[pd.DataFrame]], np.ndarray]:
    # tacit assumptions (must be ensured by calling function; not checked here):
    # * intervals and points must have exactly one column level
    # * point_col is a _single_ column in points
    # * intervals_on and points_on are either None or _single_ columns in intervals and points, respectively with
    #   * identical sortable, Numpy-native dtypes and
    #   * no NaN values
    # * point_col, intervals_on, points_on and the various interval-defining columns must have pairwise distinct names
    # * points and intervals are sorted ascending wrt. points_on and intervals_on, respectively, unless they are None

    type_col = "__type__"
    idx_col = "__idx__"
    on = intervals_on

    intervals_index = intervals.index
    intervals.reset_index(drop=True, inplace=True)

    if points_on is None:
        points = points[[point_col]].reset_index(drop=True, inplace=False)
        points.sort_values(point_col, inplace=True)
    else:
        points = _lexsort_partially_sorted(
            points[[points_on, point_col]].reset_index(drop=True, inplace=False),
            [points_on, point_col],
        )
        points.rename({points_on: on}, axis=1, inplace=True)
    mask = points[point_col].notna()
    if not mask.all():
        points = points[mask].copy()
    idx = points.index
    points.reset_index(drop=True, inplace=True)
    points[type_col] = 0
    points[idx_col] = points.index.astype(np.float64)

    out = None if intersect else []
    for a, a_closed, b, b_closed in interval_def:
        if a is None:
            if b is None:
                # corner case: this has nothing to do with intervals
                if on is None:
                    df = pd.DataFrame(
                        index=intervals.index,
                        data=dict(
                            first=points[idx_col].min().astype(np.int64), last=points[idx_col].max().astype(np.int64)
                        ),
                    )
                else:
                    df = points.groupby(on)[idx_col].agg(["first", "last"])
                    df = intervals.join(df, on=on, how="inner")
                    df["first"] = df["first"].astype(np.int64)
                    df["last"] = df["last"].astype(np.int64)
                    df.sort_index(inplace=True)
                    df = df[["first", "last"]]
            else:
                if b_closed:
                    b_type = 1
                    p_type = 0
                else:
                    b_type = 0
                    p_type = 1

                mask = intervals[b].notna()

                if on is None:
                    df2 = intervals.loc[mask, [b]].rename({b: point_col}, axis=1)
                    df2[type_col] = b_type
                    df2[idx_col] = np.nan

                    points[type_col] = p_type

                    df = pd.concat([points, df2] if p_type < b_type else [df2, points], axis=0)
                    df.sort_values(point_col, kind="stable", inplace=True)

                    last_idx = df[idx_col].ffill()[df[type_col] == b_type]
                    last_idx.dropna(inplace=True)
                    df = last_idx.to_frame("last")
                    df["first"] = points[idx_col].min().astype(np.int64)
                else:
                    # construct DataFrame with interval endpoints and points, and make sure it is sorted wrt. `on`
                    on_vals = intervals.loc[mask, on].values
                    type_vals = np.full(on_vals.shape, fill_value=b_type, dtype=np.int8)
                    idx_vals = np.full(on_vals.shape, fill_value=np.nan, dtype=np.float64)

                    insert_loc = np.searchsorted(on_vals, points[on].values, side="left")
                    df = pd.DataFrame(
                        index=np.insert(intervals[mask].index, insert_loc, points.index),
                        data={
                            on: np.insert(on_vals, insert_loc, points[on].values),
                            point_col: np.insert(intervals.loc[mask, b].values, insert_loc, points[point_col].values),
                            type_col: np.insert(
                                type_vals,
                                insert_loc,
                                np.full_like(len(points), fill_value=p_type, dtype=type_vals.dtype),
                            ),
                            idx_col: np.insert(idx_vals, insert_loc, points[idx_col].values),
                        },
                    )

                    # sort the partially-sorted (wrt. `on`) `df` additionally wrt. `point_col` and `type_col`
                    df = _lexsort_partially_sorted(df, [on, point_col, type_col])

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
                df.sort_index(inplace=True)
        else:
            if b is None:
                if a_closed:
                    a_type = 0
                    p_type = 1
                else:
                    a_type = 1
                    p_type = 0

                mask = intervals[a].notna()

                if on is None:
                    df1 = intervals.loc[mask, [a]].rename({a: point_col}, axis=1)
                    df1[type_col] = a_type
                    df1[idx_col] = np.nan

                    points[type_col] = p_type

                    df = pd.concat([points, df1] if p_type < a_type else [df1, points], axis=0)
                    df.sort_values(point_col, kind="stable", inplace=True)

                    first_idx = df[idx_col].bfill()[df[type_col] == a_type]
                    first_idx.dropna(inplace=True)
                    df = first_idx.to_frame("first")
                    df["last"] = points[idx_col].max().astype(np.int64)
                else:
                    # construct DataFrame with interval endpoints and points, and make sure it is sorted wrt. `on`
                    on_vals = intervals.loc[mask, on].values
                    type_vals = np.full(on_vals.shape, fill_value=a_type, dtype=np.int8)
                    idx_vals = np.full(on_vals.shape, fill_value=np.nan, dtype=np.float64)

                    insert_loc = np.searchsorted(on_vals, points[on].values, side="left")
                    df = pd.DataFrame(
                        index=np.insert(intervals[mask].index, insert_loc, points.index),
                        data={
                            on: np.insert(on_vals, insert_loc, points[on].values),
                            point_col: np.insert(intervals.loc[mask, a].values, insert_loc, points[point_col].values),
                            type_col: np.insert(
                                type_vals,
                                insert_loc,
                                np.full_like(len(points), fill_value=p_type, dtype=type_vals.dtype),
                            ),
                            idx_col: np.insert(idx_vals, insert_loc, points[idx_col].values),
                        },
                    )

                    # sort the partially-sorted (wrt. `on`) `df` additionally wrt. `point_col` and `type_col`
                    df = _lexsort_partially_sorted(df, [on, point_col, type_col])

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
                df.sort_index(inplace=True)
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

                if on is None:
                    df1 = intervals.loc[mask, [a]].rename({a: point_col}, axis=1)
                    df1[type_col] = a_type
                    df1[idx_col] = np.nan

                    df2 = intervals.loc[mask, [b]].rename({b: point_col}, axis=1)
                    df2[type_col] = b_type
                    df2[idx_col] = np.nan

                    points[type_col] = p_type

                    # ensure correct order, to enable a stable sort afterward
                    lst = [df1, df2, points]
                    df = pd.concat([lst[i] for i in np.argsort([a_type, b_type, p_type])], axis=0)
                    df.sort_values(point_col, kind="stable", inplace=True)

                    # find first point in each interval
                    first_idx = df[idx_col].bfill()[df[type_col] == a_type]

                    # find last point in each interval
                    last_idx = df[idx_col].ffill()[df[type_col] == b_type]
                else:
                    # construct DataFrame with interval endpoints and points, and make sure it is sorted wrt. `on`
                    index_vals = np.repeat(intervals.index[mask], 2)
                    on_vals = np.repeat(intervals.loc[mask, on].values, 2)
                    point_vals = np.stack([intervals.loc[mask, a].values, intervals.loc[mask, b].values], 1).flatten()
                    type_vals = a_type + (b_type - a_type) * (np.arange(len(on_vals), dtype=np.int8) % 2)
                    idx_vals = np.full(on_vals.shape, fill_value=np.nan, dtype=np.float64)

                    # Note: searchsorted-insert-combos only give the correct result if *both* arrays passed to
                    #   searchsorted are sorted! This condition is fortunately satisified here.
                    #   Simple counterexample: np.searchsorted(np.array([0, 3]), np.array([2, 1]))
                    insert_loc = np.searchsorted(on_vals, points[on].values, side="left" if p_type < 2 else "right")
                    df = pd.DataFrame(
                        index=np.insert(index_vals, insert_loc, points.index),
                        data={
                            on: np.insert(on_vals, insert_loc, points[on].values),
                            point_col: np.insert(point_vals, insert_loc, points[point_col].values),
                            type_col: np.insert(
                                type_vals,
                                insert_loc,
                                np.full_like(len(points), fill_value=p_type, dtype=type_vals.dtype),
                            ),
                            idx_col: np.insert(idx_vals, insert_loc, points[idx_col].values),
                        },
                    )

                    # sort the partially-sorted (wrt. `on`) `df` additionally wrt. `point_col` and `type_col`
                    # strictly speaking, `df` is not necessarily stable wrt. `type_col`: if the right endpoint of one
                    # interval equals the left endpoint of another interval, then the corresponding rows might be in
                    # the wrong order; the relative order of these rows is irrelevant, though
                    df = _lexsort_partially_sorted(
                        df,
                        [on, point_col, type_col],
                        n_trailing_stable=(0 if p_type == 1 else 1),
                    )

                    # find first point in each interval
                    first_idx = df.groupby(on)[idx_col].bfill()[df[type_col] == a_type]

                    # find last point in each interval
                    last_idx = df.groupby(on)[idx_col].ffill()[df[type_col] == b_type]

                df = (
                    first_idx[first_idx.notna()]
                    .to_frame("first")
                    .join(last_idx[last_idx.notna()].to_frame("last"), how="inner")
                )
                df["first"] = df["first"].astype(np.int64)
                df["last"] = df["last"].astype(np.int64)
                df.sort_index(inplace=True)

        # `df["first"] <= df["last"]` is necessary to capture cases where, for instance, the first point in
        # an interval comes after the interval's endpoint (and the interval is therefore actually empty)
        df = df[df["first"] <= df["last"]]

        if intersect:
            if out is None:
                out = df
            else:
                # note: both `out` and `df` have unique integer indexes, so can easily be joined
                out = out.join(df, rsuffix="_r", how="inner")
                out["left"] = np.maximum(out["left"], out["left_r"])
                out["right"] = np.minimum(out["right"], out["right_r"])
                out.drop(["left_r", "right_r"], axis=1, inplace=True)
                out = out[out["left"] <= out["right"]]
        else:
            out.append(df)

    # restore original index
    intervals.index = intervals_index
    if isinstance(out, pd.DataFrame):
        out.index = intervals_index[out.index]
    elif isinstance(out, list):
        for o in out:
            o.index = intervals_index[o.index]

    return out, idx


def _lexsort_partially_sorted(
    df: pd.DataFrame,
    cols: list,
    return_indexer: bool = False,
    n_trailing_stable: int = 0,
) -> pd.DataFrame:
    """Sort a DataFrame lexicographically by a list of columns, assuming it is already sorted by the first column.

    Parameters
    ----------
    df : DataFrame
        DataFrame to sort.
    cols : list
        List of columns to sort by.
    return_indexer : bool, default=False
        Return the sorting indexer instead of the sorted DataFrame. If True, the DataFrame can be sorted by
        `df.iloc[indexer]`.
    n_trailing_stable : int, default=0
        Number of trailing columns in `cols` that are already "stable". This means, whenever two rows have the same
        values in all but the last `n_trailing_stable` columns in `cols`, then they are already ordered correctly wrt.
        the last `n_trailing_stable` columns as well.
        If `len(cols) <= n_trailing_stable + 2`, a more efficient sorting algorithm can be used.

    Returns
    -------
    DataFrame or array
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

    if len(cols) <= n_trailing_stable + 1 or len(df) <= 1:
        return np.arange(len(df)) if return_indexer else df
    else:
        col = cols[0]
        keys = [df[c].values for c in cols[1:][::-1]]

        idx = np.flatnonzero(df[col].values[:-1] != df[col].values[1:])  # indices of last elements in each group
        if len(idx) == 0:
            # only one group => sort remaining columns
            if len(keys) == n_trailing_stable + 1:
                indexer = np.argsort(keys[-1], kind=("stable" if n_trailing_stable > 0 else None))
            else:
                indexer = np.lexsort(keys)
        elif len(idx) * 5 >= len(df):
            # if there are many small groups, `np.lexsort` is faster
            # group size 5 has been empirically found to be a good cut-off point, at least if all groups are roughly
            # equinumerous
            # if `np.lexsort` were known to be stable, a more efficient solution might be possible taking
            # `n_trailing_stable` into account
            indexer = np.lexsort(keys + [df[col].values])
        else:
            sizes = np.empty(len(idx) + 1, dtype=np.int64)
            sizes[0] = idx[0] + 1
            sizes[1:-1] = np.diff(idx)
            sizes[-1] = len(df) - idx[-1]
            indexer = np.empty(len(df), dtype=np.int64)
            i = 0
            # although we have a Python loop here, iterating over all unique values in `col`, this solution is *much*
            # faster than blindly applying `np.lexsort` if the number of unique values is relatively small compared to
            # the total number of elements
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
