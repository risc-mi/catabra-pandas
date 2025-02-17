#  Copyright (c) 2025. RISC Software GmbH.
#  All rights reserved.

from typing import Optional, Union

import numpy as np
import pandas as pd


def get_loc(
    index: pd.Index, target, allow_mask: bool = True, allow_slice: bool = True
) -> Union[np.ndarray, int, slice]:
    """Get integer location(s), slice or mask of given target(s) in a given Index.

    Parameters
    ----------
    index : pd.Index
        The index in which to search for the target(s).
    target
        The target(s) to search for. Can be anything that `loc` accepts, including masks, (partial) labels, iterables
        of labels, and slices.
    allow_mask : bool, default=True
        If True, the output may be a boolean mask.
    allow_slice : bool, default=True
        If True, the output may be a slice.

    Returns
    -------
    np.ndarray | int | slice
        Location(s), slice or mask of `target` in `index`. Can be passed to `iloc`.

    See Also
    --------
    pd.Index.get_indexer_for: does not support partial labels, and returns -1 for missing labels
    pd.Index.get_loc: only supports single labels, and does not support masks
    """
    if isinstance(target, tuple) or np.isscalar(target):
        out = index.get_loc(target)
        if isinstance(out, np.ndarray):
            if not allow_mask and out.dtype.kind == "b":
                out = np.flatnonzero(out)
        elif isinstance(out, slice) and not allow_slice:
            out = np.arange(*out.indices(len(index)))
        return out
    elif isinstance(target, np.ndarray) and target.dtype.kind == "b":
        # mask
        if target.ndim == 1:
            if len(target) == len(index):
                return target if allow_mask else np.flatnonzero(target)
            else:
                raise IndexError("Boolean index has wrong length: {} instead of {}".format(len(target), len(index)))
    elif isinstance(target, list) and all(isinstance(v, bool) for v in target):
        # mask
        if len(target) == len(index):
            return np.asarray(target) if allow_mask else np.flatnonzero(target)
        else:
            raise IndexError("Boolean index has wrong length: {} instead of {}".format(len(target), len(index)))

    # fall-back strategy (slow)
    out = pd.Series(index=index, data=pd.RangeIndex(len(index))).loc[target]
    if isinstance(out, pd.Series):
        out = out.values
    return out


def iloc_loc(df: pd.DataFrame, rows, cols):
    """Get a DataFrame slice from a single or multiple `iloc` row indices and `loc` column labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to slice.
    rows
        Row specification, anything that `iloc` accepts, e.g., integers, iterables of integers, or slices.
    cols
        Column specification, anything that `loc` accepts, e.g., a single label, an iterable of labels, or partial
        labels if `df` has a Multi-column-Index.

    Returns
    -------
    DataFrame slice, can be a DataFrame, a Series, or a scalar.

    See Also
    --------
    iloc_loc_assign
    get_loc
    """
    return df.iloc[rows, get_loc(df.columns, cols)]


def iloc_loc_assign(df: pd.DataFrame, rows, cols, value):
    """Assign a value to a slice of a DataFrame specified by `iloc` row indices and `loc` column labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to modify.
    rows
        Row specification, anything that `iloc` accepts, e.g., integers, iterables of integers, or slices.
    cols
        Column specification, anything that `loc` accepts, e.g., a single label, an iterable of labels, or partial
        labels if `df` has a Multi-column-Index.
    value
        Target value, anything with suitable data type and broadcastable shape.

    Returns
    -------
    `value`, to enable chained assignments as in

        iloc_loc_assign(df1, rows, cols, iloc_loc_assign(df2, rows, cols, value))

    or

        df1.iloc[rows] = iloc_loc_assign(df2, rows, cols, value)

    See Also
    --------
    iloc_loc
    get_loc
    """
    df.iloc[rows, get_loc(df.columns, cols)] = value
    return value


def group_intervals(
    df: pd.DataFrame,
    group_by=None,
    point_col=None,
    start_col=None,
    stop_col=None,
    distance=None,
    inclusive: bool = True,
) -> pd.Series:
    """Group intervals wrt. their distance to each other. Intervals can also be isolated points, i.e.,
    single-point intervals of the form `[x, x]`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with intervals.
    group_by : optional
        Additional column(s) to group `df` by, optional. If given, the computed grouping refines the given one, in the
        sense that any two intervals belonging to the same computed group are guaranteed to belong to the same given
        group, too. Can be the name of a single column or a list of column names and/or row index levels. Strings are
        interpreted as column names or row index names, integers are interpreted as row index levels.
    point_col : str, optional
        Name of the column in `df` containing both start- and end points of single-point intervals. If given, both
        `start_col` and `stop_col` must be None.
    start_col : str, optional
        Name of the column in `df` containing start (left) endpoints of intervals. If given, `point_col` must be None.
    stop_col : str, optional
        Name of the column in `df` containing end (right) endpoints of intervals. If given, `point_col` must be None.
        Note that the function tacitly assumes that no interval ends before it starts, although this is not checked.
        If this assumption is violated, the returned results may not be correct.
    distance : optional
        Maximum allowed distance between two intervals for being put into the same group. Should be non-negative.
        The distance between two intervals is the single-linkage distance, i.e., the minimum distance between any two
        points in the respective intervals. This means, for example, that the distance between overlapping intervals is
        always 0.
    inclusive : bool, default=False
        Whether `distance` is inclusive.

    Notes
    -----
    The returned grouping is the reflexive-transitive closure of the proximity relation induced by `distance`.
    Formally: Let :math:`R` be the binary relation on the set of intervals in `df` such that :math:`R(I_1, I_2)` holds
    iff the distance between :math:`I_1` and :math:`I_2` is less than (or equal to) `distance` (and additionally
    :math:`I_1` and :math:`I_2` belong to the same groups specified by `group_by`). :math:`R` is obviously symmetric,
    so its reflexive-transitive closure :math:`R^*` is an equivalence relation on the set of intervals in `df`. The
    returned grouping corresponds precisely to this equivalence relation, in the sense that there is one group per
    equivalence class and vice versa.
    Note that if two intervals belong to the same group, their distance may still be larger than `distance`.

    Returns
    -------
    pd.Series
        Series with the same row index as `df`, in the same order, whose values are group indices.
    """

    assert distance is not None

    if point_col is None:
        assert start_col in df.columns
        if stop_col is None:
            stop_col = start_col
        else:
            assert stop_col in df.columns
    else:
        assert point_col in df.columns
        assert start_col is None
        assert stop_col is None
        start_col = point_col
        stop_col = point_col

    if group_by is None:
        group_by = []
    else:
        group_by = _parse_column_specs(df, group_by)

        if not all(isinstance(g, (str, tuple)) for g in group_by):
            # construct new DataFrame
            df_new = pd.DataFrame(
                data={
                    f"g{i}": df[g]
                    if isinstance(g, (str, tuple))
                    else (df.index.get_level_values(g) if isinstance(g, int) else g)
                    for i, g in enumerate(group_by)
                }
            )
            group_by = [f"g{i}" for i in range(len(group_by))]
            df_new["start"] = df[start_col].values
            if stop_col == start_col:
                stop_col = "start"
            else:
                df_new["stop"] = df[stop_col].values
                stop_col = "stop"
            start_col = "start"
            df_new.index = df.index
            df = df_new

    sorting_col = "__sorting__"
    if df.columns.nlevels > 1:
        sorting_col = tuple([sorting_col] + [""] * (df.columns.nlevels - 1))
    assert sorting_col not in df.columns

    df[sorting_col] = np.arange(len(df))
    df_sorted = df.sort_values(group_by + [start_col])
    df.drop([sorting_col], axis=1, inplace=True, errors="ignore")

    start = df_sorted[start_col]
    if start_col == stop_col:
        stop = start
    elif group_by:
        stop = df_sorted.groupby(group_by)[stop_col].cummax()
        assert (stop.index == df_sorted.index).all()
    else:
        stop = df_sorted[stop_col].cummax()
    # `stop` is the per-group cumulative maximum of all previous interval ends.
    # This ensures that some kind of triangle inequality holds: if all interval endpoints are modified according to
    # this procedure and `I_1` and `I_2` are consecutive intervals, then there does not exist another interval `I_3`
    # with `dist(I_1, I_3) < dist(I_1, I_2)` and `dist(I_2, I_3) < dist(I_1, I_2)`. This property is crucial for
    # ensuring that the grouping indeed corresponds to the reflexive-transitive closure of the proximity relation.
    # If the complete linkage distance were used, no ordering of `df` could establish the above property.
    # Counterexample: I_1 = [0, 10], I_2 = [2, 9], I_3 = [3, 8]. No matter how `df` is sorted, `I_1` and `I_2` would
    # always end up next to each other, but `dist(I_1, I_3) = 7 < 9 = dist(I_1, I_2)` and
    # `dist(I_2, I_3) = 6 < 9 = dist(I_1, I_2)`.

    same_group_as_prev = shift_compare(
        start, -1, np.less_equal if inclusive else np.less, other=stop + distance, fill_value=True
    )
    if len(same_group_as_prev):
        for g in group_by:
            same_group_as_prev &= shift_equal(df_sorted[g], -1, fill_value=True)
    out = pd.Series(index=df_sorted[sorting_col].values, data=(~same_group_as_prev).cumsum(), dtype=np.int64)
    out.sort_index(inplace=True)
    out.index = df.index
    return out


def combine_intervals(
    df: pd.DataFrame,
    start_col: str = "start",
    stop_col: str = "stop",
    length_col: str = None,
    attr_cols=(),
    group_by=None,
    n_min: int = 1,
    n_max: int = None,
) -> pd.DataFrame:
    """Combine intervals (e.g., of measurements of different signals) to obtain new intervals where at least `n_min`
    original intervals overlap. Suitable values can be provided for `n_min` to cover intersection and union.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the original intervals. Must have columns `start_col`, `stop_col` and all columns in `attr_cols`
        and `group_by`.
    start_col : str, default="start"
        Name of the column containing the start (left) endpoints of the intervals.
    stop_col : str, optional, default="stop"
        Name of the column containing the stop (right) endpoints of the intervals. Precisely one of `stop_col` and
        `length_col` must be given.
    length_col : str, optional
        Name of the column containing interval lengths. Precisely one of `stop_col` and `length_col` must be given.
    attr_cols : list of str, default=[]
        List of columns whose values describe attributes associated with individual intervals, like measured
        parameters. Intervals with the same attributes are considered "equivalent" in the sense that overlaps of such
        intervals still only count as _one_ interval. Pass an empty list if any overlaps should count as multiple
        intervals.
    group_by : list of str, optional
        List of columns by which to group `df`. Intervals are not combined across groups.
    n_min : int, default=1
        Minimum number of overlapping intervals to include in the result.
    n_max : int, optional
        Maximum number of overlapping intervals to include in the result, or None to impose no maximum.

    Returns
    -------
    DataFrame with columns `start_col`, `stop_col` (or `length_col`) and the columns in `group_by`.

    Examples
    --------
    Assume `df` is a DataFrame containing measurement intervals, with columns "subject_id", "start", "stop" and "param".

    Find all intervals where at least `n` _distinct_ parameters were measured simultaneously, grouping by subject:
    >>> combine_intervals(
    ...     df,
    ...     attr_cols=["param"],
    ...     start_col="start",
    ...     stop_col="stop",
    ...     group_by="subject_id",
    ...     n_min=n
    ... )   # doctest: +SKIP

    Same as above, but count simultaneous measurements of the same parameter as distinct measurements:
    >>> combine_intervals(
    ...     df,
    ...     start_col="start",
    ...     stop_col="stop",
    ...     group_by="subject_id",
    ...     n_min=n
    ... )   # doctest: +SKIP

    Find all intervals where the same parameter was measured multiple times:
    >>> combine_intervals(
    ...     df,
    ...     start_col="start",
    ...     stop_col="stop",
    ...     group_by=["subject_id", "param"],
    ...     n_min=2
    ... )   # doctest: +SKIP

    Find all measurement gaps of a parameter. Note that only gaps _between_ two intervals are returned; the infinite
    non-measurement periods before the first and after the last interval must be added manually:
    >>> combine_intervals(
    ...     df,
    ...     start_col="start",
    ...     stop_col="stop",
    ...     group_by=["subject_id", "param"],
    ...     n_min=0,
    ...     n_max=0
    ... )   # doctest: +SKIP

    Find the start of the first and end of the last measurement interval per parameter:
    >>> combine_intervals(
    ...     df,
    ...     start_col="start",
    ...     stop_col="stop",
    ...     group_by=["subject_id", "param"],
    ...     n_min=0,
    ...     n_max=None
    ... )   # doctest: +SKIP
    """

    assert n_max is None or n_min <= n_max
    assert (stop_col is None) != (length_col is None)

    if isinstance(attr_cols, str):
        attr_cols = [attr_cols]
    if group_by is None:
        group_by = []
    elif isinstance(group_by, str):
        group_by = [group_by]

    assert not any(c in group_by for c in attr_cols)

    if attr_cols:
        all_combos = df.groupby(attr_cols).size().to_frame()[[]].reset_index()
    else:
        all_combos = pd.DataFrame(index=[0])

    aux = []
    for n in all_combos.index:
        mask = pd.Series(True, index=df.index)
        for c in all_combos.columns:
            mask &= df[c] == all_combos[c].iloc[n]
        start = df.loc[mask, group_by + [start_col]]
        start.rename({start_col: "__t"}, axis=1, inplace=True)
        if stop_col is None:
            stop = start["__t"] + df.loc[mask, length_col]
        else:
            stop = df.loc[mask, stop_col]
        stop = stop.to_frame("__t")
        if group_by:
            stop[group_by] = start[group_by]
        start["__" + str(n)] = 1
        stop["__" + str(n)] = -1
        aux.extend([start, stop])

    if aux:
        aux = pd.concat(aux, axis=0, ignore_index=True, sort=True)
        aux.fillna(0, inplace=True)
        aux = aux.groupby(group_by + ["__t"]).sum()
        aux.sort_index(inplace=True)
        aux.reset_index(drop=False, inplace=True)
        if all_combos.shape[1] == 0:
            aux["__a"] = aux[["__" + str(n) for n in all_combos.index]].sum(axis=1).cumsum()
        else:
            aux["__a"] = 0
            for n in all_combos.index:
                aux["__a"] += aux["__" + str(n)].cumsum() > 0
        mask = np.ones((len(aux),), dtype=bool)
        for g in group_by:
            mask &= shift_equal(aux[g], 1)
        mask[-1] = False
        # `mask[i]` is True iff `i`-th element belongs to same group as next
        assert (mask | (aux["__a"] == 0)).all()

        if n_max is None:
            aux["__a"] = n_min <= aux["__a"]
        else:
            aux["__a"] = (n_min <= aux["__a"]) & (aux["__a"] <= n_max)
        # ensure that the last element of each group is False (could be violated if `n_min` is 0)
        aux["__a"] &= mask

        mask = shift_unequal(aux["__a"], -1)
        mask[0] = aux["__a"].iloc[0]
        aux = aux[mask]
        if not aux.empty:
            assert aux["__a"].iloc[0]  # first row must be True
            assert not aux["__a"].iloc[-1]  # last row must be False
            assert len(aux) % 2 == 0  # even number of rows
            start = aux[group_by + ["__t"]].iloc[::2].reset_index(drop=True)
            stop = aux[group_by + ["__t"]].iloc[1::2].reset_index(drop=True)
            assert (start[group_by] == stop[group_by]).all().all()
            start.rename({"__t": start_col}, axis=1, inplace=True)
            stop = stop["__t"]
            mask = start[start_col] < stop
            if mask.any():
                if stop_col is None:
                    stop = (stop - start[start_col]).to_frame(length_col)
                else:
                    stop = stop.to_frame(stop_col)
                return start.join(stop)[mask].reset_index(drop=True)

    return pd.DataFrame(columns=group_by + [start_col, length_col if stop_col is None else stop_col])


def prev_next_values(
    df: pd.DataFrame,
    sort_by=None,
    group_by=None,
    columns=None,
    first_indicator_name=None,
    last_indicator_name=None,
    keep_sorted: bool = False,
    inplace: bool = False,
) -> pd.DataFrame:
    """Find the previous/next values of some columns in DataFrame `df`, for every entry. Additionally, entries can be
    grouped and previous/next values only searched within each group.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame.
    sort_by : list | str, optional
        The column(s) to sort by. Can be the name of a single column or a list of column names and/or row index levels.
        Strings are interpreted as column names or row index names, integers are interpreted as row index levels.
        ATTENTION! N/A values in columns to sort by are not ignored; rather, they are treated in the same way as Pandas
        treats such values in `DataFrame.sort_values()`, i.e., they are put at the end.
    group_by : list | str, optional
        Column(s) to group `df` by, optional. Same values as `sort_by`.
    columns : dict
        A dict mapping column names to dicts of the form

        ::

            {
                "prev_name": <prev_name>,
                "prev_fill": <prev_fill>,
                "next_name": <next_name>,
                "next_fill": <next_fill>
            }

        `prev_name` and `next_name` are the names of the columns in the result, containing the previous/next values.
        If any of them is None, the corresponding previous/next values are not computed for that column.
        `prev_fill` and `next_fill` specify which values to assign to the first/last entry in every group, which does
        not have any previous/next values.
        Note that column names not present in `df` are tacitly skipped.
    first_indicator_name : str, optional
        Name of the column in the result containing boolean indicators whether the corresponding entries come first in
        their respective groups. If None, no such column is added.
    last_indicator_name : str, optional
        Name of the column in the result containing boolean indicators whether the corresponding entries come last in
        their respective groups. If None, no such column is added.
    keep_sorted : bool, default=False
        Keep the result sorted wrt. `group_by` and `sort_by`. If False, the order of rows of the result is identical
        to that of `df`.
    inplace : bool, default=False
        If `True`, the new columns are added to `df`.

    Returns
    -------
    pd.DataFrame
        The modified DataFrame if `inplace` is True, a DataFrame with the requested previous/next values otherwise.
    """

    if columns is None:
        columns = {}
    elif isinstance(columns, (list, np.ndarray)):
        columns = {k: dict(prev_name=f"{k}_prev", next_name=f"{k}_next") for k in columns if k in df.columns}
    elif isinstance(columns, dict):
        columns = {k: v for k, v in columns.items() if k in df.columns and ("prev_name" in v or "next_name" in v)}
    elif columns in df.columns:
        columns = {columns: dict(prev_name=f"{columns}_prev", next_name=f"{columns}_next")}
    else:
        columns = {}

    if not (columns or first_indicator_name or last_indicator_name):
        return df if inplace else pd.DataFrame(index=df.index)

    assert sort_by is not None
    sort_by = _parse_column_specs(df, sort_by)
    assert len(sort_by) > 0

    if group_by is None:
        group_by = []
    else:
        group_by = _parse_column_specs(df, group_by)
        if any(s in group_by for s in sort_by):
            raise ValueError("sort_by and group_by must be disjoint.")

    sorting_col = "__sorting__"

    prev_mask = np.zeros(len(df), dtype=bool)  # True iff previous element belongs to different group
    if len(df) == 0:
        sorting = np.zeros(0, dtype=np.int32)
        df_sorted = df
    else:
        if all(isinstance(g, (str, tuple)) for g in group_by) and all(isinstance(s, (str, tuple)) for s in sort_by):
            if len(df) > 0:  # otherwise row index would be renamed to None
                if df.columns.nlevels > 1:
                    sorting_col_df = tuple([sorting_col] + [""] * (df.columns.nlevels - 1))
                else:
                    sorting_col_df = sorting_col
                assert sorting_col_df not in df.columns
                df[sorting_col_df] = np.arange(len(df))
            else:
                sorting_col_df = None
            if inplace and keep_sorted:
                df.sort_values(group_by + sort_by, inplace=True)
                df_sorted = df
            else:
                df_sorted = df.sort_values(group_by + sort_by)
            if len(df) > 0:
                sorting = df_sorted[sorting_col_df].values
                df.drop([sorting_col_df], axis=1, inplace=True)
            else:
                sorting = np.zeros(0, dtype=np.int32)
            df_aux = df_sorted
        else:
            # construct new DataFrame
            df_aux = pd.DataFrame(
                data={
                    k: df[c]
                    if isinstance(c, (str, tuple))
                    else (df.index.get_level_values(c) if isinstance(c, int) else c)
                    for k, c in [("g" + str(i), g) for i, g in enumerate(group_by)]
                    + [("s" + str(i), s) for i, s in enumerate(sort_by)]
                }
            )
            df_aux[sorting_col] = np.arange(len(df))
            group_by = [f"g{i}" for i in range(len(group_by))]
            sort_by = [f"s{i}" for i in range(len(sort_by))]
            df_aux.sort_values(group_by + sort_by, inplace=True)
            sorting = df_aux[sorting_col].values
            if inplace and keep_sorted:
                if df.columns.nlevels > 1:
                    sorting_col_df = tuple([sorting_col] + [""] * (df.columns.nlevels - 1))
                else:
                    sorting_col_df = sorting_col
                assert sorting_col_df not in df.columns
                df[sorting_col_df] = np.argsort(sorting)
                df.sort_values([sorting_col_df], inplace=True)
                df.drop([sorting_col_df], axis=1, inplace=True)
                # `df` is now sorted as `df_aux`
                df_sorted = df
            else:
                df_sorted = df.iloc[sorting]

        for g in group_by:
            prev_mask |= shift_unequal(df_aux[g], -1)
        prev_mask[0] = True

        if df_aux is not df_sorted:
            del df_aux

    next_mask = roll1d(prev_mask, -1)  # True iff next element belongs to different group
    new_columns = []
    for k, v in columns.items():
        col = v.get("prev_name")
        if col is not None:
            if len(df_sorted) == 0:
                s = pd.Series(index=df_sorted.index, data=df_sorted[k], name=col)
            else:
                s = pd.Series(index=df_sorted.index, data=roll1d(df_sorted[k].values, 1), name=col)
                s[prev_mask] = v.get("prev_fill", pd.Timedelta(None) if s.dtype.kind == "m" else None)
            new_columns.append(s)

        col = v.get("next_name")
        if col is not None:
            if len(df_sorted) == 0:
                s = pd.Series(index=df_sorted.index, data=df_sorted[k], name=col)
            else:
                s = pd.Series(index=df_sorted.index, data=roll1d(df_sorted[k].values, -1), name=col)
                s[next_mask] = v.get("next_fill", pd.Timedelta(None) if s.dtype.kind == "m" else None)
            new_columns.append(s)
    if first_indicator_name is not None:
        new_columns.append(pd.Series(index=df_sorted.index, data=prev_mask, name=first_indicator_name))
    if last_indicator_name is not None:
        new_columns.append(pd.Series(index=df_sorted.index, data=next_mask, name=last_indicator_name))

    if new_columns:
        if inplace:
            if keep_sorted:
                for s in new_columns:
                    df[s.name] = s
            else:
                sorting_inv = np.argsort(sorting)
                for s in new_columns:
                    df[s.name] = s.iloc[sorting_inv]
            out = df
        else:
            out = pd.concat(new_columns, axis=1, sort=False)
            if len(out) > 0 and not keep_sorted:
                if out.columns.nlevels > 1:
                    sorting_col_out = tuple([sorting_col] + [""] * (out.columns.nlevels - 1))
                else:
                    sorting_col_out = sorting_col
                assert sorting_col_out not in out.columns
                out[sorting_col_out] = sorting
                out.sort_values([sorting_col_out], inplace=True)
                out.drop([sorting_col_out], axis=1, inplace=True)
    elif inplace:
        out = df
    elif keep_sorted:
        out = pd.DataFrame(index=df_sorted.index)
    else:
        out = pd.DataFrame(index=df.index)

    return out


def partition_series(s: pd.Series, n, shuffle: bool = True) -> pd.Series:
    """Partition a given Series into as few groups as possible, such that the sum of the series' values in each group
    does not exceed a given threshold.

    Parameters
    ----------
    s : pd.Series
        The series to partition. The data type should allow for taking sums and comparing elements, and all values are
        assumed to be non-negative.
    n
        The threshold, of the same data type as `s`.
    shuffle : bool, default=True
        Randomly shuffle `s` before generating the partition. If False, this function is deterministic.

    Returns
    -------
    pd.Series
        A new series with the same index as `s`, with values from 0 to `g - 1` specifying the group ID of each entry
        (`g` is the total number of groups).

    Notes
    -----
    The partitions returned by this function may not be optimal, since finding optimal partitions is computationally
    difficult. Also note that `s` may contain entries whose value exceeds `n`; such entries are put into singleton
    groups.
    """

    out = pd.Series(index=s.index, data=0, dtype=np.int64)
    if s.sum() <= n:
        return out

    groups = {}
    if shuffle:
        rng = np.random.permutation(len(s))
    else:
        rng = range(len(s))
    m = 0
    for i in rng:
        x = s.iloc[i]
        j = -1
        if x < n:
            for k, v in groups.items():
                if v + x <= n:
                    groups[k] += x
                    j = k
                    break
        if j < 0:
            j = m
            m += 1
            groups[j] = x
        out.iloc[i] = j

    return out


def impute(
    df: pd.DataFrame,
    method: str = "ffill",
    group_by: Union[int, str, pd.Series, pd.Index, np.ndarray, None] = None,
    limit: Optional[int] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Impute a DataFrame by forward/backward filling and/or linear interpolation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to impute. Must be correctly ordered already.
    method : str
        Imputation method to use, one of:
        * "ffill": forward filling,
        * "bfill": backward filling,
        * "afill": average of forward- and backward filling, i.e., entries that would be filled by both methods are set
            to the average of the two values,
        * "lfill": linear interpolation followed by forward- and backward filling; note that if all groups have 3 or
            fewer rows, "lfill" is equivalent to "afill",
        * "linear": linear interpolation; note that no extrapolation happens (equivalent to `area="inside"` in
            `pd.DataFrame.interpolate()`), and also note that row index values are completely ignored.
    group_by : str | int | pd.Series | pd.Index | array, optional
        Column or index to group by. Integers are interpreted as row index levels, strings as column names.
        Series and arrays may be provided as well; this could be the result of function `factorize()`, for instance.
    limit : int, optional
        Imputation limit, applies to both forward/backward filling and linear interpolation.
    inplace : bool, default=False
        Modify `df` in place.

    Returns
    -------
    pd.DataFrame
        Imputed DataFrame. Due to the nature of the provided imputation methods, the result may still contain NaN.

    Notes
    -----
    This function mainly boosts the performance of grouped linear interpolations, i.e., when `group_by` is not None and
    `method` is "linear" or "lfill": depending on the input, a speed-up factor of up to 10 is possible (especially if
    `on` is not mixed up).
    Other imputation strategies, like "ffill", are not faster than plain `df.groupby(on).ffill()`; they are merely
    covered for the sake of completeness.
    """

    if method == "afill":
        df0 = impute(df, method="ffill", group_by=group_by, limit=limit, inplace=False)
        df = impute(df, method="bfill", group_by=group_by, limit=limit, inplace=inplace)
        df.fillna(df0, inplace=True)
        df0.fillna(df, inplace=True)

        if isinstance(group_by, (int, np.integer, pd.Series, pd.Index, np.ndarray)):
            columns = list(df.columns)
        else:
            columns = [c for c in df.columns if c != group_by]

        # this also works for Timestamps
        df[columns] += (df0[columns] - df[columns]) / 2

        return df
    elif method == "lfill":
        df = impute(df, method="linear", group_by=group_by, limit=limit, inplace=inplace)
        impute(df, method="ffill", group_by=group_by, limit=limit, inplace=True)
        impute(df, method="bfill", group_by=group_by, limit=limit, inplace=True)
        return df

    if group_by is None:
        if method == "ffill":
            out = df.ffill(axis=0, limit=limit, inplace=inplace)
        elif method == "bfill":
            out = df.bfill(axis=0, limit=limit, inplace=inplace)
        elif method == "linear":
            out = df.interpolate(method="linear", axis=0, limit=limit, inplace=inplace, limit_area="inside")
        else:
            raise ValueError('`method` must be "ffill", "bfill", "afill", "lfill" or "linear".')
        return df if out is None else out

    if method not in ("ffill", "bfill", "linear"):
        raise ValueError('`method` must be "ffill", "bfill", "afill", "lfill" or "linear".')

    if isinstance(group_by, (int, np.integer)):
        columns = list(df.columns)
        group_values = df.index.get_level_values(group_by)
    elif isinstance(group_by, pd.Series):
        columns = list(df.columns)
        if len(group_by) != len(df) or not (group_by.index == df.index).all():
            group_by = group_by.reindex(df.index)
        group_values = group_by.values
    elif isinstance(group_by, pd.Index):
        if isinstance(group_by, pd.MultiIndex):
            raise ValueError("Multiindex is not supported")
        elif len(group_by) != len(df):
            raise ValueError("Grouper and axis must be same length")
        columns = list(df.columns)
        group_values = group_by.get_level_values(0)
    elif isinstance(group_by, np.ndarray):
        if len(group_by) != len(df):
            raise ValueError("Grouper and axis must be same length")
        columns = list(df.columns)
    else:
        columns = [c for c in df.columns if c != group_by]
        group_values = df[group_by].values

    if not inplace:
        df = df.copy()

    # the following works for "ffill" and "bfill", too, but empirical results suggest that in these cases
    # `df.groupby(on).ffill()` is even faster, so we leave it
    if method == "linear":
        # check whether groups are mixed up
        mask0 = shift_unequal(group_values, 1)

        if len(np.unique(group_values[mask0])) == mask0.sum():
            indexer = None
        else:
            indexer = np.argsort(group_values, kind="stable")
            df = df.iloc[indexer]
            group_values = group_values[indexer]

        # `mask` specifies those entries that cannot be imputed, and therefore must be set to NaN after imputation
        mask = _can_impute(
            df[columns],
            group_values,
            forward=method in ("ffill", "linear"),
            backward=method in ("bfill", "linear"),
            union=True,
        )

        if method == "linear":
            df[columns] = df[columns].interpolate(method=method, axis=0, limit=limit, limit_area="inside")
        elif method == "ffill":
            df[columns] = df[columns].ffill(axis=0, limit=limit)
        else:
            df[columns] = df[columns].bfill(axis=0, limit=limit)

        for c in columns:
            df.loc[mask[c], c] = None

        if indexer is not None:
            df = df.iloc[np.argsort(indexer)]

    elif method == "ffill":
        df[columns] = df[columns].groupby(group_values).ffill(limit=limit)
    else:
        df[columns] = df[columns].groupby(group_values).bfill(limit=limit)

    return df


def grouped_mode(series: pd.Series, dropna: bool = True) -> pd.DataFrame:
    """Group the given Series `series` by its row index and compute mode aggregations. If there are more than one most
    frequent values in a group, the "first" is chosen, i.e., the result is always one single value.

    Parameters
    ----------
    series : pd.Series
        The Series to aggregate. The number of row index levels is arbitrary.
    dropna : bool, default=True
        Drop NaN values before computing the mode. If True, the most frequent value of a group is NaN iff all values of
        the group are NaN.

    Returns
    -------
    pd.DataFrame
        The mode-aggregated DataFrame with columns "mode" and "count", and one row for each group.
        "count" is 0 iff `dropna` is True and all values of the respective group are NaN.

    Notes
    -----
    Very fast method to compute grouped mode, based on [1]. Using the built-in `mode()` function is not possible,
    because it returns the list of most frequent values. Work-arounds a la `lambda x: x.mode()[0]` are **terribly**
    slow.

    References
    ----------
    .. [1] https://stackoverflow.com/a/38216118
    """

    group_levels = list(range(series.index.nlevels + 1))
    idx = series.groupby(level=group_levels[:-1]).size().index
    mask = series.notna().values if dropna else np.ones(len(series), dtype=bool)
    if mask.any():
        series = series.to_frame("mode")
        series.set_index("mode", append=True, inplace=True)
        df = series[mask].groupby(level=group_levels, dropna=False, observed=True).size().to_frame("count")
        df.reset_index(level=-1, inplace=True)
        df.sort_values("count", ascending=False, inplace=True, kind="stable")
        df = df.loc[~df.index.duplicated()].reindex(idx)
        df["count"] = df["count"].fillna(0).astype(np.int64)
    elif len(series) > 0:
        df = pd.Series(index=idx, data=None, dtype=series.dtype).to_frame("mode")
        df["count"] = 0
    else:
        df = series.to_frame("mode")
        df["count"] = 0

    return df


def inner_or_cross_join(left: pd.DataFrame, right: pd.DataFrame, on=None) -> pd.DataFrame:
    """Return the inner join or cross join of two DataFrames, depending on whether the column to join on actually
    occurs in the DataFrames.

    Parameters
    ----------
    left : pd.DataFrame
        The first DataFrame.
    right : pd.DataFrame
        The second DataFrame.
    on : str, optional
        The column to join on, or None.

    Returns
    -------
    pd.DataFrame
        If `on` is not None and occurs in `left`, return the inner join of `left` (column `on`) and `right` (row index).
        Otherwise, return the cross join of `left` and `right`; in that case , the index of the result corresponds to
        the (replicated) index of `left` and the index of `right` is completely ignored.

    Notes
    -----
    Functionally, the cross join operation is equivalent to joining `left` and `right` on a constant-valued column:

    ::

        left["join_column"] = 0
        right.index = 0
        left.join(right, on="join_column").drop("join_column", axis=1)
    """
    assert not any(c in right.columns for c in left.columns), "Columns of `left` and `right` are not disjoint."
    if on is None or on not in left.columns:
        out = pd.DataFrame(
            index=np.tile(left.index, len(right)), data={c: np.tile(left[c].values, len(right)) for c in left.columns}
        )
        for c in right.columns:
            out[c] = np.repeat(right[c].values, len(left))
        return out
    else:
        return left.join(right, on=on, how="inner")


def roll1d(arr, shift: int):
    """Roll elements of a 1-D array-like.

    Elements that roll beyond the last position are re-introduced at the first.

    Parameters
    ----------
    arr : pd.DataFrame | pd.Series | pd.Index | pd.Categorical | pd.ExtensionArray | array-like
        Input array-like. Must support indexing with int-arrays, either as `arr[idx]` or `arr.iloc[idx]`.
    shift : int
        The number of places by which elements are shifted.

    Returns
    -------
    Output array-like, with the same shape and type as `a`.

    See Also
    --------
    np.roll

    Notes
    -----
    In contrast to `np.roll`, which converts everything into a plain array before rolling, this function correctly
    handles all sorts of pandas objects (including ExtensionArrays) and returns output with the same type.
    If `shift` is in {-1, 0, 1}, the function is at least as efficient as `np.roll`, depending on the type of `arr`.

    Examples
    --------
    >>> x = np.arange(10)
    >>> roll1d(x, 2)
    array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> roll1d(x, -2)
    array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1])
    """

    if len(arr) <= 1:
        return arr
    else:
        shift %= len(arr)
        if shift == 0:
            return arr
        elif isinstance(arr, np.ndarray):
            return np.roll(arr, shift, axis=0)
        elif isinstance(arr, pd.CategoricalIndex):
            pd.Index(
                pd.Categorical.from_codes(
                    np.roll(arr.codes, shift, axis=0),
                    categories=arr.categories,
                    ordered=arr.ordered,
                ),
                name=arr.name,
            )
        elif isinstance(arr, pd.Index):
            if arr.nlevels == 1 and arr.dtype.kind in "uifbMmO":
                return arr.__class__(np.roll(arr, shift, axis=0))
        elif shift in (1, -1) and hasattr(arr, "shift"):
            idx = -1 if shift > 0 else 0
            return arr.shift(shift, fill_value=(arr.iloc[idx] if hasattr(arr, "iloc") else arr[idx]))

        idx = np.arange(-shift, len(arr) - shift)
        return arr.iloc[idx] if hasattr(arr, "iloc") else arr[idx]


def shift_compare(arr, shift: int, cmp, other=None, fill_value: bool = False) -> np.ndarray:
    """Compare a given 1-D array-like element-wise to a shifted version of itself or another array-like.

    Parameters
    ----------
    arr : pd.Series | pd.Index | pd.Categorical | pd.ExtensionArray | array-like
        The array-like to compare.
    shift : int
        The number of places by which elements are shifted.
    cmp : callable
        The comparison relation, typically one of `np.equal`, `np.not_equal`, `np.less`, etc.
        Must accept two equal-length array-likes as input, plus an optional third array-like for storing the output.
    other : pd.Series | pd.Index | pd.Categorical | pd.ExtensionArray | array-like, optional
        Other array-like to which `arr` is compared, with the same length and suitable data type.
        None defaults to `arr`.
    fill_value : bool, default=False
        The value used for filling output elements that cannot be determined by comparison, namely the first `-shift`
        elements if `shift` is negative and the last `shift` elements if it is positive.

    Returns
    -------
    mask : np.ndarray
        Boolean mask, array with the same length as `arr`.
        `mask[i]` is True iff `cmp(arr[i], other[i + shift])`, or `fill_value` is True and `i + shift` is no valid
        index.
        So, if `shift` is -1, `mask` selects all elements that compare to the previous element in `other`, and if
        `shift` is 1, it selects all elements that compare to the subsequent element.

    Notes
    -----
    If `shift` in `(-len(arr), len(arr))`, this function is functionally equivalent to
    `cmp(arr, roll1d(other, -shift))`, modulo setting some output elements to `fill_value`.
    Note the `-shift` in `roll1d`!

    In constrast to function `roll1d`, `shift_compare(arr, shift, cmp)` is *not* the same as
    `shift_compare(arr, shift + len(arr), cmp)`!
    """
    if len(arr) == 0:
        return np.empty(arr.shape, dtype=bool)
    elif shift == 0:
        if other is None:
            return np.ones(len(arr), dtype=bool)
        else:
            if isinstance(arr, pd.Series):
                arr = arr.values
            if isinstance(other, pd.Series):
                other = other.values
            return cmp(arr, other)
    else:
        out = np.ones(len(arr), dtype=bool) if fill_value else np.zeros(len(arr), dtype=bool)
        if shift <= -len(arr) or shift >= len(arr):
            pass
        else:
            if isinstance(arr, pd.Series):
                arr = arr.values
            if other is None:
                other = arr
            elif isinstance(other, pd.Series):
                other = other.values
            shift = -shift
            if shift < 0:
                cmp(arr[:shift], other[-shift:], out=out[:shift])
            else:
                cmp(arr[shift:], other[:-shift], out=out[shift:])
        return out


def shift_equal(arr, shift: int, other=None, fill_value: bool = False) -> np.ndarray:
    return shift_compare(arr, shift, np.equal, other=other, fill_value=fill_value)


def shift_unequal(arr, shift: int, other=None, fill_value: bool = True) -> np.ndarray:
    return shift_compare(arr, shift, np.not_equal, other=other, fill_value=fill_value)


def factorize(
    left: Union[pd.DataFrame, pd.Series, pd.Categorical, pd.Index, np.ndarray],
    right: Union[pd.DataFrame, pd.Series, pd.Categorical, pd.Index, np.ndarray, None] = None,
    sort: bool = False,
    return_count: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, int], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, int]]:
    """Factorize DataFrames, Series, Indexes or arrays. That means, rows are mapped to integer keys such that rows with
    identical values are mapped to the same key, and rows with distinct values to distinct keys. This is often a useful
    preprocessing step in group-by- and join-related functions.

    Parameters
    ----------
    left : pd.DataFrame | pd.Series | pd.Index | array
        First object to factorize.
    right : pd.DataFrame | pd.Series | pd.Index | array, optional
        Second object to factorize alongside `left`, optional. If given, matching rows in `left` and `right` are mapped
        to the same key, and distinct rows to distinct keys.
        The type of `right` does not need to be the same as the type of `left`, but if `left` has multiple
        columns/levels, `right` needs to have the same number of columns/levels.
    sort : bool, default=False
        If False, the first row in `left` is mapped to the smallest key, the second unique row to the second-smallest
        key, and so on. This does *not* necessarily imply that the returned arrays are sorted, e.g., if the first row
        appears multiple times.
        If True, the smallest row in `left` and `right` (wrt. lexicographic ordering of all columns/levels) is mapped
        to the smallest key, the second-smallest row to the second-smallest key, and so on.
        See Notes for more details.
    return_count : bool, default=False
        Return the number of unique keys.

    Returns
    -------
    left_keys : array
        Keys of `left`, 1-D array with the same length as `left`.
    right_keys : array
        Keys of `right`, 1-D array with the same length as `right`. Only returned if `right` is not None.
    count : int
        Number of unique keys appearing in `left_keys` and `right_keys`. Only returned if `return_count` is True.
        Note that the keys are not necessarily in the interval `[0, count)`.

    See Also
    --------
    pandas.core.reshape.merge.get_join_indexers
    pandas.core.reshape.merge._factorize_keys
    pandas.core.reshape.merge._get_join_keys

    Notes
    -----
    Column/index names in `left` and `right` are completely ignored. NaN and +/-Inf are treated as separate factors.

    With `sort=False`, `left.iloc[_map(left_keys)]` preserves the order of rows in `left`;
    `right.iloc[_map(right_keys)]` does not necessarily preserve the order of rows in `right`, though.
    (NB: `_map` is meant to map the keys to the range `[0, len(left))` and `[0, len(right))`, respectively.
    That's *not* the same as computing the ranks of the keys!)

    With `sort=True`, `left.iloc[np.argsort(left_keys)]` sorts `left` and `right.iloc[np.argsort(right_keys)]` sorts
    `right`.
    """
    # largely inspired by functions in pandas.core.reshape.merge, e.g., `get_join_indexers()`

    if isinstance(left, pd.Categorical):
        left = pd.Series(left)
    elif isinstance(left, pd.DataFrame) and left.shape[1] == 1:
        left = left.iloc[:, 0]
    if isinstance(right, pd.Categorical):
        right = pd.Series(right)
    elif isinstance(right, pd.DataFrame) and right.shape[1] == 1:
        right = right.iloc[:, 0]

    if isinstance(left, pd.Series):
        if left.dtype.name == "category":
            if right is None:
                left = left.cat.codes.values
            elif isinstance(right, pd.Series) and left.dtype == right.dtype:
                left = left.cat.codes.values
                right = right.cat.codes.values
            else:
                left = np.asarray(left.values)
        else:
            left = np.asarray(left.values)
    elif isinstance(left, pd.Index):
        if not isinstance(left, pd.MultiIndex):
            left = np.asarray(left._values)
    elif isinstance(left, np.ndarray):
        if left.ndim != 1:
            raise ValueError(f"left must be an array with rank 1, but got rank {left.ndim}")

    if isinstance(right, pd.Series):
        right = np.asarray(right.values)
    elif isinstance(right, pd.Index):
        if not isinstance(right, pd.MultiIndex):
            right = np.asarray(right._values)
    elif isinstance(right, np.ndarray):
        if right.ndim != 1:
            raise ValueError(f"right must be an array with rank 1, but got rank {right.ndim}")

    # convert datetime arrays into DatetimeArray
    if isinstance(left, np.ndarray) and left.dtype.kind == "M":
        left = pd.Index(left)._values
    if isinstance(right, np.ndarray) and right.dtype.kind == "M":
        right = pd.Index(right)._values

    # left and right are now either DataFrame, MultiIndex or 1-D array-like; right may also be None

    if right is None:
        if len(left) == 0:
            res = (np.empty(0, dtype=np.int64), 0)
        elif isinstance(left, pd.DataFrame):
            res = factorize(left, right=left.iloc[:1], sort=sort, return_count=return_count)[::2]
        else:
            res = factorize(left, right=left[:1], sort=sort, return_count=return_count)[::2]

        return res if return_count else res[0]
    elif len(left) == 0:
        lkey = np.empty(0, dtype=np.int64)
        res = factorize(right, sort=sort, return_count=return_count)
        return ((lkey,) + res) if isinstance(res, tuple) else (lkey, res)
    elif len(right) == 0:
        rkey = np.empty(0, dtype=np.int64)
        res = factorize(left, sort=sort, return_count=return_count)
        return (res[0], rkey, res[1]) if return_count else (res, rkey)
    else:
        if isinstance(left, pd.DataFrame):
            if isinstance(right, pd.DataFrame):
                if left.shape[1] != right.shape[1]:
                    raise ValueError(
                        "left and right must have the same number of columns,"
                        f" but got {left.shape[1]} and {right.shape[1]}"
                    )
                mapped = (
                    factorize(s, r, sort=sort, return_count=True) for (_, s), (_, r) in zip(left.items(), right.items())
                )
                zipped = zip(*mapped)
                lcodes, rcodes, shape = (list(x) for x in zipped)
                shape = tuple(shape)
            elif isinstance(right, pd.MultiIndex):
                if left.shape[1] != right.nlevels:
                    raise ValueError(
                        "left and right must have the same number of columns/levels,"
                        f" but got {left.shape[1]} and {right.nlevels}"
                    )
                lcodes, rcodes, shape = _factorize_df_multiindex(left, right, sort, False)
            else:
                raise ValueError(
                    f"if left is a DataFrame, right must be a DataFrame or MultiIndex, but got {type(right)}"
                )

            # get flat i8 join keys
            lkey, rkey = pd.core.reshape.merge._get_join_keys(lcodes, rcodes, shape, sort)
            return (lkey, rkey, len(np.union1d(lkey, rkey))) if return_count else (lkey, rkey)
        elif isinstance(left, pd.MultiIndex):
            if isinstance(right, pd.DataFrame):
                if left.nlevels != right.shape[1]:
                    raise ValueError(
                        "left and right must have the same number of columns/levels,"
                        f" but got {left.nlevels} and {right.shape[1]}"
                    )
                lcodes, rcodes, shape = _factorize_df_multiindex(right, left, sort, True)
            elif isinstance(right, pd.MultiIndex):
                if left.nlevels != right.nlevels:
                    raise ValueError(
                        "left and right must have the same number of levels,"
                        f" but got {left.nlevels} and {right.nlevels}"
                    )
                mapped = (
                    factorize(left.levels[i]._values, right.levels[i]._values, sort=sort, return_count=True)
                    for i in range(left.nlevels)
                )
                zipped = zip(*mapped)
                lcodes, rcodes, shape = (list(x) for x in zipped)
                lcodes = list(map(np.take, lcodes, left.codes))
                rcodes = list(map(np.take, rcodes, right.codes))

                # fix labels if there were any nulls
                for i in range(left.nlevels):
                    lmask = left.codes[i] == -1
                    rmask = right.codes[i] == -1
                    lany = lmask.any()
                    rany = rmask.any()
                    if lany or rany:
                        if lany:
                            lcodes[i][lmask] = shape[i]
                        if rany:
                            rcodes[i][rmask] = shape[i]
                        shape[i] += 1

                shape = tuple(shape)
            else:
                raise ValueError(
                    f"if left is a MultiIndex, right must be a DataFrame or MultiIndex, but got {type(right)}"
                )

            # get flat i8 join keys
            lkey, rkey = pd.core.reshape.merge._get_join_keys(lcodes, rcodes, shape, sort)
            return (lkey, rkey, len(np.union1d(lkey, rkey))) if return_count else (lkey, rkey)
        else:
            if isinstance(right, pd.DataFrame) or isinstance(right, pd.MultiIndex):
                raise ValueError(f"if left is an array, right must be an array, too, but got {type(right)}")
            else:
                res = pd.core.reshape.merge._factorize_keys(left, right, sort=sort)
                return res if return_count else res[:2]


def _parse_column_specs(df: Union[pd.DataFrame, "dask.dataframe.DataFrame"], spec) -> list:  # type: ignore # noqa F821
    if isinstance(spec, (tuple, str, int, np.ndarray, pd.Series)):
        spec = [spec]
    out = []
    for s in spec:
        if isinstance(s, int):
            if s < 0:
                s += df.index.nlevels
            assert 0 <= s < df.index.nlevels
        elif isinstance(s, str):
            if s not in df.columns:
                s = list(df.index.names).index(s)
        elif isinstance(s, tuple):
            assert s in df.columns
        elif isinstance(df, pd.DataFrame):
            assert len(s) == len(df)
        out.append(s)

    return out


def _can_impute(
    df: Union[pd.DataFrame, pd.Series],
    group_values: np.ndarray,
    forward: bool = False,
    backward: bool = False,
    union: bool = True,
) -> Union[list, pd.DataFrame, pd.Series]:
    # tacit assumptions:
    # * `group_values` is not mixed up
    out = []
    is_na = df.isna()
    with np.testing.suppress_warnings("always") as sup:
        sup.filter(RuntimeWarning)
        template = 0.0 / (1.0 - is_na.astype(np.float32))  # NaN stays NaN, other values are 0

    if forward:
        shift = [(-1, template)]
    else:
        shift = []
    if backward:
        shift.append((1, template.copy()))

    for s, mask_cur in shift:
        mask0 = shift_unequal(group_values, s, fill_value=True)

        if isinstance(df, pd.DataFrame):
            for c in df.columns:
                mask_cur.loc[is_na[c] & mask0, c] = 1.0
        else:
            mask_cur[is_na & mask0] = 1.0

        if s < 0:
            mask_cur.ffill(axis=0, inplace=True)
        else:
            mask_cur.bfill(axis=0, inplace=True)

        # all values in `mask_cur` are either 0 or 1; 1-values correspond to entries that cannot be imputed
        mask_cur = mask_cur.astype(bool)

        if union and len(out) > 0:
            out[0] |= mask_cur
        else:
            out.append(mask_cur)

    return out[0] if len(out) == 1 else out


def _factorize_df_multiindex(
    df: pd.DataFrame, index: pd.MultiIndex, sort: bool, swap: bool
) -> tuple[np.ndarray, np.ndarray, tuple]:
    mapped = (
        factorize(np.asarray(index.levels[i]._values), s.values, sort=sort, return_count=True)
        for i, (_, s) in enumerate(df.items())
    )
    zipped = zip(*mapped)
    rcodes, lcodes, shape = (list(x) for x in zipped)
    if sort and not swap:
        rcodes = list(map(np.take, rcodes, index.codes))
    else:
        rcodes = [a.astype("i8", subok=False, copy=True) for a in index.codes]

    # fix right labels if there were any nulls
    for i, (_, s) in enumerate(df.items()):
        mask = index.codes[i] == -1
        if mask.any():
            # check if there already was any nulls at this location
            # if there was, it is factorized to `shape[i] - 1`
            a = s.values[lcodes[i] == shape[i] - 1]
            if a.size == 0 or not a[0] != a[0]:
                shape[i] += 1

            rcodes[i][mask] = shape[i] - 1

    if swap:
        lcodes, rcodes = rcodes, lcodes

    return lcodes, rcodes, tuple(shape)
