#  Copyright (c) 2024. RISC Software GmbH.
#  All rights reserved.

import warnings
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .misc import factorize


def merge_intervals(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on=None,
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    how: str = "left",
    left_start: Optional[str] = None,
    left_stop: Optional[str] = None,
    right_start: Optional[str] = None,
    right_stop: Optional[str] = None,
    include_left_start: bool = True,
    include_left_stop: bool = True,
    include_right_start: bool = True,
    include_right_stop: bool = True,
    suffixes: tuple = ("_x", "_y"),
    keep_order: bool = True,
    copy: Optional[bool] = None,
    return_indexers: bool = False,
) -> Union[pd.DataFrame, np.ndarray]:
    """Join columns of two DataFrames based on interval overlaps.

    Parameters
    ----------
    left : pd.DataFrame
        First DataFrame to consider. Must have columns `left_start`, `left_stop` and `on`, if specified.
    right : pd.DataFrame | list of pd.DataFrame
        Other DataFrame(s) to consider. Must have columns `right_start`, `right_stop` and `on`, if specified.
    on : list of str, optional
        List of columns on which to join, in addition to intervals.
        Mutually exclusive with `left_on`, `right_on`, `left_index` and `right_index`.
    left_on : list of str, optional
        List of columns in `left` on which to join, in addition to intervals.
        Mutually exclusive with `on` and `left_index`.
    right_on : list of str, optional
        List of columns in `right` on which to join, in addition to intervals.
        Mutually exclusive with `on` and `right_index`.
    left_index : list | bool
        List of index levels in `left` on which to join, in addition to intervals, or True to join on all levels.
        Mutually exclusive with `on` and `left_on`.
    right_index : list | bool
        List of index levels in `right` on which to join, in addition to intervals, or True to join on all levels.
        Mutually exclusive with `on` and `right_on`.
    how : str, default="left"
        How to handle the operation of the two objects:
        * "inner": The result contains a row for all combinations of rows in `left` and `right` with non-empty
            intersection. The order of rows corresponds to that of `left`.
        * "left": Like "inner", but add all rows in `left` that would be missing.
        * "right": Like "left", but with `left` and `right` swapped. Note that this only affects the order of rows,
            not the order of columns in the result.
        * "outer": Like "inner", but add all rows in `left` and `right` that would be missing.
    left_start : str, optional
        Name of the column in `left` that contains left interval endpoints. If None, the intervals do not have a left
        endpoint. If equal to `left_stop`, the intervals are treated as isolated points instead.
    left_stop : str, optional
        Name of the column in `left` that contains right interval endpoints. If None, the intervals do not have a right
        endpoint.
    right_start : str, optional
        Name of the column in `right` that contains left interval endpoints. If None, the intervals do not have a left
        endpoint. If equal to `right_stop`, the intervals are treated as isolated points instead.
    right_stop : str, optional
        Name of the column in `right` that contains right interval endpoints. If None, the intervals do not have a
        right endpoint.
    include_left_start : bool, default=True
        Include left interval endpoints of `left`. Ignored if `left_start` is None.
    include_left_stop : bool, default=True
        Include right interval endpoints of `left`. Ignored if `left_stop` is None.
    include_right_start : bool, default=True
        Include left interval endpoints of `right`. Ignored if `right_start` is None.
    include_right_stop : bool, default=True
        Include right interval endpoints of `right`. Ignored if `right_stop` is None.
    suffixes : (str, str), default=("_x", "_y")
        A length-2 sequence where each element is optionally a string indicating the suffix to add to overlapping
        column names in `left` and `right` respectively. Pass a value of None to indicate that the column name from
        `left` or `right` should be left as-is, with no suffix. At least one of the values must not be None.
        Ignored if `return_indexer` is True.
    keep_order : bool, default=True
        Maintain the order of rows in `right` (or `left` if `how` is "right") in the result. That means, if a row in
        `left` has multiple matching rows in `right`, their order is as in `right`.
        Setting this parameter to False avoids some internal sorting, thereby increasing efficiency.
    copy : bool, optional
        If False, avoid copy if possible. Ignored if `return_indexers` is True.
    return_indexers : bool, default=False
        Return indexers instead of the join of `left` and `right`.

    Returns
    -------
    pd.DataFrame | array
        If `return_indexers` is False: the join of `left` and `right`. In contrast to `pd.merge()`, the result always
        has a fresh RangeIndex, even when setting `left_index` or `right_index` to True.
        If `return_indexers` is True: array of shape `(2, N)` that describe which rows of `left` to combine with which
        rows of `right` (`iloc`-indices). -1 refers to missing rows, if `how` is not "inner".

    Notes
    -----

    Data types:
    Interval data types can be arbitrary, as long as they match between `left` and `right` and support sorting with
    `np.argsort` and `np.lexsort`. It is important to note that an interval is defined as the set of all points greater
    than (or equal) to the left endpoint, and less than (or equal) to the right endpoint, assuming a continuum as the
    underlying set from which points can be drawn. Hence, any interval whose left endpoint is strictly less than its
    right endpoint is automatically non-empty. For instance, the interval `(0, 1)` with integer endpoints is non-empty,
    even though it does not contain any integers.

    Empty and infinite intervals:
    Intervals in `left` and `right` may be empty (i.e., have larger left- than right endpoint) or infinite. Such
    intervals are handled correctly by this function. However, the behavior of this function is undefined if there are
    intervals of the form `(-inf, -inf)` or `(+inf, +inf)`, either explicitly or implicitly by setting some of
    `left_start` etc. to None.

    NaN interval endpoints:
    Intervals with NaN endpoints are _always_ treated like empty intervals, even if the other endpoint is +/-inf.

    Order of rows:
    The order of rows in the result follows their order in `left` (or `right` if `how` is set to "right"). If a row in
    `left` has multiple matching rows in `right`, their order is as in `right` if `keep_order` is True, and random
    otherwise.
    If `how` is "left", missing rows are inserted once at their original positions. If `how` is "outer", missing rows
    from `left` are inserted at their original positions, and then missing rows from `right` are inserted at the end.
    This deviates from `pd.merge()`, because intervals do not possess a natural sorting order: they could be sorted
    lexicographically, reverse-lexicographically, by size, etc.
    """

    if how == "right":
        indexers = merge_intervals(
            right,
            left,
            how="left",
            on=on,
            left_on=right_on,
            right_on=left_on,
            left_index=right_index,
            right_index=left_index,
            left_start=right_start,
            left_stop=right_stop,
            right_start=left_start,
            right_stop=left_stop,
            include_left_start=include_right_start,
            include_left_stop=include_right_stop,
            include_right_start=include_left_start,
            include_right_stop=include_left_stop,
            keep_order=keep_order,
            return_indexers=True,
        )[::-1]

        if return_indexers:
            return indexers
        else:
            return _reindex_and_concat(left, right, indexers, suffixes, copy)

    if how not in ("left", "outer", "inner"):
        raise ValueError(f'`how` must be one of "left", "right", "outer" or "inner", but got "{how}"')

    if on is None:
        if left_on is None:
            if left_index is False:
                left_on = []
            elif left_index is True:
                left_on = list(range(left.index.nlevels))
            elif isinstance(left_index, int):
                left_on = [left_index]
                left_index = True
            elif isinstance(left_index, list):
                left_on = left_index
                left_index = True
            else:
                raise ValueError(f"`left_index` must be bool, integer or list, but got {type(left_index)}")
        else:
            if left_index is not False:
                raise ValueError("Can only pass argument `left_on` OR `left_index`, not both.")
            elif not isinstance(left_on, list):
                left_on = [left_on]
        if right_on is None:
            if right_index is False:
                right_on = []
            elif right_index is True:
                right_on = list(range(right.index.nlevels))
            elif isinstance(right_index, int):
                right_on = [right_index]
                right_index = True
            elif isinstance(right_index, list):
                right_on = right_index
                right_index = True
            else:
                raise ValueError(f"`right_index` must be bool, integer or list, but got {type(right_index)}")
        else:
            if right_index is not False:
                raise ValueError("Can only pass argument `right_on` OR `right_index`, not both.")
            elif not isinstance(right_on, list):
                right_on = [right_on]
        if len(left_on) != len(right_on):
            raise ValueError(
                f"`left_on` and `right_on` must have the same length, but got {len(left_on)} and {len(right_on)}"
            )
    else:
        if left_on is not None or right_on is not None:
            raise ValueError("Can only pass argument `on` OR `left_on` and `right_on`, not a combination of both.")
        elif left_index is not False or right_index is not False:
            raise ValueError(
                "Can only pass argument `on` OR `left_index` and `right_index`, not a combination of both."
            )
        elif not isinstance(on, list):
            on = [on]
        left_on = right_on = on

    # reset the row index; makes everything a lot easier
    left_orig = left
    right_orig = right
    left = left.reset_index(drop=True, inplace=False)
    right = right.reset_index(drop=True, inplace=False)

    # get rid of empty intervals
    if left_start is None:
        if left_stop is not None:
            mask = left[left_stop].notna()
            if not mask.all():
                left = left[mask]
    elif left_stop is None:
        mask = left[left_start].notna()
        if not mask.all():
            left = left[mask]
    elif left_start == left_stop:
        if include_left_start and include_left_stop:
            mask = left[left_start].notna()
            if not mask.all():
                left = left[mask]
        else:
            warnings.warn(
                "If `left` is meant to be joined on isolated points,"
                " `include_left_start` and `include_left_stop` should be True."
            )
            left = left.iloc[:0]
    else:
        # removes rows with NaN endpoints
        mask = (
            left[left_start] <= left[left_stop]
            if include_left_start and include_left_stop
            else left[left_start] < left[left_stop]
        )
        if not mask.all():
            left = left[mask]
    if right_start is None:
        if right_stop is not None:
            mask = right[right_stop].notna()
            if not mask.all():
                right = right[mask]
    elif right_stop is None:
        mask = right[right_start].notna()
        if not mask.all():
            right = right[mask]
    elif right_start == right_stop:
        if include_right_start and include_right_stop:
            mask = right[right_start].notna()
            if not mask.all():
                right = right[mask]
        else:
            warnings.warn(
                "If `right` is meant to be joined on isolated points,"
                " `include_right_start` and `include_right_stop` should be True."
            )
            right = right.iloc[:0]
    else:
        # removes rows with NaN endpoints
        mask = (
            right[right_start] <= right[right_stop]
            if include_right_start and include_right_stop
            else right[right_start] < right[right_stop]
        )
        if not mask.all():
            right = right[mask]

    if len(left_on):
        if left_index:
            lfactor = left_orig.index[left.index]
            drop = list(set(range(lfactor.nlevels)).difference(left_on))
            if drop:
                lfactor = lfactor.droplevel(drop)
            left_on = np.argsort(left_on)
            if not (left_on[:-1] < left_on[1:]).all():
                lfactor = lfactor.reorder_levels(left_on)
        else:
            lfactor = left[left_on]
        if right_index:
            rfactor = right_orig.index[right.index]
            drop = list(set(range(rfactor.nlevels)).difference(right_on))
            if drop:
                rfactor = rfactor.droplevel(drop)
            right_on = np.argsort(right_on)
            if not (right_on[:-1] < right_on[1:]).all():
                rfactor = rfactor.reorder_levels(right_on)
        else:
            rfactor = right[right_on]

        left_on, right_on = factorize(lfactor, right=rfactor, sort=False, return_count=False)

        # restrict to intersection wrt. equality constraints
        mask = np.isin(left_on, right_on)
        if not mask.all():
            left = left[mask]
            left_on = left_on[mask]

        mask = np.isin(right_on, left_on)
        if not mask.all():
            right = right[mask]
            right_on = right_on[mask]
    else:
        left_on = right_on = None

    # `left` and `right` still have their original order, i.e., strictly increasing row indexes
    # some rows might have been removed, though
    if (
        (left_start is None and left_stop is None)
        or (right_start is None and right_stop is None)
        or (left_start is None and right_start is None)
        or (left_stop is None and right_stop is None)
    ):
        if left_on is None:
            # cross join => not allowed
            raise ValueError("No columns to perform merge on.")
        else:
            # standard equi-join on `left_on` and `right_on`
            indexer = _get_equi_join_indexers([left_on], [right_on], left_indexer=left.index, right_indexer=right.index)
    elif left_start == left_stop and right_start == right_stop:
        if left_on is None:
            # standard equi-join on `left_start` and `right_start`
            indexer = _get_equi_join_indexers(
                [left[left_start].values],
                [right[right_start].values],
                left_indexer=left.index,
                right_indexer=right.index,
            )
        else:
            # standard equi-join on `[left_on, left_start]` and `[right_on, right_start]`
            indexer = _get_equi_join_indexers(
                [left_on, left[left_start].values],
                [right_on, right[right_start].values],
                left_indexer=left.index,
                right_indexer=right.index,
            )
    else:
        if (
            left_start is None
            or left_stop is None
            or right_start is None
            or right_stop is None
            or right_start == right_stop
        ):
            # right endpoints contained in left intervals

            if left_start == left_stop:
                # we have either left_start \in (-inf, right_stop) or left_start \in (right_start, inf)
                # the first is equivalent to right_stop \in (left_start, inf)
                # the second is equivalent to right_start \in (-inf, left_stop)
                if right_start is None:
                    left_stop = None
                    include_left_start = include_right_stop
                else:
                    left_start = None
                    include_left_stop = include_right_start

            right_col = right_stop if left_stop is None or right_start is None else right_start

            if left_on is not None:
                sorter = np.argsort(left_on)
                left = left.iloc[sorter]
                left_on = left_on[sorter]

            sorter = _grouped_lexsort(right, right_on, [right_col], return_indexer=True, validate_groups=True)
            right = right.iloc[sorter]
            if right_on is not None:
                right_on = right_on[sorter]

            spec = _find_contained_points(
                left,
                right[right_col],
                left_on,
                right_on,
                [(left_start, include_left_start, left_stop, include_left_stop)],
            )
            spec[0].sort_index(inplace=True)
            indexer = _explode(spec[0], right.index)  # this is the "inner" indexer, first row is increasing
        elif left_start == left_stop:
            # left start points contained in right intervals

            if right_on is not None:
                sorter = np.argsort(right_on)
                right = right.iloc[sorter]
                right_on = right_on[sorter]

            sorter = _grouped_lexsort(left, left_on, [left_start], return_indexer=True, validate_groups=True)
            left = left.iloc[sorter]
            if left_on is not None:
                left_on = left_on[sorter]

            spec = _find_contained_points(
                right,
                left[left_start],
                right_on,
                left_on,
                [(right_start, include_right_start, right_stop, include_right_stop)],
            )
            indexer = _explode(spec[0], left.index)[::-1]  # this is the "inner" indexer
            if keep_order:
                sorter = _grouped_lexsort(
                    pd.Series(indexer[1]).to_frame("__tmp__"),
                    indexer[0],
                    ["__tmp__"],
                    return_indexer=True,
                    validate_groups=True,
                )
                keep_order = False  # no need to sort it again in _finalize_indexers
            else:
                sorter = np.argsort(indexer[0])
            indexer = indexer[:, sorter]
        else:
            # proper interval overlaps

            # let L be one of [, (, and R be one of ], )
            # two *non-empty* intervals La, bR, Lc, dR are overlapping iff
            #
            #       LR   LR        disj 1    OR    disj 2
            #  0    ()   ()     a in [c, d)     c in (a, b)
            #  1    ()   (]     a in [c, d)     c in (a, b)
            #  2    ()   [)     a in [c, d)     c in (a, b)
            #  3    ()   []     a in [c, d)     c in (a, b)
            #  4    (]   ()     a in [c, d)     c in (a, b)
            #  5    (]   (]     a in [c, d)     c in (a, b)
            #  6    (]   [)     a in [c, d)     c in (a, b]
            #  7    (]   []     a in [c, d)     c in (a, b]
            #  8    [)   ()     a in (c, d)     c in [a, b)
            #  9    [)   (]     a in (c, d]     c in [a, b)
            # 10    [)   [)     a in (c, d)     c in [a, b)
            # 11    [)   []     a in (c, d]     c in [a, b)
            # 12    []   ()     a in (c, d)     c in [a, b)
            # 13    []   (]     a in (c, d]     c in [a, b)
            # 14    []   [)     a in (c, d)     c in [a, b]
            # 15    []   []     a in (c, d]     c in [a, b]
            #
            # note that disj1 and disj2 are always mutually exclusive

            # sort `left` by `left_on` and `left_start` (a)
            sorter = _grouped_lexsort(left, left_on, [left_start], return_indexer=True, validate_groups=True)
            left = left.iloc[sorter]
            if left_on is not None:
                left_on = left_on[sorter]

            # sort `right` by `right_on` and `right_start` (c)
            sorter = _grouped_lexsort(right, right_on, [right_start], return_indexer=True, validate_groups=True)
            right = right.iloc[sorter]
            if right_on is not None:
                right_on = right_on[sorter]

            # a in Lc, dR
            spec = _find_contained_points(
                right,
                left[left_start],
                right_on,
                left_on,
                [(right_start, not include_left_start, right_stop, include_left_start and include_right_stop)],
            )
            indexer_a = _explode(spec[0], left.index)[::-1]  # this is the "inner" indexer of a in Lc, dR
            indexer_a = indexer_a[:, np.argsort(indexer_a[0])]

            # c in La, bR
            spec = _find_contained_points(
                left,
                right[right_start],
                left_on,
                right_on,
                [(left_start, include_left_start, left_stop, include_left_stop and include_right_start)],
            )
            spec[0].sort_index(inplace=True)
            indexer_c = _explode(spec[0], right.index)  # this is the "inner" indexer of c in La, bR

            # `indexer_a` and `indexer_c` are pairwise disjoint, so we can simply concatenate them
            # their first rows are both increasing, so we can use np.searchsorted to maintain increasing first row
            loc = np.searchsorted(indexer_a[0], indexer_c[0])
            indexer = np.insert(indexer_a, loc, indexer_c, axis=1)  # this is the "inner" indexer, increasing first row

    indexer = _finalize_indexers(indexer, len(left_orig), len(right_orig), how, keep_order)

    if return_indexers:
        return indexer
    else:
        return _reindex_and_concat(left_orig, right_orig, indexer, suffixes, copy)


def _explode(row_spec: pd.DataFrame, indexer: Optional[np.ndarray] = None) -> np.ndarray:
    n = row_spec["last"] - row_spec["first"] + 1
    out = np.empty((2, n.sum()), dtype=row_spec["last"].dtype)
    out[0] = np.repeat(row_spec.index, n)
    cs = np.roll(np.cumsum(n.values), 1)
    cs[:1] = 0
    out[1] = np.repeat(row_spec["first"].values - cs, n) + np.arange(out.shape[1], dtype=out.dtype)
    if indexer is not None:
        np.take(np.asarray(indexer, out.dtype), out[1], out=out[1])

    return out


def _get_equi_join_indexers(
    left_keys: List[np.ndarray],
    right_keys: List[np.ndarray],
    left_indexer: Optional[np.ndarray] = None,
    right_indexer: Optional[np.ndarray] = None,
) -> np.ndarray:
    # assumption: `left_indexer` and `right_indexer`, if given, are strictly increasing
    lidx, ridx = pd.core.reshape.merge.get_join_indexers(left_keys, right_keys, sort=False, how="inner")
    if lidx is None:
        if ridx is None:
            lidx = np.arange(len(left_keys[0]))
            ridx = lidx
        else:
            lidx = np.arange(len(ridx), dtype=ridx.dtype)
            ridx = np.asarray(ridx)
    else:
        # `lidx` is increasing
        lidx = np.asarray(lidx)
        if ridx is None:
            ridx = np.arange(len(lidx), dtype=lidx.dtype)
        else:
            ridx = np.asarray(ridx)

    if left_indexer is not None:
        np.take(np.asarray(left_indexer, lidx.dtype), lidx, out=lidx)
    if right_indexer is not None:
        np.take(np.asarray(right_indexer, ridx.dtype), ridx, out=ridx)

    return np.stack([lidx, ridx], axis=0)


def _finalize_indexers(indexers: np.ndarray, left_len: int, right_len: int, how: str, keep_order: bool) -> np.ndarray:
    # `indexers[0]` is assumed to be increasing
    # `how` cannot be "right"!

    if keep_order:
        # sort `indexers`
        sorter = _grouped_lexsort(
            pd.Series(indexers[1]).to_frame("__tmp__"),
            indexers[0],
            ["__tmp__"],
            return_indexer=True,
            validate_groups=False,
        )
        indexers = indexers[:, sorter]

    if how == "outer":
        missing_right = np.setdiff1d(np.arange(right_len), indexers[1])
    else:
        missing_right = []

    if how in ("left", "outer"):
        # add missing rows from left
        missing = np.setdiff1d(np.arange(left_len), indexers[0])
        if len(missing):
            loc = np.searchsorted(indexers[0], missing)
            missing = np.stack([missing, -np.ones(len(missing), dtype=missing.dtype)], axis=0)
            indexers = np.insert(indexers, loc, missing, axis=1)

    if len(missing_right):
        # add missing rows from right
        tmp = np.empty((2, indexers.shape[1] + len(missing_right)), dtype=indexers.dtype)
        tmp[:, : indexers.shape[1]] = indexers
        tmp[0, indexers.shape[1] :] = -1
        tmp[1, indexers.shape[1] :] = missing_right
        indexers = tmp

    return indexers


def _reindex_and_concat(left: pd.DataFrame, right: pd.DataFrame, indexers: np.ndarray, suffixes, copy) -> pd.DataFrame:
    llabels, rlabels = pd.core.reshape.merge._items_overlap_with_suffix(left.columns, right.columns, suffixes)

    left_missing = (indexers[0] < 0).any()
    right_missing = (indexers[1] < 0).any()

    if left_missing:
        left = left.reset_index(drop=True, inplace=False)
        left = left.reindex(indexers[0])

        if right_missing:
            right = right.reset_index(drop=True, inplace=False)
            right = right.reindex(indexers[1])
        else:
            right = right.iloc[indexers[1]]
    else:
        if right_missing:
            left = left.iloc[indexers[0]]
            right = right.reset_index(drop=True, inplace=False)
            right = right.reindex(indexers[1])
        else:
            # fast track, inspired by PyJanitor:
            # https://github.com/pyjanitor-devs/pyjanitor/blob/dev/janitor/functions/conditional_join.py#L1174
            dct = {}
            for key, (_, value) in zip(llabels, left.items()):
                dct[key] = value.values[indexers[0]]
            for key, (_, value) in zip(rlabels, right.items()):
                dct[key] = value.values[indexers[1]]
            return pd.DataFrame(dct, copy=copy)

    left.reset_index(drop=True, inplace=True)
    right.reset_index(drop=True, inplace=True)

    left.columns = llabels
    right.columns = rlabels

    return pd.concat([left, right], axis=1, copy=copy)


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
                        data=dict(first=int(points_idx.min()), last=int(points_idx.max())),
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
                    df["first"] = int(points_idx.min())
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
                    df["last"] = int(points_idx.max())
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
