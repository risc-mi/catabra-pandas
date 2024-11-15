#  Copyright (c) 2024. RISC Software GmbH.
#  All rights reserved.

import numpy as np
import pandas as pd

from catabra_pandas import merge_intervals


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
