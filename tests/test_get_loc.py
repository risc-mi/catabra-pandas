#  Copyright (c) 2025. RISC Software GmbH.
#  All rights reserved.

import numpy as np
import pandas as pd
import pytest

from catabra_pandas import get_loc


def _compare_loc(index: pd.Index, target):
    res = get_loc(index, target, allow_mask=False, allow_slice=False)
    out = pd.Series(index=index, data=pd.RangeIndex(len(index))).loc[target]
    if isinstance(out, pd.Series):
        out = out.values
    return np.all(res == out)


def _test_single(index: pd.Index, rng):
    target = rng.choice(index.get_level_values(0))
    assert _compare_loc(index, target)

    try:
        get_loc(index, 123456789)
    except KeyError:
        pass
    else:
        assert False

    target = rng.choice(index.get_level_values(0), size=rng.randint(len(index) // 2))
    assert _compare_loc(index, target)

    target = target.tolist()
    assert _compare_loc(index, target)

    target[-1] = -1
    try:
        get_loc(index, target)
    except KeyError:
        pass
    else:
        assert False

    target = rng.uniform(0, 1, size=len(index)) > 0.8
    assert _compare_loc(index, target)

    target = target.tolist()
    assert _compare_loc(index, target)

    target = rng.uniform(0, 1, size=(len(index), 5)) > 0.8
    assert _compare_loc(index, target)

    try:
        get_loc(index, rng.uniform(0, 1, size=len(index) + 1) > 0.8)
    except IndexError:
        pass
    else:
        assert False


@pytest.mark.parametrize(
    argnames=["seed"],
    argvalues=[
        (42,),
        (1234,),
        (2904,),
        (98765,),
        (34,),
        (753159,),
        (37008,),
        (115,),
        (0,),
        (6657,),
        (90270,),
    ],
)
def test_unique(seed: int):
    rng = np.random.RandomState(seed=seed)
    index = pd.Index(rng.choice(1000, size=rng.randint(5, 500), replace=False))
    _test_single(index, rng)
    q1, q2 = np.quantile(index.get_level_values(0), [0.25, 0.75], method="nearest")
    target = slice(q1, q2)
    assert _compare_loc(index, target)


@pytest.mark.parametrize(
    argnames=["seed"],
    argvalues=[
        (43,),
        (1235,),
        (2905,),
        (98766,),
        (35,),
        (753160,),
        (37009,),
        (116,),
        (1,),
        (6658,),
        (90271,),
    ],
)
def test_non_unique(seed: int):
    rng = np.random.RandomState(seed=seed)
    index = pd.Index(rng.choice(200, size=rng.randint(100, 500), replace=True))
    _test_single(index, rng)


@pytest.mark.parametrize(
    argnames=["seed"],
    argvalues=[
        (41,),
        (1233,),
        (2903,),
        (98764,),
        (33,),
        (753158,),
        (37007,),
        (114,),
        (9999999,),
        (6656,),
        (90269,),
    ],
)
def test_multi_complete(seed: int):
    rng = np.random.RandomState(seed=seed)
    l1 = rng.choice(list("abcdefghijklmnopqrstuvwxyz_0123456789"), size=rng.randint(3, 20), replace=False)
    l2 = rng.choice(100, size=rng.randint(10, 50), replace=True)
    index = pd.MultiIndex.from_product([l1, l2])

    target = (rng.choice(l1), rng.choice(l2))
    assert _compare_loc(index, target)

    try:
        get_loc(index, ("???", 5))
    except KeyError:
        pass
    else:
        assert False

    target = list(
        zip(rng.choice(l1, size=rng.randint(len(index) // 2)), rng.choice(l2, size=rng.randint(len(index) // 2)))
    )
    assert _compare_loc(index, target)

    target[-1] = (l1[0], -1)
    try:
        get_loc(index, target)
    except KeyError:
        pass
    else:
        assert False

    target = rng.uniform(0, 1, size=len(index)) > 0.8
    assert _compare_loc(index, target)

    target = target.tolist()
    assert _compare_loc(index, target)

    target = rng.uniform(0, 1, size=(len(index), 5)) > 0.8
    assert _compare_loc(index, target)

    try:
        get_loc(index, (rng.uniform(0, 1, size=len(index) + 1) > 0.8).tolist())
    except IndexError:
        pass
    else:
        assert False


@pytest.mark.parametrize(
    argnames=["seed"],
    argvalues=[
        (51,),
        (2233,),
        (3903,),
        (108764,),
        (43,),
        (853158,),
        (47007,),
        (214,),
        (10999999,),
        (7656,),
        (100269,),
    ],
)
def test_multi_partial(seed: int):
    rng = np.random.RandomState(seed=seed)
    l2 = rng.choice(list("abcdefghijklmnopqrstuvwxyz_0123456789"), size=rng.randint(3, 20), replace=False)
    l1 = rng.choice(100, size=rng.randint(10, 50), replace=True)
    index = pd.MultiIndex.from_product([l1, l2])
    _test_single(index, rng)
