#  Copyright (c) 2024. RISC Software GmbH.
#  All rights reserved.

import numpy as np
import pandas as pd
import pytest

from catabra_pandas import factorize

from .util import create_random_series


def _convert_dtype(s: pd.Series, dtype: str) -> pd.Series:
    if dtype == "float":
        s = s.astype(np.float32) / 7
        s[s >= 0.3] = np.nan
        return s
    elif dtype == "timedelta":
        return pd.to_timedelta(s * 7317, unit="ms")
    elif dtype == "timestamp":
        return pd.Timestamp(0) + _convert_dtype(s, "timedelta")
    else:
        return s


@pytest.mark.parametrize(
    argnames=["n_rows", "n_cols", "index", "n_rows_right", "index_right", "seed"],
    argvalues=[
        (100, 1, False, 0, False, 0),
        (100, 1, False, 0, False, 1),
        (100, 1, False, 0, False, 2),
        (100, 1, False, 0, False, 3),
        (100, 1, True, 0, False, 42),
        (100, 1, True, 0, False, 43),
        (100, 1, True, 0, False, 44),
        (100, 1, True, 0, False, 45),
        (100, 3, False, 0, False, 95304),
        (100, 3, False, 0, False, 95305),
        (100, 3, False, 0, False, 95306),
        (100, 3, False, 0, False, 95307),
        (100, 3, True, 0, False, 774210),
        (100, 3, True, 0, False, 774211),
        (100, 3, True, 0, False, 774212),
        (100, 3, True, 0, False, 774213),
        (100, 1, False, 200, False, 3333),
        (100, 1, False, 200, False, 3334),
        (100, 1, False, 200, False, 1234),
        (100, 1, False, 200, False, 5678),
        (100, 1, True, 200, False, 52),
        (100, 1, True, 200, False, 53),
        (100, 1, True, 200, False, 54),
        (100, 1, True, 200, False, 55),
        (100, 2, True, 200, False, 324),
        (100, 2, True, 200, False, 325),
        (100, 2, True, 200, False, 326),
        (100, 2, True, 200, False, 327),
        (100, 1, False, 200, True, 98765),
        (100, 1, False, 200, True, 76543),
        (100, 1, False, 200, True, 54321),
        (100, 1, False, 200, True, 32109),
        (100, 2, False, 200, True, 8804),
        (100, 2, False, 200, True, 8805),
        (100, 2, False, 200, True, 8806),
        (100, 2, False, 200, True, 8807),
        (100, 2, False, 200, False, 640378),
        (100, 2, False, 200, False, 640379),
        (100, 2, False, 200, False, 640380),
        (100, 2, False, 200, False, 640381),
        (100, 2, True, 200, True, 50557),
        (100, 2, True, 200, True, 50558),
        (100, 2, True, 200, True, 50559),
        (100, 2, True, 200, True, 50560),
    ],
)
def test_random(n_rows: int, n_cols: int, index: bool, n_rows_right: int, index_right: bool, seed: int):
    rng = np.random.RandomState(seed=seed)
    dtypes = rng.choice(["int", "bool", "category", "str"], size=n_cols, replace=True)

    left_df = pd.DataFrame(data={f"c{i}": create_random_series(n_rows, dt, rng) for i, dt in enumerate(dtypes)})
    dtype_mapping = {}
    for k, v in left_df.items():
        if v.dtype.kind in "ui":
            r = rng.randint(4)
            if r == 0:
                continue
            elif r == 1:
                dtype_mapping[k] = "float"
            elif r == 2:
                dtype_mapping[k] = "timedelta"
            elif r == 3:
                dtype_mapping[k] = "timestamp"
            left_df[k] = _convert_dtype(v, dtype_mapping[k])
    if n_cols == 1:
        if index:
            left = left_df.set_index(left_df.columns[0], inplace=False, append=False).index
        else:
            left = left_df.iloc[:, 0]
    elif index:
        left = left_df.set_index(list(left_df.columns), inplace=False, append=False).index
    else:
        left = left_df

    sort = rng.uniform(0, 1) > 0.5

    if n_rows_right > 0:
        right_df = pd.DataFrame(data={f"c{i}": create_random_series(n_rows, dt, rng) for i, dt in enumerate(dtypes)})
        for k, v in dtype_mapping.items():
            right_df[k] = _convert_dtype(right_df[k], v)
        if n_cols == 1:
            if index_right:
                right = right_df.set_index(right_df.columns[0], inplace=False, append=False).index
            else:
                right = right_df.iloc[:, 0]
        elif index_right:
            right = right_df.set_index(list(right_df.columns), inplace=False, append=False).index
        else:
            right = right_df

        lkeys, rkeys, count = factorize(left, right=right, sort=sort, return_count=True)

        assert lkeys.dtype.kind == "i"
        assert rkeys.dtype.kind == "i"

        keys = np.concatenate([lkeys, rkeys])
        df = pd.concat([left_df, right_df], axis=0)
    else:
        keys, count = factorize(left, sort=sort, return_count=True)

        assert keys.dtype.kind == "i"
        df = left_df

    assert len(np.unique(keys)) == count
    assert (~df.duplicated()).sum() == count
    assert (df.groupby(keys).apply(lambda x: (~x.duplicated()).sum()) == 1).all()
