# CaTabRa-pandas

<p align="center">
  <a href="#About"><b>About</b></a> &bull;
  <a href="#Quickstart"><b>Quickstart</b></a> &bull;
  <a href="#References"><b>References</b></a> &bull;
  <a href="#Contact"><b>Contact</b></a> &bull;
  <a href="#Acknowledgments"><b>Acknowledgments</b></a>
</p>

[![Platform Support](https://img.shields.io/badge/python->=3.6-blue)]()
[![Platform Support](https://img.shields.io/badge/pandas->=1.0-blue)]()
[![Platform Support](https://img.shields.io/badge/platform-Linux%20|%20Windows%20|%20MacOS-blue)]()

## About

**CaTabRa-pandas** is a Python library with a couple of useful functions for efficiently working with [pandas](https://pandas.pydata.org/) DataFrames. In particular, many of these functions are concerned with DataFrames containing *intervals*, i.e., DataFrames with (at least) two columns `"start"` and `"stop"` defining the left and right endpoints of intervals.

**Highlights**:
* Resample observations with respect to arbitrary (possibly irregular, possibly overlapping) windows: `catabra_pandas.resample_eav` and `catabra_pandas.resample_interval`.
* Compute the intersection, union, difference, etc. of intervals: `catabra_pandas.combine_intervals`.
* Group intervals by their distance to each other: `catabra_pandas.group_intervals`.
* For each point in a given DataFrame, find the interval that contains it: `catabra_pandas.find_containing_interval`.
* Find the previous/next observation for each entry in a DataFrame of timestamped observations: `catabra_pandas.prev_next_values`.

Each of these functions lacks a native pandas implementation, and is implemented *extremely efficiently* in **CaTabRa-pandas**. DataFrames with 10M+ rows are no problem!

**[Dask](https://docs.dask.org/en/stable/index.html) DataFrames are partly supported, too.**

If you are interested in **CaTabRa-pandas**, you might be interested in **[CaTabRa](https://github.com/risc-mi/catabra)**, too: **CaTabRa** is a full-fledged tabular data analysis framework that enables you to calculate statistics, generate appealing visualizations and train machine learning models with a single command.

## Quickstart

**CaTabRa-pandas** has minimal requirements and can be installed in every environment with Python >= 3.6 and pandas >= 1.0.

Once installed, **CaTabRa-pandas** can be readily used:

```python
import pandas as pd
import catabra_pandas

# use-case: resample observations wrt. given windows
observations = pd.DataFrame(
    data={
        "subject_id": [0, 0, 0, 0, 1, 1],
        "attribute": ["HR", "Temp", "HR", "HR", "Temp", "HR"],
        "timestamp": [1, 1, 5, 7, 2, 3],
        "value": [82.7, 36.9, 79.5, 78.7, 37.2, 89.4]
    }
)
windows = pd.DataFrame(
    data={
        ("subject_id", ""): [0, 0, 1],
        ("timestamp", "start"): [0, 4, 1],
        ("timestamp", "stop"): [6, 8, 4]
    }
)
catabra_pandas.resample_eav(
    observations,
    windows,
    agg={
        "HR": ["mean", "p75", "r-1"],   # mean value, 75-th percentile, last observed value
        "Temp": ["count", "mode"]     # standard deviation, mode
    },
    entity_col="subject_id",
    time_col="timestamp",
    attribute_col="attribute",
    value_col="value"
)
```

```python
import pandas as pd
import catabra_pandas

# use-case: find containing intervals
# note: intervals must be pairwise disjoint (in each group)
intervals = pd.DataFrame(
    data={
        "subject_id": [0, 0, 1],
        "start": [0.5, 3.0, -10.7],
        "stop": [2.3, 10., 10.7]
    }
)
points = pd.DataFrame(
    data={
        "subject_id": [0, 0, 0, 1, 1],
        "point": [1.0, 2.5, 9.9, 0.0, -8.8]
    }
)
catabra_pandas.find_containing_interval(
    points,
    intervals,
    ["point"],
    start_col="start",
    stop_col="stop",
    group_by="subject_id"
)
```

## References

**If you use CaTabRa-pandas in your research, we would appreciate citing the following conference paper:**

* A. Maletzky, S. Kaltenleithner, P. Moser and M. Giretzlehner.
  *CaTabRa: Efficient Analysis and Predictive Modeling of Tabular Data*. In: I. Maglogiannis, L. Iliadis, J. MacIntyre
  and M. Dominguez (eds), Artificial Intelligence Applications and Innovations (AIAI 2023). IFIP Advances in
  Information and Communication Technology, vol 676, pp 57-68, 2023.
  [DOI:10.1007/978-3-031-34107-6_5](https://doi.org/10.1007/978-3-031-34107-6_5)

  ```
  @inproceedings{CaTabRa2023,
    author = {Maletzky, Alexander and Kaltenleithner, Sophie and Moser, Philipp and Giretzlehner, Michael},
    editor = {Maglogiannis, Ilias and Iliadis, Lazaros and MacIntyre, John and Dominguez, Manuel},
    title = {{CaTabRa}: Efficient Analysis and Predictive Modeling of Tabular Data},
    booktitle = {Artificial Intelligence Applications and Innovations},
    year = {2023},
    publisher = {Springer Nature Switzerland},
    address = {Cham},
    pages = {57--68},
    isbn = {978-3-031-34107-6},
    doi = {10.1007/978-3-031-34107-6_5}
  }
  ```

## Contact

If you have any inquiries, please open a GitHub issue.

## Acknowledgments

This project is financed by research subsidies granted by the government of Upper Austria. RISC Software GmbH is Member
of UAR (Upper Austrian Research) Innovation Network.