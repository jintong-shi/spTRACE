"""Tests for CNV catalogue cleaning and event sampling."""

import numpy as np
import pandas as pd
import pytest

from sptrace.cnv_utils import clean_all_df, sample_uniform_event


def make_catalog():
    return pd.DataFrame(
        {
            "Chromosome": ["1", "chr2", "3", "X"],
            "Start": [100, 2000, 5000, 10],
            "End": [1100, 3000, 5100, 2_000_010],
            "Copy_Number": [3, 4, 2, 1],
            "Major_Copy_Number": [2, 2, 2, 1],
            "Minor_Copy_Number": [1, 2, 2, 0],
        }
    )


def test_chromosome_prefix_is_normalised():
    out = clean_all_df(make_catalog(), drop_normal=False)
    assert set(out["Chromosome"]) == {"chr1", "chr2", "chr3", "chrX"}


def test_drop_normal_removes_balanced_diploid_segments():
    out = clean_all_df(make_catalog(), drop_normal=True)
    balanced = (out["Major_Copy_Number"] == 2) & (out["Minor_Copy_Number"] == 2)
    assert not balanced.any()
    # rows 2 (2/2) and 1 (2/2) both drop; two segments remain
    assert len(out) == 2


def test_min_len_filters_short_segments():
    out = clean_all_df(make_catalog(), drop_normal=False, min_len=1_000_000)
    assert len(out) == 1
    assert out.iloc[0]["Chromosome"] == "chrX"


def test_columns_are_matched_case_insensitively():
    df = make_catalog().rename(
        columns={"Chromosome": "chromosome", "Copy_Number": "Total_Copy_Number"}
    )
    out = clean_all_df(df, drop_normal=False)
    assert list(out.columns) == [
        "Chromosome",
        "Start",
        "End",
        "Copy_Number",
        "Major_Copy_Number",
        "Minor_Copy_Number",
    ]


def test_missing_required_column_raises():
    df = make_catalog().drop(columns=["Minor_Copy_Number"])
    with pytest.raises(AssertionError, match="Missing column"):
        clean_all_df(df)


def test_everything_filtered_out_raises():
    df = make_catalog()
    with pytest.raises(ValueError, match="All CNV segments filtered out"):
        clean_all_df(df, drop_normal=False, min_len=10**12)


def test_non_numeric_coordinates_are_dropped():
    df = make_catalog()
    df["Start"] = df["Start"].astype(object)
    df.loc[0, "Start"] = "not-a-number"
    out = clean_all_df(df, drop_normal=False)
    assert len(out) == 3


def test_sample_uniform_event_returns_a_normalised_row():
    catalog = clean_all_df(make_catalog(), drop_normal=False)
    np.random.seed(0)
    row = sample_uniform_event(catalog)
    assert row["Chromosome"].startswith("chr")
    assert isinstance(row["row_id"], int)
    for col in ("Start", "End", "Major_Copy_Number", "Minor_Copy_Number"):
        assert isinstance(row[col], int)


def test_sample_uniform_event_is_seed_reproducible():
    catalog = clean_all_df(make_catalog(), drop_normal=False)
    np.random.seed(7)
    first = sample_uniform_event(catalog).to_dict()
    np.random.seed(7)
    assert sample_uniform_event(catalog).to_dict() == first


def test_sample_uniform_event_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        sample_uniform_event(pd.DataFrame())


def test_sample_uniform_event_reports_missing_columns():
    df = pd.DataFrame({"Chromosome": ["chr1"], "Start": [1]})
    with pytest.raises(ValueError, match="missing columns"):
        sample_uniform_event(df)
