# sptrace/cnv_utils.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict

def clean_all_df(
    all_df: pd.DataFrame,
    drop_normal: bool = True,
    min_len: int | None = None,
    standardize_chr: bool = True,
) -> pd.DataFrame:
    """
    Keep only the required columns and normalise their format.
    Expected columns: Chromosome, Start, End, Copy_Number, Major_Copy_Number, Minor_Copy_Number.
    """
    df = all_df.copy()

    # Tolerant column lookup
    def pick(*cands):
        for c in cands:
            if c in df.columns: return c
            lc = c.lower()
            for col in df.columns:
                if col.lower() == lc: return col
        raise AssertionError(f"Missing column among {cands}")

    df = df[[ pick("Chromosome"),
              pick("Start"),
              pick("End"),
              pick("Copy_Number","Total_Copy_Number"),
              pick("Major_Copy_Number"),
              pick("Minor_Copy_Number") ]]

    df.columns = ["Chromosome","Start","End","Copy_Number","Major_Copy_Number","Minor_Copy_Number"]

    if standardize_chr:
        df["Chromosome"] = df["Chromosome"].astype(str)
        df["Chromosome"] = df["Chromosome"].str.replace("^chr","",regex=True)
        df["Chromosome"] = "chr" + df["Chromosome"]

    for c in ["Start","End","Copy_Number","Major_Copy_Number","Minor_Copy_Number"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().astype({
        "Start":int,"End":int,"Copy_Number":int,
        "Major_Copy_Number":int,"Minor_Copy_Number":int
    })

    if drop_normal:
        df = df[~((df["Major_Copy_Number"]==2) & (df["Minor_Copy_Number"]==2))]

    if min_len is not None:
        df = df[(df["End"] - df["Start"] + 1) >= int(min_len)]

    if df.empty:
        raise ValueError("All CNV segments filtered out. Relax filters.")

    return df


REQUIRED_COLS = ["Chromosome", "Start", "End", "Major_Copy_Number", "Minor_Copy_Number"]

def sample_uniform_event(cnv_df: pd.DataFrame) -> pd.Series:
    if cnv_df is None or len(cnv_df) == 0:
        raise ValueError("cnv_df is empty.")

    # Make sure the required columns are present
    missing = [c for c in REQUIRED_COLS if c not in cnv_df.columns]
    if missing:
        raise ValueError(f"cnv_df missing columns: {missing}")

    # Draw a row at random
    ridx = np.random.randint(0, len(cnv_df))
    row = cnv_df.iloc[ridx].copy()

    # Add a row_id if absent, using the drawn position
    if "row_id" not in row.index:
        row["row_id"] = int(ridx)
    else:
        # force an int
        try:
            row["row_id"] = int(row["row_id"])
        except Exception:
            row["row_id"] = int(ridx)

    # Normalise dtypes and the chromosome prefix
    for c in ["Start", "End", "Major_Copy_Number", "Minor_Copy_Number"]:
        row[c] = int(row[c])
    chrom = str(row["Chromosome"])
    if not chrom.startswith("chr"):
        row["Chromosome"] = "chr" + chrom

    return row
