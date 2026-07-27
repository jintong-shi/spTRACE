#!/usr/bin/env python3
"""Convert simulated AD/DP matrices into Numbat's allele-count table.

Console entry point: ``stcnv-export-numbat``.

Numbat expects a long-format, gzip-compressed TSV with one row per
(cell, SNP) pair carrying non-zero depth, ordered by genomic coordinate. The
genetic map (used to annotate each SNP with its position in centimorgans) is
not redistributable, so its path must be supplied with ``--genetic-map``; the
file shipped with Eagle v2.4.1 as ``genetic_map_hg38_withX.txt.gz`` works.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse


def _normalize_chrom(x) -> str:
    """Strip a 'chr' prefix and any trailing '.0' from a chromosome label."""
    return str(x).lower().replace("chr", "").replace(".0", "")


def get_position_cM_table(chr_pos_vector, geneticmap_file):
    """Interpolate the genetic position (cM) of each (chrom, pos) pair.

    Column names in the genetic map are detected case-insensitively, so both
    Eagle- and shapeit-style maps are accepted.
    """
    try:
        df = pd.read_csv(geneticmap_file, sep=r"\s+")
    except (ValueError, pd.errors.ParserError):
        df = pd.read_csv(geneticmap_file, sep="\t")

    cols_lower = {c.lower(): c for c in df.columns}
    col_chr = cols_lower.get("chr") or cols_lower.get("chrom") or cols_lower.get("chromosome")
    col_pos = cols_lower.get("pos") or cols_lower.get("position")
    col_cm = cols_lower.get("pos_cm") or cols_lower.get("genetic_map(cm)") or cols_lower.get("cm")

    if not col_cm:
        candidates = [c for c in df.columns if "cm" in c.lower()]
        if candidates:
            col_cm = candidates[-1]

    if not (col_chr and col_pos and col_cm):
        raise ValueError(
            f"Could not identify chromosome/position/cM columns in {geneticmap_file}. "
            f"Found columns: {list(df.columns)}"
        )

    df["clean_chr"] = df[col_chr].apply(_normalize_chrom)
    input_df = pd.DataFrame(chr_pos_vector, columns=["chrom", "pos"])
    input_df["clean_chr"] = input_df["chrom"].apply(_normalize_chrom)

    result_cM = np.zeros(len(input_df))

    for chrom in input_df["clean_chr"].unique():
        ref_subset = df[df["clean_chr"] == chrom]
        if ref_subset.empty:
            continue
        ref_subset = ref_subset.sort_values(by=col_pos)
        group_mask = input_df["clean_chr"] == chrom
        result_cM[group_mask] = np.interp(
            input_df.loc[group_mask, "pos"].values,
            ref_subset[col_pos].values,
            ref_subset[col_cm].values,
        )

    return result_cM


def dump_numbat_sparse(ad_path, dp_path, snp_path, meta_path, output_path, genetic_map):
    """Write a Numbat allele-count table from the simulated AD/DP matrices."""
    print("Starting Numbat export...")

    # 1. Metadata -> barcodes
    print("Loading metadata...")
    if str(meta_path).endswith(".parquet"):
        meta = pd.read_parquet(meta_path)
    else:
        meta = pd.read_csv(meta_path, sep="\t")

    # Prefer an explicit cell_id / barcode column, else fall back to the index.
    if "cell_id" in meta.columns:
        barcodes = meta["cell_id"].astype(str).values
    elif "barcode" in meta.columns:
        barcodes = meta["barcode"].astype(str).values
    else:
        barcodes = meta.index.astype(str).values

    print(f"   Metadata has {len(barcodes)} cells.")

    # 2. Read the matrices and transpose to Numbat's orientation.
    print("Loading matrices...")
    try:
        # The simulator writes cells (rows) x SNPs (columns).
        dp_raw = scipy.io.mmread(dp_path)
        ad_raw = scipy.io.mmread(ad_path)

        print(f"   Input matrix shape: {dp_raw.shape} (rows=cells, cols=SNPs)")
        print("   -> Transposing to (SNPs x cells)...")
        dp_coo = dp_raw.transpose().tocoo()
        ad_csr = ad_raw.transpose().tocsr()  # CSR so the fancy indexing below is fast
    except Exception as exc:
        print(f"[Error] Reading matrix: {exc}")
        return 1

    # 3. Consistency checks
    if dp_coo.shape[1] != len(barcodes):
        print(
            f"[Warning] Transposed matrix columns ({dp_coo.shape[1]}) "
            f"!= barcodes ({len(barcodes)}); falling back to generated names."
        )
        barcodes = np.array([f"Spot_{i}" for i in range(dp_coo.shape[1])])

    print("Loading SNP metadata...")
    df_snps = pd.read_csv(snp_path)

    if len(df_snps) != dp_coo.shape[0]:
        print(
            f"[Error] SNP file rows ({len(df_snps)}) != matrix rows "
            f"({dp_coo.shape[0]}) after transpose."
        )
        return 1

    # 4. Genetic distance in centimorgans
    print("   Calculating genetic distances (cM)...")
    try:
        snp_coords = list(zip(df_snps["chrom"].values, df_snps["pos"].values))
        df_snps["cM"] = get_position_cM_table(snp_coords, geneticmap_file=genetic_map)
    except Exception as exc:
        print(f"[Error] Failed in cM calculation: {exc}")
        return 1

    # 5. Assemble the long-format table
    print("Constructing Numbat table...")

    # After transposing, dp_coo.row indexes SNPs and dp_coo.col indexes cells.
    current_snps = df_snps.iloc[dp_coo.row]
    snp_ids = current_snps["snp_id"].values

    # SNP IDs look like "<chrom>_<pos>_<ref>_<alt>"; fall back to N/N if not.
    split_ids = pd.Series(snp_ids).str.split("_", expand=True)
    if split_ids.shape[1] >= 4:
        refs = split_ids[2].values
        alts = split_ids[3].values
    else:
        refs = ["N"] * len(snp_ids)
        alts = ["N"] * len(snp_ids)

    genes = (
        current_snps["gene_id"].values if "gene_id" in current_snps else ["NA"] * len(snp_ids)
    )

    df_numbat = pd.DataFrame(
        {
            "cell": barcodes[dp_coo.col],
            "snp_id": snp_ids,
            "CHROM": current_snps["chrom"].apply(_normalize_chrom).values,
            "POS": current_snps["pos"].values,
            "cM": current_snps["cM"].values,
            "REF": refs,
            "ALT": alts,
            "AD": np.asarray(ad_csr[dp_coo.row, dp_coo.col]).ravel(),
            "DP": dp_coo.data,
            "GT": "0|1",
            "gene": genes,
        }
    )

    df_numbat = df_numbat[df_numbat["DP"] > 0]

    print("   Sorting output...")
    # Numbat requires genomic-coordinate order, then cell order.
    df_numbat.sort_values(by=["CHROM", "POS", "cell"], inplace=True)

    print(f"Saving to {output_path}...")
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    df_numbat.to_csv(output_path, sep="\t", index=False, compression="gzip")
    print("Done.")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="stcnv-export-numbat",
        description="Convert simulated AD/DP matrices to a Numbat allele-count table.",
    )
    parser.add_argument(
        "--baf-dir",
        "--baf_dir",
        dest="baf_dir",
        required=True,
        help="Directory holding AD.mtx and DP.mtx (the simulator's baf/ output)",
    )
    parser.add_argument(
        "--snp-file", "--snp_file", dest="snp_file", required=True, help="SNP info CSV"
    )
    parser.add_argument(
        "--meta-file",
        "--meta_file",
        dest="meta_file",
        required=True,
        help="Cell/spot metadata table with a cell_id or barcode column",
    )
    parser.add_argument(
        "--genetic-map",
        "--geneticmap",
        dest="genetic_map",
        required=True,
        help="Genetic map with chromosome, position and cM columns "
        "(e.g. Eagle's genetic_map_hg38_withX.txt.gz)",
    )
    parser.add_argument("--output", required=True, help="Output .tsv.gz path")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    ad_file = os.path.join(args.baf_dir, "AD.mtx")
    dp_file = os.path.join(args.baf_dir, "DP.mtx")
    for path in (ad_file, dp_file, args.snp_file, args.meta_file, args.genetic_map):
        if not os.path.exists(path):
            print(f"[Error] Input not found: {path}", file=sys.stderr)
            return 1

    return dump_numbat_sparse(
        ad_file, dp_file, args.snp_file, args.meta_file, args.output, args.genetic_map
    )


if __name__ == "__main__":
    raise SystemExit(main())
