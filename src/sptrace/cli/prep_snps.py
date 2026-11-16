#!/usr/bin/env python3
"""Map a list of SNP IDs onto genes, producing the simulator's snp_file input.

Console entry point: ``sptrace-prep-snps``.

Input SNP IDs are expected in ``<chrom>_<pos>_<ref>_<alt>`` form, e.g.
``1_12345_A_C``. Only the chromosome and position drive the mapping; the full ID
is carried through so REF/ALT survive into the Numbat export.

The output CSV has columns ``snp_id, gene_id, chrom, pos`` and is what
``[files].snp_file`` in a design file should point at.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_snp_ids(path: Path) -> np.ndarray:
    """Load SNP IDs from a .npy array or a single-column text/CSV file."""
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True)

    sep = "\t" if path.suffix in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)
    for col in ("snp_id", "snp", "id"):
        if col in df.columns:
            return df[col].values
    return df.iloc[:, 0].values


def map_snps_to_genes(df_snp: pd.DataFrame, df_genes: pd.DataFrame) -> pd.DataFrame:
    """Assign each SNP to a gene whose [start, end] interval contains it."""
    mapped_rows = []
    common_chroms = set(df_snp["chrom"].unique()) & set(df_genes["chr_clean"].unique())

    for chrom in tqdm(sorted(common_chroms), desc="Processing chromosomes"):
        snps_on_chrom = df_snp[df_snp["chrom"] == chrom]
        genes_on_chrom = df_genes[df_genes["chr_clean"] == chrom]

        if snps_on_chrom.empty or genes_on_chrom.empty:
            continue

        for _, gene_row in genes_on_chrom.iterrows():
            mask = (snps_on_chrom["pos"] >= gene_row["start"]) & (
                snps_on_chrom["pos"] <= gene_row["end"]
            )
            found = snps_on_chrom[mask]
            if not found.empty:
                tagged = found.copy()
                tagged["gene_id"] = gene_row["gene"]
                mapped_rows.append(tagged)

    if not mapped_rows:
        return pd.DataFrame(columns=["snp_id", "gene_id", "chrom", "pos"])

    return pd.concat(mapped_rows, ignore_index=True)[["snp_id", "gene_id", "chrom", "pos"]]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="sptrace-prep-snps",
        description="Map SNP IDs onto genes to build the simulator's snp_file input.",
    )
    parser.add_argument(
        "--snp-ids",
        required=True,
        type=Path,
        help="SNP ID list: a .npy array, or a text/CSV file with one ID per row",
    )
    parser.add_argument(
        "--gene-table",
        required=True,
        type=Path,
        help="Gene annotation CSV with columns gene, chr, start, end "
        "(the same file used as [files].gene_table_file)",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    print(f"Loading SNP IDs from {args.snp_ids}...")
    snp_data = load_snp_ids(args.snp_ids)

    print("Parsing SNP strings...")
    df_snp = pd.DataFrame({"snp_id": pd.Series(snp_data).astype(str)})
    split_data = df_snp["snp_id"].str.split("_", expand=True)
    if split_data.shape[1] < 2:
        print(
            "[Error] SNP IDs do not look like '<chrom>_<pos>_<ref>_<alt>'. "
            f"First value: {df_snp['snp_id'].iloc[0]!r}",
            file=sys.stderr,
        )
        return 1

    df_snp["chrom"] = split_data[0].astype(str).str.replace("chr", "", regex=False)
    df_snp["pos"] = pd.to_numeric(split_data[1], errors="coerce")
    df_snp = df_snp.dropna(subset=["pos"])
    df_snp["pos"] = df_snp["pos"].astype(int)
    print(f"Total SNPs: {len(df_snp)}")

    print(f"Loading gene table from {args.gene_table}...")
    df_genes = pd.read_csv(args.gene_table)
    missing = {"gene", "chr", "start", "end"} - set(df_genes.columns)
    if missing:
        print(f"[Error] Gene table is missing column(s): {sorted(missing)}", file=sys.stderr)
        return 1

    # Harmonise chromosome labels with the SNP side (no 'chr' prefix).
    df_genes = df_genes.dropna(subset=["chr", "start", "end"]).copy()
    df_genes["chr_clean"] = df_genes["chr"].astype(str).str.replace("chr", "", regex=False)
    df_genes["start"] = df_genes["start"].astype(int)
    df_genes["end"] = df_genes["end"].astype(int)

    print("Mapping SNPs to genes...")
    df_final = map_snps_to_genes(df_snp, df_genes)

    if df_final.empty:
        print(
            "[Error] No SNPs mapped to any gene. Check that the SNP positions and the "
            "gene table use the same genome build and chromosome naming.",
            file=sys.stderr,
        )
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(df_final)} mapped SNPs to {args.output}...")
    df_final.to_csv(args.output, index=False)
    print("Done. Point [files].snp_file at this file in your design TOML.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
