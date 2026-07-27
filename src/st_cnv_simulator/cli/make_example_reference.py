#!/usr/bin/env python3
"""Build a small, self-contained example reference set.

Console entry point: ``stcnv-make-example-reference``.

Two of the simulator's reference inputs cannot be redistributed with the
package: the single-cell reference expression matrix (hundreds of megabytes) and
the allele-specific CNV catalogue (derived from controlled-access TCGA data).
This script synthesises stand-ins for both so that a fresh clone can run
end-to-end in a couple of minutes.

Real gene coordinates, chromosome lengths, cell-type labels and SNP positions
are taken from ``data/reference/`` and subsampled. Only the count matrix and the
CNV catalogue are simulated.

The result is fine for smoke tests, tutorials and CI. It is **not** a
substitute for the real reference data in a benchmark: the expression matrix has
no biological covariance structure, and the CNV segments are drawn from a toy
distribution rather than from observed tumours.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Cell types that must survive subsampling for the shipped example design to work.
KEEP_CELLTYPES = (
    "Hepatocyte",
    "Fibroblast",
    "Macrophage",
    "Endothelial",
    "T",
    "B",
    "T_reg",
    "Plasma",
    "NK",
    "Tumor",
)

# Copy-number states for the toy catalogue, as (major, minor) with weights.
CN_STATES = (
    ((1, 0), 0.28),  # LOH / hemizygous deletion
    ((2, 0), 0.20),  # copy-neutral LOH
    ((2, 1), 0.22),  # single-copy gain of one allele
    ((3, 1), 0.15),  # gain
    ((3, 2), 0.08),  # gain
    ((4, 1), 0.04),  # amplification
    ((0, 0), 0.03),  # homozygous deletion
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="stcnv-make-example-reference",
        description="Generate a small runnable example reference set.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("data/reference"),
        help="Directory holding the bundled real annotation files (default: data/reference)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reference/example"),
        help="Where to write the example set (default: data/reference/example)",
    )
    parser.add_argument(
        "--n-genes", type=int, default=1200, help="Number of genes to keep (default: 1200)"
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        default=600,
        help="Number of reference cells to keep (default: 600)",
    )
    parser.add_argument(
        "--n-cnv-segments",
        type=int,
        default=4000,
        help="Number of CNV segments in the toy catalogue (default: 4000)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    return parser.parse_args(argv)


def subsample_genes(gene_order: pd.DataFrame, snps: pd.DataFrame, n_genes: int, rng):
    """Pick n_genes, preferring genes that carry SNPs so BAF output is non-trivial."""
    # A handful of gene symbols occur twice in the real annotation (e.g. GGT1 on
    # two overlapping chr22 intervals). The reference expression matrix must have
    # a unique gene index, so keep the first occurrence of each symbol.
    gene_order = gene_order.drop_duplicates(subset=["gene"], keep="first")

    with_snps = gene_order[gene_order["gene"].isin(set(snps["gene_id"]))]
    n_with = min(len(with_snps), n_genes // 2)
    chosen = list(rng.choice(with_snps["gene"].values, size=n_with, replace=False))

    remaining = gene_order[~gene_order["gene"].isin(set(chosen))]
    n_more = min(len(remaining), n_genes - n_with)
    chosen += list(rng.choice(remaining["gene"].values, size=n_more, replace=False))

    kept = gene_order[gene_order["gene"].isin(set(chosen))].copy()
    return kept.sort_values(["chr", "start"]).reset_index(drop=True)


def synthesise_counts(genes, celltypes: pd.Series, rng) -> pd.DataFrame:
    """Draw a negative-binomial count matrix with per-cell-type mean shifts.

    Genes are rows, cells are columns -- the layout the simulator expects for
    ``[files].refexp_file``.
    """
    n_genes, n_cells = len(genes), len(celltypes)

    # Gene-level baseline: log-normal, so a few genes dominate the library.
    gene_mean = rng.lognormal(mean=1.0, sigma=1.3, size=n_genes)

    # Per-cell-type multiplicative shift, so cell types are distinguishable.
    labels = sorted(celltypes.unique())
    shift = {
        lab: rng.lognormal(mean=0.0, sigma=0.5, size=n_genes) for lab in labels
    }
    shift_matrix = np.column_stack([shift[lab] for lab in celltypes.values])

    # Per-cell library size factor.
    lib_factor = rng.lognormal(mean=0.0, sigma=0.3, size=n_cells)

    mu = gene_mean[:, None] * shift_matrix * lib_factor[None, :]

    # Negative binomial via a gamma-Poisson mixture (dispersion 0.5).
    dispersion = 0.5
    gamma = rng.gamma(shape=1.0 / dispersion, scale=dispersion * mu)
    counts = rng.poisson(gamma).astype(np.int32)

    return pd.DataFrame(counts, index=genes, columns=celltypes.index)


def synthesise_cnv_catalog(chr_lengths: pd.DataFrame, n_segments: int, rng) -> pd.DataFrame:
    """Draw a toy allele-specific CNV catalogue over the given chromosomes."""
    lengths = chr_lengths.set_index("chr")["length"].astype(np.int64)
    # Sample chromosomes in proportion to their length.
    weights = (lengths / lengths.sum()).values
    chroms = rng.choice(lengths.index.values, size=n_segments, p=weights)

    states = [s for s, _ in CN_STATES]
    probs = np.array([w for _, w in CN_STATES], dtype=float)
    probs /= probs.sum()
    state_idx = rng.choice(len(states), size=n_segments, p=probs)

    rows = []
    for chrom, si in zip(chroms, state_idx):
        chrom_len = int(lengths[chrom])
        # Segment lengths spanning focal (~100 kb) to arm-level (~100 Mb).
        seg_len = int(min(chrom_len - 1, 10 ** rng.uniform(5.0, 8.0)))
        start = int(rng.integers(1, max(2, chrom_len - seg_len)))
        major, minor = states[si]
        rows.append(
            {
                "Chromosome": chrom,
                "Start": start,
                "End": start + seg_len,
                "Copy_Number": major + minor,
                "Major_Copy_Number": major,
                "Minor_Copy_Number": minor,
            }
        )

    return pd.DataFrame(rows).sort_values(["Chromosome", "Start"]).reset_index(drop=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    ref = args.reference_dir
    required = ["gene_order.csv", "chr_lengths.csv", "celltype.tsv", "mapped_snps.csv"]
    missing = [f for f in required if not (ref / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing bundled reference file(s) in {ref}: {missing}. "
            "Pass --reference-dir pointing at the repository's data/reference directory."
        )

    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    gene_order = pd.read_csv(ref / "gene_order.csv")
    chr_lengths = pd.read_csv(ref / "chr_lengths.csv")
    snps = pd.read_csv(ref / "mapped_snps.csv")
    celltype = pd.read_csv(
        ref / "celltype.tsv", sep="\t", header=0, index_col=0, names=["celltype"]
    )

    # --- genes -------------------------------------------------------------
    genes = subsample_genes(gene_order, snps, args.n_genes, rng)
    genes.to_csv(out / "gene_order.csv", index=False)
    print(f"gene_order.csv        {len(genes):>7} genes")

    kept_chroms = set(genes["chr"])
    chr_lengths_kept = chr_lengths[chr_lengths["chr"].isin(kept_chroms)]
    chr_lengths_kept.to_csv(out / "chr_lengths.csv", index=False)
    print(f"chr_lengths.csv       {len(chr_lengths_kept):>7} chromosomes")

    # --- cells -------------------------------------------------------------
    celltype = celltype[celltype["celltype"].isin(KEEP_CELLTYPES)]
    # Stratified subsample: keep every cell type, proportional to its abundance,
    # with a floor of 5 cells so no type disappears.
    frac = min(1.0, args.n_cells / len(celltype))
    parts = []
    for _lab, grp in celltype.groupby("celltype"):
        n_keep = max(min(5, len(grp)), int(round(len(grp) * frac)))
        parts.append(grp.iloc[rng.permutation(len(grp))[:n_keep]])
    celltype_kept = pd.concat(parts).sort_index()

    celltype_out = celltype_kept.copy()
    celltype_out.index.name = "cell_type_RNA"
    celltype_out.to_csv(out / "celltype.tsv", sep="\t", header=False)
    print(f"celltype.tsv          {len(celltype_kept):>7} cells across "
          f"{celltype_kept['celltype'].nunique()} types")

    # --- expression --------------------------------------------------------
    counts = synthesise_counts(genes["gene"].values, celltype_kept["celltype"], rng)
    counts.to_csv(out / "counts.tsv", sep="\t")
    print(f"counts.tsv            {counts.shape[0]} genes x {counts.shape[1]} cells (synthetic)")

    # --- SNPs --------------------------------------------------------------
    snps_kept = snps[snps["gene_id"].isin(set(genes["gene"]))]
    snps_kept.to_csv(out / "mapped_snps.csv", index=False)
    print(f"mapped_snps.csv       {len(snps_kept):>7} SNPs")

    # --- CNV catalogue -----------------------------------------------------
    catalog = synthesise_cnv_catalog(chr_lengths_kept, args.n_cnv_segments, rng)
    catalog.to_csv(out / "cnv_catalog.csv", index=False)
    print(f"cnv_catalog.csv       {len(catalog):>7} segments (synthetic, NOT TCGA)")

    (out / "README.md").write_text(
        "# Example reference set (generated)\n\n"
        f"Generated by `stcnv-make-example-reference --seed {args.seed}`.\n\n"
        "`gene_order.csv`, `chr_lengths.csv`, `celltype.tsv` and `mapped_snps.csv` are\n"
        "subsampled from the real annotation in `data/reference/`.\n\n"
        "`counts.tsv` and `cnv_catalog.csv` are **synthetic stand-ins**. They exist so\n"
        "the pipeline runs end-to-end without the large or access-controlled inputs.\n"
        "Do not use them to produce benchmark results -- the count matrix has no\n"
        "biological covariance structure and the CNV segments come from a toy\n"
        "distribution, not from observed tumours. See `data/reference/README.md`.\n",
        encoding="utf-8",
    )

    print(f"\nExample reference written to {out}")
    print("Run it with:\n  stcnv-simulate --design configs/design_example.toml --output out/example")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
