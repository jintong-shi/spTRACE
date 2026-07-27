#!/usr/bin/env python3
"""Run a single spatial-transcriptomics CNV simulation from a design file.

Console entry point: ``stcnv-simulate``.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import _toml as toml  # noqa: E402
from .. import pipeline, utils  # noqa: E402
from ..cnv_utils import clean_all_df  # noqa: E402
from ..qc import analyze_replicate, generate_qc_plots  # noqa: E402

# Keys under [files] that hold filesystem paths. Relative paths in a design
# file are resolved against the directory containing that design file, so a
# checkout can be cloned or moved anywhere without editing the config.
PATH_KEYS = (
    "chr_lengths_file",
    "gene_table_file",
    "refexp_file",
    "refcelltype_file",
    "snp_file",
    "cnv_catalog_file",
)

# Chromosomes carried through the simulation.
VALID_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def synthetic_dir(out_dir_main: str) -> str:
    """Every simulated data file lives under this one directory."""
    return os.path.join(out_dir_main, "synthetic")


def prepare_synthetic_dir(out_dir_main: str, overwrite: bool) -> str:
    """Create <output>/synthetic, refusing to clobber an existing one.

    Called before the simulation rather than after it, for two reasons: the BAF
    matrices are written into this directory while the pipeline runs, and a
    refusal here should cost nothing rather than discard a completed run.
    """
    out_dir = synthetic_dir(out_dir_main)
    if os.path.exists(out_dir):
        if not overwrite:
            raise FileExistsError(
                f"{out_dir} already exists. Pass --overwrite to replace it, "
                "or choose a different --output directory."
            )
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="stcnv-simulate",
        description="Simulate spatial transcriptomics data with allele-specific CNVs.",
    )
    parser.add_argument("--design", required=True, help="Path to the design TOML file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed; overrides [reproducibility].seed in the design file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing <output>/synthetic directory instead of failing",
    )
    parser.add_argument("--debug", action="store_true", help="Dump intermediate objects for QC")
    parser.add_argument(
        "--no-parquet",
        action="store_true",
        help="Write tab-separated text instead of Parquet (avoids the pyarrow dependency)",
    )
    parser.add_argument("--skip-visual", action="store_true", help="Skip all diagnostic figures")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Threads used for expression generation. The default of 1 is both "
        "reproducible and, in our measurements, no slower than 8 -- the worker is "
        "GIL-bound. Values >1 make the counts depend on thread scheduling.",
    )
    parser.add_argument(
        "--quick-purity-only",
        "--quick_purity_only",
        dest="quick_purity_only",
        action="store_true",
        help="Run tissue growth plus normal-cell infusion only, then report tumour purity; "
        "skips expression, ST and BAF generation",
    )
    return parser.parse_args(argv)


def resolve_paths(design: dict, design_file: Path) -> dict:
    """Resolve relative path entries in [files] against the design file's directory."""
    base = design_file.resolve().parent
    files = design.setdefault("files", {})

    # `tcga_all_seg` was the original name for the CNV catalogue. Accept it so
    # older design files keep working, but prefer the source-neutral name.
    if "cnv_catalog_file" not in files and "tcga_all_seg" in files:
        files["cnv_catalog_file"] = files.pop("tcga_all_seg")

    for key in PATH_KEYS:
        raw = files.get(key)
        if not isinstance(raw, str):
            continue
        path = Path(os.path.expandvars(raw)).expanduser()
        if not path.is_absolute():
            path = base / path
        files[key] = str(path)

    missing = [
        f"  [files].{k} = {files[k]}"
        for k in PATH_KEYS
        if k in files and not Path(files[k]).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Reference input(s) named in the design file do not exist:\n"
            + "\n".join(missing)
            + "\n\nSee data/reference/README.md for how to obtain them, or run "
            "`stcnv-make-example-reference` to generate a small runnable example set."
        )
    return design


def normalize_log(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Median-of-ratios library size normalisation, followed by log1p."""
    lib_sizes = expr_df.sum(axis=1)
    size_factors = lib_sizes / np.median(lib_sizes.replace(0, np.nan))
    norm = expr_df.div(size_factors, axis=0)
    return np.log1p(norm)


def normalize_lineage_pairs(lineage_raw):
    """Return a 2-column dataframe (child_id, parent_id) with numeric IDs when possible."""
    if isinstance(lineage_raw, pd.DataFrame):
        if "child_id" in lineage_raw.columns and "parent_id" in lineage_raw.columns:
            lineage_df = lineage_raw[["child_id", "parent_id"]].copy()
        else:
            lineage_df = lineage_raw.iloc[:, :2].copy()
            lineage_df.columns = ["child_id", "parent_id"]
    else:
        lineage_arr = np.asarray(lineage_raw)
        if lineage_arr.ndim != 2 or lineage_arr.shape[1] < 2:
            raise ValueError("lineage data must be a 2-column structure.")
        lineage_df = pd.DataFrame(lineage_arr[:, :2], columns=["child_id", "parent_id"])

    lineage_df["child_id"] = pd.to_numeric(lineage_df["child_id"], errors="coerce").astype("Int64")
    lineage_df["parent_id"] = pd.to_numeric(lineage_df["parent_id"], errors="coerce").astype(
        "Int64"
    )
    lineage_df = lineage_df.dropna(subset=["child_id"]).copy()
    lineage_df = lineage_df.drop_duplicates(subset=["child_id"], keep="first")
    return lineage_df


def get_all_cell_ids(lineage_df: pd.DataFrame, surviving_ids):
    """Union of surviving IDs and every ID appearing in the lineage table."""
    surv = np.asarray(surviving_ids, dtype=int)
    all_ids = set(surv.tolist())
    all_ids.update(lineage_df["child_id"].dropna().astype(int).tolist())
    all_ids.update(lineage_df["parent_id"].dropna().astype(int).tolist())
    all_ids.discard(-1)
    return np.array(sorted(all_ids), dtype=int)


def build_tumor_lineage_df(res, lineage_raw=None):
    """Build the surviving-tumour lineage, keeping parent IDs even when the parent died."""
    cell_ids = np.asarray(res["cell_ids"], dtype=int)
    cell_types = np.asarray(res["tissue_data"]["cell_type"]).astype(str)
    is_tumor = np.char.lower(cell_types) == "tumor"
    tumor_ids = cell_ids[is_tumor]

    if lineage_raw is None:
        lineage_raw = res["lineage"]
    lineage_df = normalize_lineage_pairs(lineage_raw)
    lineage_df = lineage_df[lineage_df["child_id"].isin(tumor_ids)].copy()

    parent_out = []
    for pid in lineage_df["parent_id"]:
        if pd.isna(pid):
            parent_out.append("dropped_cell")
        elif int(pid) == -1:
            parent_out.append("initial_seed")
        else:
            parent_out.append(int(pid))

    lineage_df["parent_id"] = parent_out
    return lineage_df.sort_values("child_id").reset_index(drop=True)


def _safe_write_table(df: pd.DataFrame, parquet_path: str, no_parquet: bool = False, sep="\t"):
    """Write Parquet by default; with ``no_parquet`` write a TSV with the same stem."""
    if no_parquet:
        txt_path = os.path.splitext(parquet_path)[0] + ".tsv"
        df.to_csv(txt_path, sep=sep, index=False)
    else:
        df.to_parquet(parquet_path)


def extract_results(
    res,
    out_dir_main,
    common_gene_order,
    *,
    no_parquet: bool = False,
):
    """Unpack the simulation result object into tabular data files.

    The directory is prepared by :func:`prepare_synthetic_dir` before the
    simulation starts, because the BAF matrices are written into it mid-run.
    """
    out_dir = synthetic_dir(out_dir_main)
    os.makedirs(out_dir, exist_ok=True)
    print("Extracting results, saving under:", out_dir)

    sc_out_dir = os.path.join(out_dir, "single_cell_data")
    os.makedirs(sc_out_dir, exist_ok=True)

    # --- single-cell layer -------------------------------------------------
    sc_data = res["tissue_data"]

    n_cells, _n_genes = sc_data["expression"].shape
    cell_ids = np.asarray(res["cell_ids"], dtype=int)  # e.g. [0, 5, 6, 12, 20, ...]
    if len(cell_ids) != n_cells:
        raise ValueError(f"cell_ids length ({len(cell_ids)}) != n_cells ({n_cells})")
    # Keep the original simulator IDs everywhere instead of row indices.
    cell_names = pd.Index([f"Cell_{cid}" for cid in cell_ids])
    gene_names = common_gene_order

    sc_expr = pd.DataFrame(sc_data["expression"], columns=gene_names, index=cell_names)
    _safe_write_table(
        sc_expr.reset_index(), os.path.join(sc_out_dir, "sc-expression.parquet"), no_parquet
    )

    sc_genome_profile = pd.DataFrame(
        sc_data["genome_profile"], columns=gene_names, index=cell_names
    )
    _safe_write_table(
        sc_genome_profile.reset_index(),
        os.path.join(sc_out_dir, "sc-genome_profile.parquet"),
        no_parquet,
    )

    meta_array = np.hstack(
        (
            sc_data["coordinates"],
            sc_data["cluster_index"].reshape(-1, 1),
            sc_data["cell_status"].reshape(-1, 1),
            sc_data["clone_id"].reshape(-1, 1),
            sc_data["cell_type"].reshape(-1, 1),
        )
    )
    sc_meta_data = pd.DataFrame(
        meta_array,
        columns=["x", "y", "cluster", "cell_status", "clone_id", "cell_type"],
        index=cell_names,
    )
    sc_meta_data.to_csv(os.path.join(sc_out_dir, "sc-meta_data.tsv"), sep="\t")

    # --- spatial (spot) layer ---------------------------------------------
    st_out_dir = os.path.join(out_dir, "spatial_data")
    os.makedirs(st_out_dir, exist_ok=True)
    st_data = res["st_data"]
    spot_names = pd.Index([f"Spot_{x}" for x in range(st_data["expression"].shape[0])])

    st_expr = pd.DataFrame(st_data["expression"], columns=gene_names, index=spot_names)
    st_genome_profile = pd.DataFrame(
        st_data["genome_profile"], columns=gene_names, index=spot_names
    )
    st_meta_data = pd.DataFrame(st_data["coordinates"], columns=["x", "y"], index=spot_names)
    st_meta_data["cell_count"] = st_data["cell_count"]

    if st_data["cluster_index"] is not None:
        cluster_names = pd.Index(
            [f"Cluster_{x}" for x in range(st_data["cluster_index"].shape[1])]
        )
        cluster_data = pd.DataFrame(
            st_data["cluster_index"], columns=cluster_names, index=spot_names
        )
        cluster_data.to_csv(os.path.join(st_out_dir, "cluster_data.tsv"), sep="\t")

    st_annotation = pd.DataFrame(st_data["spot_status"], index=spot_names, columns=["annotation"])
    st_annotation["annotation"] = st_annotation["annotation"].map({1: "benign", 0: "aberrant"})

    for df, name in zip(
        [st_expr.T, st_genome_profile.T, st_meta_data, st_annotation],
        ["st-expression", "st-genome_profile", "st-meta", "st-annotation"],
    ):
        _safe_write_table(
            df.reset_index(), os.path.join(st_out_dir, name + ".parquet"), no_parquet
        )

    with open(os.path.join(st_out_dir, "cell_by_spot.txt"), "w") as fh:
        for k, s in enumerate(st_data["cell_by_spot"]):
            # cell_by_spot stores actual cell IDs, not array indices.
            ids = s if isinstance(s, list) else [s]
            fh.write(f"Spot_{k} : " + ",".join(str(int(x)) for x in ids if x is not None) + "\n")

    # --- lineage / genome tables ------------------------------------------
    cluster_names = pd.Index([f"Cluster_{x}" for x in range(res["cluster_lineage"].shape[0])])
    cluster_lineage = pd.DataFrame(
        res["cluster_lineage"], index=cluster_names, columns=cluster_names
    )

    if isinstance(res["lineage"], pd.DataFrame) and {"child_id", "parent_id"}.issubset(
        res["lineage"].columns
    ):
        lineage_df = res["lineage"][["child_id", "parent_id"]].copy()
    else:
        lineage_df = build_tumor_lineage_df(res)
    lineage_df = lineage_df[lineage_df["child_id"].isin(cell_ids)].copy()
    lineage_df.index = [f"Cell_{int(cid)}" for cid in lineage_df["child_id"].astype(int)]

    # Full ancestry, including cells that died. Only ~1 MB, and without it any walk
    # from a founder to its surviving descendants breaks at the first dead
    # intermediate, understating subclone counts.
    lineage_all_df = None
    if isinstance(res.get("lineage_all"), pd.DataFrame) and {"child_id", "parent_id"}.issubset(
        res["lineage_all"].columns
    ):
        lineage_all_df = res["lineage_all"][["child_id", "parent_id"]].copy()
        lineage_all_df = lineage_all_df.sort_values("child_id").reset_index(drop=True)
        lineage_all_df.index = [
            f"Cell_{int(cid)}" for cid in lineage_all_df["child_id"].astype(int)
        ]

    genome_data = res["genome_data"]
    genome_data[["start", "end"]] = genome_data[["start", "end"]].astype(float).astype(int)
    # genome_data comes back in genomic order (chromosome, then position), while
    # gene_names is alphabetical -- the order of the expression matrix columns.
    # Reindex by gene symbol so the label, the coordinates and the expression
    # column all refer to the same gene; assigning gene_names to .index directly
    # would relabel every row with an unrelated gene.
    if len(genome_data) != len(gene_names):
        raise ValueError(
            f"genome_data has {len(genome_data)} rows but there are {len(gene_names)} "
            "genes in the expression matrix; cannot align the two."
        )
    genome_data = genome_data.set_index("gene", drop=False).loc[gene_names]
    genome_data.index.name = None

    for df, name in zip(
        [cluster_lineage, lineage_df, genome_data], ["cluster_lineage", "lineage", "genome"]
    ):
        df.to_csv(os.path.join(out_dir, name + ".tsv"), sep="\t")

    if lineage_all_df is not None:
        lineage_all_df.to_csv(os.path.join(out_dir, "lineage_all.tsv"), sep="\t")

    if "all_cell_ids" in res:
        all_ids = np.asarray(res["all_cell_ids"], dtype=int)
        survived_set = set(cell_ids.tolist())
        all_cell_df = pd.DataFrame(
            {
                "cell_id": all_ids,
                "cell_name": [f"Cell_{int(x)}" for x in all_ids],
                "is_survived": [int(x) in survived_set for x in all_ids],
            }
        )
        all_cell_df.to_csv(os.path.join(out_dir, "all_cell_ids.tsv"), sep="\t", index=False)


def prefilter_cnv_by_gene_mid(
    cnv_df: pd.DataFrame,
    gene_table: pd.DataFrame,
    chrom_col="Chromosome",
    start_col="Start",
    end_col="End",
    gene_chr_col="chr",
    gene_start_col="start",
    gene_end_col="end",
    min_genes: int = 1,
) -> pd.DataFrame:
    """Drop CNV segments that cover too few gene midpoints.

    Keeps rows with at least ``min_genes`` gene midpoints inside [Start, End].
    Adds two columns: ``row_id`` (original row number) and ``n_genes_overlap``.
    """

    def ensure_chr_prefix(s):
        s = s.astype(str)
        need = ~s.str.startswith("chr", na=False)
        s.loc[need] = "chr" + s.loc[need]
        return s

    df = cnv_df.copy()

    # Record the original row number so segments stay traceable downstream.
    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df))

    # Reset to a 0..N-1 RangeIndex so label-based and positional indexing agree.
    df = df.reset_index(drop=True)

    for c in [start_col, end_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df[chrom_col] = ensure_chr_prefix(df[chrom_col])

    gt = gene_table.copy()
    gt[gene_chr_col] = ensure_chr_prefix(gt[gene_chr_col])
    gt[gene_start_col] = pd.to_numeric(gt[gene_start_col], errors="coerce")
    gt[gene_end_col] = pd.to_numeric(gt[gene_end_col], errors="coerce")
    gt = gt.dropna(subset=[gene_chr_col, gene_start_col, gene_end_col])
    gt["mid"] = ((gt[gene_start_col] + gt[gene_end_col]) // 2).astype(np.int64)

    # Sorted gene midpoints per chromosome, for binary search.
    mid_by_chr = {
        chr_: np.sort(sub["mid"].values.astype(np.int64))
        for chr_, sub in gt.groupby(gene_chr_col)
    }

    n_overlap = np.zeros(len(df), dtype=np.int64)

    valid_mask = df[start_col].notna() & df[end_col].notna()
    # Grouping on the reset RangeIndex means group keys are positions 0..N-1.
    for chr_, idxs in df.loc[valid_mask].groupby(chrom_col).groups.items():
        mids = mid_by_chr.get(chr_)
        if mids is None or mids.size == 0:
            continue

        ss = df.loc[idxs, start_col].astype(np.int64).values
        ee = df.loc[idxs, end_col].astype(np.int64).values
        ss, ee = np.minimum(ss, ee), np.maximum(ss, ee)

        right = np.searchsorted(mids, ee, side="right")
        left = np.searchsorted(mids, ss, side="left")
        n_overlap[np.asarray(idxs)] = right - left

    df["n_genes_overlap"] = n_overlap
    kept = df[df["n_genes_overlap"] >= int(min_genes)].copy()
    print(
        f"[prefilter] kept {len(kept)}/{len(df)} CNV rows "
        f"(dropped {len(df) - len(kept)}) with >= {min_genes} gene mids overlapped."
    )
    if kept.empty:
        raise ValueError(
            f"No CNV segments survived the min_genes={min_genes} filter. "
            "Lower min_genes, or check that the CNV catalogue and gene table use "
            "the same genome build."
        )
    return kept


def _plot_cnv_burden_spatial(res, qc_dir):
    """Per-cell count of genes at a non-diploid copy number, over space."""
    sc = res.get("tissue_data", {})
    if "genome_per_chr" not in sc:
        return
    fig, _ax = utils.visualize_tissue(sc["genome_per_chr"], sc["coordinates"])
    fig.savefig(os.path.join(qc_dir, "cnv_burden_spatial.png"))
    plt.close(fig)


def _plot_cnv_burden_with_spot_array(res, qc_dir):
    """CNV burden over space, with the spot capture array overlaid."""
    sc = res.get("tissue_data", {})
    st = res.get("st_data", {})
    if "genome_per_chr" not in sc or "coordinates" not in st:
        return
    fig, _ax = utils.visualize_array(
        sc["genome_per_chr"],
        sc["coordinates"],
        array_x=st["coordinates"][:, 0],
        array_y=st["coordinates"][:, 1],
    )
    fig.savefig(os.path.join(qc_dir, "cnv_burden_with_spot_array.png"))
    plt.close(fig)


def _plot_sc_celltype_spatial(res, qc_dir):
    """Scatter of single cells coloured by cell type."""
    sc = res.get("tissue_data", {})
    coords = np.asarray(sc.get("coordinates"))
    cell_types = np.asarray(sc.get("cell_type"))
    if coords.size == 0 or cell_types.size == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    types = pd.Categorical(cell_types)
    cmap = mpl.colormaps["tab20"].resampled(max(len(types.categories), 1))
    colors = [cmap(types.categories.get_loc(ct)) for ct in types]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=4, alpha=0.8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Single-cell spatial celltype")
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label=str(ct), markerfacecolor=cmap(i), markersize=6
        )
        for i, ct in enumerate(types.categories)
    ]
    ax.legend(
        handles=handles, bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize=6
    )
    fig.tight_layout()
    fig.savefig(os.path.join(qc_dir, "sc_celltype_spatial.pdf"))
    plt.close(fig)


def _plot_st_celltype_proportions(res, qc_dir):
    """Per-spot cell-type proportion maps, one panel per cell type."""
    st = res.get("st_data", {})
    coords_spot = np.asarray(st.get("coordinates"))
    cell_type_by_spot = st.get("cell_type_by_spot")
    if coords_spot.size == 0 or cell_type_by_spot is None:
        return

    all_types = sorted(
        {ct for lst in cell_type_by_spot for ct in (lst if isinstance(lst, list) else [])}
    )
    if not all_types:
        return

    ncols = min(4, len(all_types))
    nrows = int(np.ceil(len(all_types) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    i = -1
    for i, ct in enumerate(all_types):
        vals = []
        for lst in cell_type_by_spot:
            if not isinstance(lst, list) or len(lst) == 0:
                vals.append(0.0)
            else:
                vals.append(sum(1 for x in lst if x == ct) / len(lst))
        ax = axes[i]
        im = ax.scatter(
            coords_spot[:, 0],
            coords_spot[:, 1],
            c=np.asarray(vals),
            s=18,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Proportion of {ct}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(qc_dir, "st_celltype_proportions.pdf"))
    plt.close(fig)


def load_design(design_file: str | Path) -> dict:
    """Load a design TOML file and resolve its relative input paths."""
    design_file = Path(design_file)
    return resolve_paths(toml.load(design_file), design_file)


def build_reference_inputs(ds: dict):
    """Load and harmonise the reference tables named in the design file.

    Returns ``(chr_lengths, gene_table, exp, df_mapped_snps, common_gene_order)``
    sharing one sorted gene order.
    """
    files = ds["files"]
    chr_lengths = pd.read_csv(files["chr_lengths_file"])
    gene_table = pd.read_csv(files["gene_table_file"])
    exp = pd.read_csv(files["refexp_file"], sep="\t", header=0, index_col=0)

    df_mapped_snps = pd.read_csv(files["snp_file"])
    if "chr" in df_mapped_snps.columns and "chrom" not in df_mapped_snps.columns:
        df_mapped_snps = df_mapped_snps.rename(columns={"chr": "chrom"})
    if "chrom" in df_mapped_snps.columns:
        df_mapped_snps["chrom"] = (
            df_mapped_snps["chrom"].astype(str).str.replace("chr", "", regex=False)
        )

    # Restrict to supported chromosomes and to genes present in both the
    # annotation and the reference expression matrix.
    gene_table = gene_table[gene_table["chr"].isin(VALID_CHROMS)]
    chr_lengths = chr_lengths[chr_lengths["chr"].isin(VALID_CHROMS)]

    gene_intersection = np.intersect1d(exp.index, gene_table["gene"].values)
    if gene_intersection.size == 0:
        raise ValueError(
            "No genes shared between the reference expression matrix and the gene table. "
            "Check that both use the same gene identifiers."
        )
    common_gene_order = np.sort(gene_intersection)

    exp = exp.loc[common_gene_order]
    gene_table = gene_table[gene_table["gene"].isin(gene_intersection)].drop_duplicates(
        subset=["gene"]
    )
    gene_table = gene_table.set_index("gene").loc[common_gene_order].reset_index()

    return chr_lengths, gene_table, exp, df_mapped_snps, common_gene_order


def build_sim_args(
    ds: dict,
    gene_table,
    exp,
    common_gene_order,
    chr_lengths,
    df_mapped_snps,
    out_dir_main: str,
    debug: bool,
    n_jobs: int = 1,
) -> dict:
    """Translate a design dict into the keyword arguments of :func:`pipeline.run`."""
    files = ds["files"]

    df_celltype = pd.read_csv(
        files["refcelltype_file"], sep="\t", header=0, index_col=0, names=["celltype"]
    )
    exp = exp.loc[exp.index.isin(gene_table["gene"])]

    # Normal (TME) and tumour baseline expression profiles.
    normal_TME_exp = utils.generate_normal_expression(
        exp, df_celltype, files["normal_celltypes"], min_counts=1e3
    )
    normal_original_expr = normal_TME_exp[files["normal_original_celltype"]].copy()
    tumor_baseline_exp = utils.generate_tumor_baseline_expressions(
        normal_original_expr,
        n_initial_cells=ds["population"]["n_initial_cells"],
        percent_de=0.3,
        logfc_range=(-1, 1),
    )

    # Cell-state transition matrix.
    if "transitions" in ds["population"]:
        tmat = np.zeros((2, 2))
        tmat[1, 0] = ds["population"]["transitions"]["s0s1"]
        tmat[0, 1] = ds["population"]["transitions"]["s1s0"]
        tmat[0, 0] = 1 - tmat[1, 0]
        tmat[1, 1] = 1 - tmat[0, 1]
    else:
        tmat = None

    # Per-state action rates and mutation rates.
    if "rates_state_0" in ds["population"] and "rates_state_1" in ds["population"]:
        rates_0 = ds["population"]["rates_state_0"]
        rates_1 = ds["population"]["rates_state_1"]
        action_rates = np.zeros((4, 2))
        for col, rates in ((0, rates_0), (1, rates_1)):
            action_rates[:, col] = np.array(
                (
                    rates["stay_rate"],
                    rates["spawn_rate"],
                    rates["move_rate"],
                    rates["death_rate"],
                )
            )
        mutation_rate = np.array((rates_0["mutation_rate"], rates_1["mutation_rate"]))
    else:
        action_rates = None
        mutation_rate = None

    if "initial_coordinates" in ds["population"]:
        init_coordinates = [(c["x"], c["y"]) for c in ds["population"]["initial_coordinates"]]
    else:
        init_coordinates = None

    # CNV event catalogue: clean, then keep only gene-dense segments.
    all_df_clean = clean_all_df(
        pd.read_csv(files["cnv_catalog_file"]),
        drop_normal=True,  # drop balanced Major=2 & Minor=2 segments
        min_len=None,  # e.g. 1e5 to enforce a minimum segment length
        standardize_chr=True,
    )
    cnv_events_df_filt = prefilter_cnv_by_gene_mid(all_df_clean, gene_table, min_genes=5)

    normal_action_rates = ds["population"].get("normal_action_rates", None)

    return dict(
        n_initial_cells=ds["population"]["n_initial_cells"],
        n_clusters=ds["population"].get("n_clusters", 10),
        max_iter=ds["population"].get("max_iterations", 500),
        max_cells=ds["population"].get("max_cells", 1e4),
        domain_side_size=ds["population"].get("domain_side_size", 200),
        lognorm_params_tumor=ds["expression"].get("lognorm_params_tumor", None),
        lognorm_params_normal=ds["expression"].get("lognorm_params_normal", None),
        dirichlet_precision=ds["expression"].get("dirichlet_precision", None),
        init_event_size=ds["population"].get("init_event_size", None),
        fraction_drop=ds["population"].get("fraction_drop", 0.05),
        spot_radius=ds["spatial"].get("spot_radius", 2.5),
        n_spots_x=ds["spatial"].get("n_spots_x", 10),
        n_spots_y=ds["spatial"].get("n_spots_y", 10),
        st_eff=ds["spatial"].get("st_eff", 0.2),
        st_theta=ds["spatial"].get("st_theta", 10),
        action_rates=action_rates,
        mutation_rate=mutation_rate,
        init_coordinates=init_coordinates,
        transitions=tmat,
        chr_lengths=chr_lengths,
        gene_table=gene_table,
        common_gene_order=common_gene_order,
        tumor_baseline_exp=tumor_baseline_exp,
        normal_TME_exp=normal_TME_exp,
        normal_celltype_proportions=ds["normal_celltype_proportions"],
        fill_prob=ds["population"].get("background_fill_prob", 0.75),
        normal_action_rates=tuple(normal_action_rates) if normal_action_rates else None,
        output_dir=out_dir_main,
        debug=debug,
        cnv_events_df=cnv_events_df_filt,
        n_init_shared=ds["initial_cnv"].get("n_shared", 1),
        n_init_private=ds["initial_cnv"].get("n_private", 2),
        config=ds,  # full design dict, for the optional feature sections
        df_mapped_snps=df_mapped_snps,
        n_jobs=n_jobs,
    )


def main(argv=None) -> int:
    print("Starting ST CNV simulation...")
    args = parse_args(argv)
    out_dir_main = args.output

    os.makedirs(out_dir_main, exist_ok=True)
    print("Output directory:", out_dir_main)

    ds = load_design(args.design)
    print("Design file loaded:", args.design)

    # Prepared up front: the pipeline writes BAF matrices straight into it, and a
    # refusal to overwrite must not come after an expensive simulation.
    prepare_synthetic_dir(out_dir_main, args.overwrite)

    # Global random seed. An explicit --seed wins over the design file, and 0 is
    # a valid seed. Without either, the run is not reproducible.
    seed = args.seed if args.seed is not None else ds.get("reproducibility", {}).get("seed")
    if seed is None:
        print(
            "WARNING: no seed set ([reproducibility].seed or --seed); this run is not reproducible."
        )
    else:
        seed = int(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"Reproducibility seed set to {seed}")

    chr_lengths, gene_table, exp, df_mapped_snps, common_gene_order = build_reference_inputs(ds)

    dargs = build_sim_args(
        ds,
        gene_table,
        exp,
        common_gene_order,
        chr_lengths,
        df_mapped_snps,
        out_dir_main,
        args.debug,
        n_jobs=args.jobs,
    )

    if args.quick_purity_only:
        dargs["generate_expression"] = False
        dargs["generate_st"] = False
        dargs["generate_baf"] = False
        dargs["skip_clustering"] = True

    # Drop unset parameters so pipeline.run() applies its own defaults.
    dargs = {k: v for k, v in dargs.items() if v is not None}

    print("Start running simulation...")
    res = pipeline.run(**dargs)

    lineage_all_df = normalize_lineage_pairs(res["lineage"])
    res["lineage_all"] = lineage_all_df
    res["all_cell_ids"] = get_all_cell_ids(lineage_all_df, res["cell_ids"])
    res["lineage"] = build_tumor_lineage_df(res, lineage_raw=lineage_all_df)

    if isinstance(res.get("event_log"), pd.DataFrame):
        res["event_log"].to_csv(
            os.path.join(synthetic_dir(out_dir_main), "event_log.tsv"),
            sep="\t",
            index=False,
        )

    if args.quick_purity_only:
        cell_types = np.asarray(res.get("tissue_data", {}).get("cell_type", []))
        n_cells = int(cell_types.shape[0])
        n_tumor = int(np.sum(cell_types == "Tumor")) if n_cells > 0 else 0
        purity = (n_tumor / n_cells) if n_cells > 0 else np.nan
        quick_df = pd.DataFrame(
            {
                "n_cells": [n_cells],
                "n_tumor_cells": [n_tumor],
                "overall_tumor_purity": [purity],
            }
        )
        quick_out = os.path.join(out_dir_main, "quick_purity_summary.tsv")
        quick_df.to_csv(quick_out, sep="\t", index=False)
        print(f"[Quick] overall_tumor_purity = {purity:.4f} ({n_tumor}/{n_cells})")
        print("Quick mode complete. Saved:", quick_out)
        return 0

    print("Extracting simulation results...")
    extract_results(
        res,
        out_dir_main,
        common_gene_order,
        no_parquet=args.no_parquet,
    )
    # The expression matrix is generated through a memmap under <output>/tmp.
    # By this point every consumer has been written out, so drop the backing file
    # rather than leave gigabytes of scratch behind (16 GB+ on a full-size run).
    tmp_dir = os.path.join(out_dir_main, "tmp")
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("Simulation complete. Data saved under:", os.path.join(out_dir_main, "synthetic"))

    if not args.skip_visual:
        qc_dir = os.path.join(out_dir_main, "qc")
        os.makedirs(qc_dir, exist_ok=True)
        for label, fn in (
            ("CNV burden spatial plot", _plot_cnv_burden_spatial),
            ("CNV burden + spot array plot", _plot_cnv_burden_with_spot_array),
            ("sc celltype spatial plot", _plot_sc_celltype_spatial),
            ("ST celltype proportion plots", _plot_st_celltype_proportions),
        ):
            try:
                fn(res, qc_dir)
            except Exception as exc:  # QC figures must never fail a completed run
                print(f"WARN: failed to draw {label}: {exc}")

        # Subclone QC: founder clones, evolved subclones, per-cell membership and the
        # spot-level views. This reads the tables just written, so it is the same
        # analysis stcnv-batch runs per replicate.
        try:
            qc_res, stats = analyze_replicate(out_dir_main)
            if "error" in stats:
                print(f"WARN: subclone QC skipped: {stats['error']}")
            else:
                if not stats.get("subclone_counts_exact", True):
                    print(
                        f"NOTE: {stats['n_broken_lineage_links']} ancestry links point at "
                        f"cells absent from lineage.tsv; n_subclones={stats['n_subclones']} "
                        "is a lower bound."
                    )
                generate_qc_plots(
                    qc_res, stats, out_dir_main, os.path.basename(os.path.normpath(out_dir_main))
                )
                print(
                    f"Subclone QC: {stats['n_subclones']} subclones "
                    f"({stats['n_base_clones']} founder + {stats['n_major_founders']} evolved), "
                    f"tumour purity {stats['purity']:.3f}"
                )
        except Exception as exc:
            print(f"WARN: subclone QC failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
