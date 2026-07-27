from typing import Tuple
from typing import Dict
import matplotlib.pyplot as plt
from . import model as m
import numpy as np
import pandas as pd
import random
import copy
import os
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist, squareform
import matplotlib as mpl

class Counter:
    def __init__(self, start: int = 0):
        self.cnt = start - 1

    def __call__(self):
        self.cnt += 1
        return self.cnt

def generate_normal_expression(exp, df_celltype, normal_celltypes, min_counts=1e5):
    """
    Get the mean expression for each normal cell type from a real data.
    
    Attributes
    ----------
    refexp_file : str
        Path to the reference expression file. The file should be a tab-separated table with genes as rows and cells as columns.

    refcelltype_file : str
        Path to the reference cell type file. The file should be a tab-separated table with cells as rownames and cell type as a column.

    normal_celltypes : list
        A list of cell types appearing in refcelltype_file to consider as normal cells.

    Returns
    -------
    exp : pd.DataFrame
        A table with rows as genes and columns as cell types.
    """
    exp = exp.T
    # join the expression and cell type table
    exp = exp.join(df_celltype, how='left')
    # aggregate the expression by cell type
    exp = exp.groupby('celltype').sum()
    exp = exp[exp.index.isin(normal_celltypes)]
    exp = exp[exp.sum(axis=1) > min_counts]
    # add pseudocount
    exp = exp + 0.5
    # normalize row sum to 1
    exp = exp.div(exp.sum(axis=1), axis=0)
    return exp.T

def generate_tumor_baseline_expressions(
    normal_expr: pd.Series,
    n_initial_cells: int = 2,
    percent_de: float = 0.3,
    logfc_range: tuple = (-1, 1),
    random_state: int = 42,
    additional_percent_de_range: tuple = (0.001, 0.02),  # extra fraction of clone-private DE genes (drawn at random)
) -> dict:
    """
    Generate a baseline tumour expression profile for each of n_initial_cells founder cells.
    
    Parameters
    ----------
    normal_expr : pd.Series
        Mean expression of each gene in normal epithelial cells; index is the gene name, value the level.
        For example: gene1 -> 10, gene2 -> 100, ...
    
    n_initial_cells : int
        How many founder tumour cells (or tumour blocks) to generate baselines for.
    
    percent_de : float
        0.3 means 30% of genes are differentially expressed in that baseline (random logFC).
    
    logfc_range : tuple
        Range of logFC, default (-2, 2). The fold change applied is exp(logFC).
        -2..2 corresponds to roughly a 0.135x..7.389x change in expression.
    
    random_state : int
        Random seed, for reproducible output.
    
    Returns
    -------
    baseline_dict : dict
        A dict of the form:
        {
          "tumor_expr_1": pd.Series(...),
          "tumor_expr_2": pd.Series(...),
          ...
        }
        Every pd.Series shares its gene index with normal_expr and holds the
        baseline tumour expression profile of that founder cell.
    
    Example
    -------
    >>> normal_expr = pd.Series({"geneA": 50, "geneB": 120, "geneC": 200})
    >>> baseline_exprs = generate_tumor_baseline_expressions(normal_expr, n_initial_cells=2)
    >>> baseline_exprs.keys()
    dict_keys(['tumor_expr_1', 'tumor_expr_2'])
    >>> baseline_exprs["tumor_expr_1"].head()
    geneA    27.03
    geneB    350.42
    geneC    200.00
    dtype: float64  (example)
    """
    rng = np.random.default_rng(random_state)
    genes = normal_expr.index
    n_genes = len(genes)
    n_shared = int(percent_de * n_genes)
    idx_all = rng.permutation(n_genes)
    idx_shared = idx_all[:n_shared]

    baseline = {}
    non_shared_pool = np.setdiff1d(np.arange(n_genes), idx_shared)

    for i in range(n_initial_cells):
        tumor_expr = normal_expr.copy()

        # Apply the fold change to the shared DE genes. Each clone draws its own magnitudes; move this out of the loop to give every clone identical folds.
        logfc_shared = rng.uniform(logfc_range[0], logfc_range[1], size=n_shared)
        fold_shared = np.exp(logfc_shared)
        tumor_expr.iloc[idx_shared] = tumor_expr.iloc[idx_shared].values * fold_shared

        # Then give each clone a little private DE (e.g. 1%)
        priv_frac = float(rng.uniform(*additional_percent_de_range))
        n_private = max(1, int(round(priv_frac * n_genes)))
        n_private = min(n_private, len(non_shared_pool))  # guard against overrunning the pool

        n_private = min(n_private, len(non_shared_pool))  # guard against overrunning the pool
        if n_private > 0:
            idx_private = rng.choice(non_shared_pool, size=n_private, replace=False)
            logfc_priv = rng.uniform(logfc_range[0], logfc_range[1], size=n_private)
            fold_priv = np.exp(logfc_priv)
            tumor_expr.iloc[idx_private] = tumor_expr.iloc[idx_private].values * fold_priv

        baseline[f"tumor_expr_{i}"] = tumor_expr
    return baseline

def copy2fold_func(copy_counts: np.ndarray, baseline_copy: int = 2) -> np.ndarray:
    """
    Convert each gene's copy number (copy_counts[g]) into a fold factor.
    
    fold_factor[g] = copy_counts[g] / baseline_copy
    For example:
      with baseline_copy=2,
        copy=0 => 0/2=0.0
        copy=1 => 1/2=0.5
        copy=2 => 2/2=1.0  (no change)
        copy=3 => 3/2=1.5
        copy=4 => 4/2=2.0
        ...
    
    Parameters
    ----------
    copy_counts : np.ndarray
        Shape (G,): the copy number of each gene.
    baseline_copy : int
        Normal gene copy number, e.g. 2 for a human diploid genome.
    
    Returns
    -------
    fold_factors : np.ndarray
        A float array of shape (G,); each element is that gene's expression multiplier relative to baseline.
    """
    fold_factors = copy_counts / baseline_copy
    return fold_factors

def clean_ax(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for sp in ax.spines.values():
        sp.set_visible(False)
def visualize_tissue(
    sc_genome_dict: Dict[str, np.ndarray],
    sc_crds: np.ndarray,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    For each cell, sum over chromosomes the number of genes whose copy count differs from 1,
    giving a per-cell count of aberrant genes,
    and scatter-plot that aberration level over space.

    Parameters
      sc_genome_dict: keyed by chromosome name, values are 2-D arrays (n_cells x n_genes_on_chrom)
      sc_crds: an (n_cells, 2) array of per-cell spatial coordinates
    """
    # Accumulate the aberrant gene count over chromosomes
    abber = np.array([sum(np.sum(arr != 1) for arr in cell_dict.values()) for cell_dict in sc_genome_dict])

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.scatter(
        sc_crds[:, 0],
        sc_crds[:, 1],
        c=abber,
        cmap=plt.cm.plasma,
        s=2,
    )

    ax.set_aspect("equal")

    clean_ax(ax)

    return fig, ax


def visualize_array(
    sc_genome: np.ndarray,
    sc_crd: np.ndarray,
    array_x: np.ndarray,
    array_y: np.ndarray,
) -> Tuple[plt.Figure, plt.Axes]:

    fig, ax = visualize_tissue(
        sc_genome,
        sc_crd,
    )

    ax.scatter(
        array_x,
        array_y,
        s=30,
        c="red",
        alpha=0.3,
    )

    return fig, ax

def get_sc_gene_count_df(sc_genome_per_chr, cell_names, gene_names_dict):
    """
    SPEEDUP (claude/01): drop-in replacement for the dict-of-dicts version.
    Equivalence-tested (pd.testing.assert_frame_equal + np.testing.assert_array_equal,
    see claude/01_get_sc_gene_count_df/test_equivalence.py).

    Builds a single (n_cells, n_cols) numpy buffer instead of a list of n_cells
    dicts each carrying n_chroms*n_genes Python-object entries.
    """
    if not sc_genome_per_chr:
        return pd.DataFrame(index=cell_names)

    first = sc_genome_per_chr[0]

    columns = []
    col_idx_map = {}
    chrom_targets = {}

    for chrom in first.keys():
        copy = chrom[-1]
        gene_list = gene_names_dict[chrom]
        targets = np.empty(len(gene_list), dtype=np.int64)
        for j, gene in enumerate(gene_list):
            col_name = f"{gene}_{copy}"
            if col_name in col_idx_map:
                targets[j] = col_idx_map[col_name]
            else:
                col_idx_map[col_name] = len(columns)
                targets[j] = len(columns)
                columns.append(col_name)
        chrom_targets[chrom] = targets

    sample = next(iter(first.values()))
    dtype = sample.dtype if hasattr(sample, "dtype") else np.int64

    n_cells = len(sc_genome_per_chr)
    n_cols = len(columns)
    arr = np.empty((n_cells, n_cols), dtype=dtype)

    for i, cell_dict in enumerate(sc_genome_per_chr):
        for chrom, counts in cell_dict.items():
            arr[i, chrom_targets[chrom]] = np.asarray(counts)

    return pd.DataFrame(arr, index=cell_names, columns=columns)
# ---------- gene / chrom helpers ----------
def normalize_chr_name(c):
    c = str(c)
    if not c.lower().startswith("chr"):
        return "chr" + c
    return c

def get_genes_sorted_by_chrom(
    gene_list: list[str],
    annotation_df: pd.DataFrame,
    chrom_order: list[str] | None = None,
) -> list[str]:
    """
    Sort gene_list by chromosome order then position. annotation_df must have columns: gene, chr, start, end.
    """
    df = annotation_df.copy()
    df["chr"] = df["chr"].astype(str).map(normalize_chr_name)
    df["mid"] = (df["start"].astype(float) + df["end"].astype(float)) / 2.0
    df = df[df["gene"].isin(gene_list)].copy()

    if chrom_order is None:
        chrom_order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

    chrom_rank = {c: i for i, c in enumerate(chrom_order)}
    df["chrom_rank"] = df["chr"].map(chrom_rank).fillna(len(chrom_order))

    df.sort_values(["chrom_rank", "mid"], inplace=True)
    sorted_genes = df["gene"].tolist()

    missing = [g for g in gene_list if g not in sorted_genes]
    return sorted_genes + missing  # genes missing from the annotation are kept at the end

def compute_chrom_boundaries(sorted_genes: list[str], annotation_df: pd.DataFrame):
    """
    Given a gene list already ordered by chromosome, return each gene's chr label and the indices where the chromosome changes.
    """
    df = annotation_df.copy()
    df["chr"] = df["chr"].astype(str).map(normalize_chr_name)
    df = df[df["gene"].isin(sorted_genes)].copy()
    gene2chr = df.set_index("gene")["chr"].to_dict()
    gene_chrs = [gene2chr.get(g, "NA") for g in sorted_genes]

    boundaries = []
    prev = gene_chrs[0]
    for i, c in enumerate(gene_chrs[1:], start=1):
        if c != prev:
            boundaries.append(i)  # boundary before gene i
            prev = c
    return gene_chrs, boundaries

# ---------- plotting without hierarchical clustering ----------
def plot_heatmap_by_cluster_and_chrom(
    matrix: np.ndarray,                # rows = cells/spots, cols = genes
    row_names: list[str],
    col_names: list[str],
    cluster_labels: np.ndarray,        # integer label per row
    annotation_df: pd.DataFrame,       # for gene chrom ordering, must have gene, chr, start, end
    out_file: str,
    top_n_genes: int | None = None,    # optionally subset genes before ordering by chromosome
    baseline_copy_number: float | None = None,  # set to 2 for a genome heatmap, to plot log2(CN/2)
    zscore_rows: bool = False,
    title: str = "Heatmap",
    figsize: tuple = (12, 6),
):
    """
    No hierarchical clustering: rows are grouped by cluster_labels, columns ordered by chromosome position, with cluster/chr colour bars and separators.
    """
    mat = matrix.astype(float).copy()
    eps = 1e-6

    # baseline transform
    if baseline_copy_number is not None:
        mat = np.log2((mat + eps) / baseline_copy_number)

    # gene pre-selection (top variance) then chrom sort
    genes = col_names
    if top_n_genes is not None and len(genes) > top_n_genes:
        vars_ = mat.var(axis=0)
        top_idx = np.argsort(-vars_)[:top_n_genes]
        mat = mat[:, top_idx]
        genes = [genes[i] for i in top_idx]

    # now sort genes by chrom position
    sorted_genes = get_genes_sorted_by_chrom(genes, annotation_df)
    gene_idx_order = [genes.index(g) for g in sorted_genes if g in genes]
    mat = mat[:, gene_idx_order]
    ordered_col_names = sorted_genes

    # row ordering: group by cluster_labels (preserve cluster order as seen)
    unique_clusters = np.unique(cluster_labels)
    row_order = []
    for cl in unique_clusters:
        idxs = np.where(cluster_labels == cl)[0].tolist()
        row_order.extend(idxs)
    ordered = mat[np.ix_(row_order, np.arange(mat.shape[1]))]
    ordered_row_names = [row_names[i] for i in row_order]
    ordered_row_clusters = cluster_labels[row_order]

    # compute chromosome boundaries for column (gene) side
    gene_chrs, chr_boundaries = compute_chrom_boundaries(ordered_col_names, annotation_df)

    # plotting
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(ordered.T, aspect="auto", interpolation="nearest")
    ax.set_title(title)

    # ticks
    if len(ordered_row_names) <= 60:
        ax.set_xticks(np.arange(len(ordered_row_names)))
        ax.set_xticklabels(ordered_row_names, rotation=90, fontsize=6)
    else:
        ax.set_xticks([])
    if len(ordered_col_names) <= 100:
        ax.set_yticks(np.arange(len(ordered_col_names)))
        ax.set_yticklabels(ordered_col_names, fontsize=6)
    else:
        ax.set_yticks([])

    # cluster stripe (rows) on top
    cmap = mpl.colormaps["tab10"]
    norm = mpl.colors.Normalize(vmin=unique_clusters.min(), vmax=unique_clusters.max())
    scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    row_colors = [scalar_map.to_rgba(cl) for cl in ordered_row_clusters]
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    # row cluster bar above
    cax_cluster = divider.append_axes("top", size="5%", pad=0.1)
    cax_cluster.imshow([row_colors], aspect="auto")
    cax_cluster.set_xticks([])
    cax_cluster.set_yticks([])
    cax_cluster.set_ylabel("Cluster", rotation=0, labelpad=30, va="center", fontsize=8)

    # chromosome stripe (genes) on left
    chr_cmap = mpl.colormaps["tab20"]
    unique_chrs = [c for c in sorted(set(gene_chrs)) if c != "NA"]
    chr_order = unique_chrs  # override to customise the order
    chr_norm = mpl.colors.Normalize(vmin=0, vmax=max(len(chr_order) - 1, 1))
    chr2idx = {c: i for i, c in enumerate(chr_order)}
    gene_chr_colors = [chr_cmap(chr2idx.get(c, len(chr_order) - 1)) for c in gene_chrs]
    cax_chr = divider.append_axes("left", size="3%", pad=0.1)
    cax_chr.imshow([[gene_chr_colors[i] for i in range(len(gene_chr_colors))]], aspect="auto")
    cax_chr.set_xticks([])
    cax_chr.set_yticks([])
    cax_chr.set_xlabel("Chrom", labelpad=20, fontsize=8)

    # draw chromosome boundary lines on heatmap (horizontal because genes on y after transpose)
    for b in chr_boundaries:
        ax.hlines(b - 0.5, xmin=0, xmax=len(ordered_row_names) - 1, colors="white", linewidth=1)

    # legends
    handles_cluster = [mpl.patches.Patch(color=scalar_map.to_rgba(cl), label=f"Cluster_{cl}") for cl in unique_clusters]
    ax.legend(
        handles=handles_cluster,
        bbox_to_anchor=(1.02, 1.02),
        loc="upper left",
        title="Row clusters",
        fontsize=6,
        title_fontsize=7,
        frameon=False,
    )
    handles_chr = [mpl.patches.Patch(color=chr_cmap(i), label=chr_order[i]) for i in range(len(chr_order))]
    ax.legend(
        handles=handles_cluster + handles_chr,
        bbox_to_anchor=(1.02, 0.5),
        loc="upper left",
        fontsize=6,
        frameon=False,
    )

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
    ylabel = "log2 fold change" if baseline_copy_number is not None else "value"
    cbar.set_label(ylabel)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)
