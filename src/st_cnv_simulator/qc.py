"""Quality-control analysis and figures for a completed simulation.

Everything here works from a run's output tables, so it applies equally to a single
``stcnv-simulate`` run and to each replicate of ``stcnv-batch``.

:func:`analyze_replicate` derives the subclone structure (founder clones, evolved
subclones and their per-cell membership) and :func:`generate_qc_plots` renders it.

The caller is responsible for selecting a non-interactive matplotlib backend; both
console entry points set ``Agg`` before importing this module.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Cell-type colour palette  (matches R scripts)
# ---------------------------------------------------------------------------
CT_COLORS = {
    "Tumor": "#d73027",
    "Hepatocyte": "#1f78b4",
    "Fibroblast": "#33a02c",
    "Macrophage": "#ff7f00",
    "Endothelial": "#6a3d9a",
    "T": "#e31a1c",
    "T_reg": "#b15928",
    "B": "#a6cee3",
    "NK": "#fb9a99",
    "Plasma": "#cab2d6",
}

MAJOR_SUBCLONE_MIN_CELL_FRAC = 0.05


def _read_table(stem, **kwargs):
    """Read a simulator output written either as Parquet or, with --no-parquet, TSV."""
    parquet, tsv = stem + ".parquet", stem + ".tsv"
    if os.path.exists(parquet):
        return pd.read_parquet(parquet)
    if os.path.exists(tsv):
        return pd.read_csv(tsv, sep="\t", **kwargs)
    raise FileNotFoundError(f"neither {parquet} nor {tsv} exists")


def _cell_id_from_name(names):
    """'Cell_1234' -> 1234."""
    return pd.Series(names).astype(str).str.replace("Cell_", "", regex=False).astype(int).values


def load_replicate_tables(rep_folder):
    """Reconstruct the fields the QC code needs from the replicate's output tables.

    The simulator no longer writes res.pkl (it reached ~10 GB per replicate), so
    everything here comes from the much smaller TSV/Parquet outputs.
    """
    syn = os.path.join(rep_folder, "synthetic")

    meta = pd.read_csv(
        os.path.join(syn, "single_cell_data", "sc-meta_data.tsv"), sep="\t", index_col=0
    )
    cell_ids = _cell_id_from_name(meta.index)

    clone_id = pd.to_numeric(meta["clone_id"], errors="coerce")
    lineage = pd.read_csv(os.path.join(syn, "lineage.tsv"), sep="\t", index_col=0)

    # Prefer the full ancestry: lineage.tsv holds surviving tumour cells only, so
    # founder -> descendant walks over it stop at the first cell that died.
    lineage_all = None
    lineage_all_path = os.path.join(syn, "lineage_all.tsv")
    if os.path.exists(lineage_all_path):
        lineage_all = pd.read_csv(lineage_all_path, sep="\t", index_col=0)

    st_data = {}
    st_meta_stem = os.path.join(syn, "spatial_data", "st-meta")
    cell_by_spot_path = os.path.join(syn, "spatial_data", "cell_by_spot.txt")
    if os.path.exists(st_meta_stem + ".parquet") or os.path.exists(st_meta_stem + ".tsv"):
        st_meta = _read_table(st_meta_stem)
        st_data["coordinates"] = st_meta[["x", "y"]].to_numpy(dtype=float)
    if os.path.exists(cell_by_spot_path):
        by_spot = []
        with open(cell_by_spot_path) as fh:
            for line in fh:
                _, _, ids = line.partition(":")
                ids = ids.strip()
                by_spot.append([int(x) for x in ids.split(",") if x] if ids else [])
        st_data["cell_by_spot"] = by_spot

    return {
        "tissue_data": {
            "cell_type": meta["cell_type"].astype(str).to_numpy(),
            "clone_id": clone_id.to_numpy(dtype=object),
            "coordinates": meta[["x", "y"]].to_numpy(dtype=float),
        },
        "cell_ids": cell_ids,
        "lineage": lineage,
        "lineage_all": lineage_all,
        "st_data": st_data,
    }


def analyze_replicate(rep_folder, min_frac=MAJOR_SUBCLONE_MIN_CELL_FRAC):
    """Count subclones and compute per-cell labels for the QC plots."""
    event_log_path = os.path.join(rep_folder, "synthetic", "event_log.tsv")

    meta_path = os.path.join(
        rep_folder, "synthetic", "single_cell_data", "sc-meta_data.tsv"
    )
    if not os.path.exists(meta_path):
        return None, {"error": f"sc-meta_data.tsv not found under {rep_folder}"}

    res = load_replicate_tables(rep_folder)

    cell_types = np.asarray(res.get("tissue_data", {}).get("cell_type", []))
    n_cells = int(cell_types.shape[0])
    n_tumor = int(np.sum(np.char.lower(cell_types.astype(str)) == "tumor")) if n_cells > 0 else 0
    purity = n_tumor / n_cells if n_cells > 0 else float("nan")

    cell_ids = np.asarray(res.get("cell_ids", []), dtype=int)
    clone_ids_raw = res.get("tissue_data", {}).get("clone_id", np.array([]))
    clone_ids = np.array(
        [-1 if (x is None or pd.isna(x)) else int(x) for x in clone_ids_raw]
    )

    tumor_mask = clone_ids >= 0
    tumor_cell_ids = set(cell_ids[tumor_mask].tolist())
    n_tumor_cells = len(tumor_cell_ids)

    # lineage_all.tsv carries every cell ever created. Falling back to lineage.tsv
    # (surviving tumour cells only) breaks any chain that passes through a dead cell,
    # which turns the descendant counts below into a LOWER BOUND. n_broken_links
    # quantifies the gap and is 0 when the full ancestry is available.
    lineage_all = res.get("lineage_all")
    if lineage_all is None or not isinstance(lineage_all, pd.DataFrame):
        lineage_all = res.get("lineage")

    parent_map = {}
    if isinstance(lineage_all, pd.DataFrame) and {"child_id", "parent_id"}.issubset(
        lineage_all.columns
    ):
        for _, row in lineage_all.iterrows():
            try:
                parent_map[int(row["child_id"])] = int(row["parent_id"])
            except (TypeError, ValueError):
                # 'initial_seed' / 'dropped_cell' -- no numeric parent to record
                continue

    known_cells = set(parent_map)
    n_broken_links = sum(1 for p in parent_map.values() if p >= 0 and p not in known_cells)

    children_map = {}
    for child, parent in parent_map.items():
        if parent >= 0:
            children_map.setdefault(parent, []).append(child)

    founder_cells = set()
    n_evolution_events = 0
    init_cnv_per_clone = {}

    if os.path.exists(event_log_path):
        event_df = pd.read_csv(event_log_path, sep="\t")
        if "iter" in event_df.columns and "cell_id" in event_df.columns:
            if "context" in event_df.columns:
                ctx = event_df["context"].astype(str).str.lower()
                init_mask = ctx.isin(["init_shared", "init_private"])
                init_rows = event_df[init_mask]
                if "clone_id" in init_rows.columns:
                    for cid in init_rows["clone_id"].unique():
                        try:
                            c = int(cid)
                            init_cnv_per_clone[c] = int((init_rows["clone_id"] == cid).sum())
                        except (TypeError, ValueError):
                            pass

            mask = event_df["iter"] > 0
            if "context" in event_df.columns:
                ctx = event_df["context"].astype(str).str.lower()
                mask = mask & ~ctx.isin(["init_shared", "init_private"])
            if "cell_type" in event_df.columns:
                mask = mask & (event_df["cell_type"].astype(str).str.lower() == "tumor")
            evolution_rows = event_df[mask]
            n_evolution_events = len(evolution_rows)
            for cid in evolution_rows["cell_id"]:
                try:
                    founder_cells.add(int(cid))
                except (TypeError, ValueError):
                    pass

    all_lineage_cells = set(parent_map.keys())
    founder_cells = founder_cells & all_lineage_cells

    def get_tumor_descendants(root_id):
        desc = set()
        stack = [root_id]
        visited = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if node in tumor_cell_ids:
                desc.add(node)
            for child in children_map.get(node, []):
                stack.append(child)
        return desc

    founder_details = []
    founder_descendants = {}
    for fc in sorted(founder_cells):
        desc = get_tumor_descendants(fc)
        frac = len(desc) / n_tumor_cells if n_tumor_cells > 0 else 0.0
        is_major = frac >= min_frac
        founder_details.append({
            "founder_cell": fc, "tumor_descendants": len(desc),
            "frac": frac, "is_major": is_major,
        })
        if is_major:
            founder_descendants[fc] = desc

    major_founders = [fd for fd in founder_details if fd["is_major"]]
    n_major_founders = len(major_founders)

    base_clone_ids = set(clone_ids[tumor_mask].tolist())
    base_clone_ids.discard(-1)
    n_base_clones = len(base_clone_ids)
    n_subclones = n_base_clones + n_major_founders

    cell_subclone = {}
    for idx in range(len(cell_ids)):
        if tumor_mask[idx]:
            cell_subclone[cell_ids[idx]] = f"Clone_{clone_ids[idx]}"
        else:
            cell_subclone[cell_ids[idx]] = "Normal"
    for fd in sorted(major_founders, key=lambda x: x["founder_cell"]):
        fc = fd["founder_cell"]
        for desc_id in founder_descendants[fc]:
            cell_subclone[desc_id] = f"Evol_{fc}"

    cell_n_evol = {cell_ids[i]: 0 for i in range(len(cell_ids))}
    for fc in founder_cells:
        stack = [fc]
        visited = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if node in cell_n_evol:
                cell_n_evol[node] += 1
            for child in children_map.get(node, []):
                stack.append(child)

    cell_n_total_cnv = {}
    for idx in range(len(cell_ids)):
        cid = cell_ids[idx]
        if tumor_mask[idx]:
            clone = clone_ids[idx]
            init_count = init_cnv_per_clone.get(clone, 0)
            cell_n_total_cnv[cid] = init_count + cell_n_evol.get(cid, 0)
        else:
            cell_n_total_cnv[cid] = 0

    stats = {
        "n_cells": n_cells, "n_tumor": n_tumor, "purity": purity,
        "n_base_clones": n_base_clones, "n_evolution_events": n_evolution_events,
        "n_major_founders": n_major_founders, "n_subclones": n_subclones,
        "founder_details": founder_details,
        "cell_subclone": cell_subclone, "cell_n_evol": cell_n_evol,
        "cell_n_total_cnv": cell_n_total_cnv,
        # Ancestry links that reference a cell absent from lineage.tsv (it died).
        # Non-zero means n_subclones / n_major_founders are lower bounds.
        "n_broken_lineage_links": n_broken_links,
        "subclone_counts_exact": n_broken_links == 0,
    }
    return res, stats


# ---------------------------------------------------------------------------
# QC plot generation
# ---------------------------------------------------------------------------

def generate_qc_plots(res, stats, rep_folder, rep_name):
    """Generate QC plots and save to rep_folder/qc/."""
    qc_dir = os.path.join(rep_folder, "qc")
    os.makedirs(qc_dir, exist_ok=True)

    td = res.get("tissue_data", {})
    coords = np.asarray(td.get("coordinates", []))
    cell_types = np.asarray(td.get("cell_type", []), dtype=str)
    cell_ids = np.asarray(res.get("cell_ids", []), dtype=int)

    if coords.ndim != 2 or coords.shape[0] == 0:
        print(f"  [WARN] No coordinate data for plotting in {rep_folder}")
        return

    x = coords[:, 0].astype(float)
    y = coords[:, 1].astype(float)

    cell_subclone = stats.get("cell_subclone", {})
    cell_n_total_cnv = stats.get("cell_n_total_cnv", {})
    subclone_arr = np.array([cell_subclone.get(cid, "Unknown") for cid in cell_ids])
    n_total_cnv_arr = np.array([cell_n_total_cnv.get(cid, 0) for cid in cell_ids])

    _plot_celltype_spatial(x, y, cell_types, qc_dir, rep_name)
    _plot_celltype_proportions(cell_types, qc_dir, rep_name, stats["purity"])
    _plot_subclone_spatial(x, y, subclone_arr, cell_types, qc_dir, rep_name,
                           stats["n_subclones"])
    _plot_cnv_events(x, y, n_total_cnv_arr, cell_types, qc_dir, rep_name)

    # ST-level plots (spot resolution)
    st_data = res.get("st_data", {})
    if st_data and "cell_by_spot" in st_data and "coordinates" in st_data:
        cell_subclone = stats.get("cell_subclone", {})
        _plot_spot_subclone_count(st_data, cell_subclone, qc_dir, rep_name)
        _plot_spot_major_subclone(st_data, cell_subclone, qc_dir, rep_name)

    print(f"  -> QC plots saved to {qc_dir}")


def _plot_celltype_spatial(x, y, cell_types, out_dir, rep_name):
    fig, ax = plt.subplots(figsize=(8, 7))
    unique_types = sorted(set(cell_types))
    plot_order = [ct for ct in unique_types if ct.lower() != "tumor"]
    if any(ct.lower() == "tumor" for ct in unique_types):
        plot_order.append("Tumor")
    for ct in plot_order:
        mask = np.char.lower(cell_types) == ct.lower()
        color = CT_COLORS.get(ct, "#999999")
        ax.scatter(x[mask], y[mask], c=color, s=8, alpha=0.6,
                   rasterized=True, linewidths=0)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(f"{rep_name}: Cell Types", fontsize=13, fontweight="bold")
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=CT_COLORS.get(ct, "#999999"),
                      markersize=8, label=ct) for ct in plot_order]
    ax.legend(handles=handles, fontsize=8, loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "sc_celltype_spatial.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_celltype_proportions(cell_types, out_dir, rep_name, purity):
    unique, counts = np.unique(cell_types, return_counts=True)
    fracs = counts / counts.sum()
    order = np.argsort(-fracs)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(unique))))
    colors = [CT_COLORS.get(unique[i], "#999999") for i in order]
    ax.barh(range(len(unique)), fracs[order], color=colors, edgecolor="white")
    ax.set_yticks(range(len(unique)))
    ax.set_yticklabels([f"{unique[i]} ({fracs[i]:.1%})" for i in order], fontsize=9)
    ax.set_xlabel("Fraction of cells")
    ax.set_title(f"{rep_name}: Cell Type Proportions (purity={purity:.3f})",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1); ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "celltype_proportions.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_subclone_spatial(x, y, subclone_arr, cell_types, out_dir, rep_name,
                           n_subclones):
    fig, ax = plt.subplots(figsize=(8, 7))
    normal_mask = np.char.lower(cell_types.astype(str)) != "tumor"
    ax.scatter(x[normal_mask], y[normal_mask], c="#e0e0e0", s=6,
               alpha=0.3, rasterized=True, linewidths=0)
    tumor_mask = ~normal_mask
    unique_clones = sorted(set(subclone_arr[tumor_mask]))
    cmap_vals = plt.cm.Set1(np.linspace(0, 1, max(len(unique_clones), 1)))
    for i, clone in enumerate(unique_clones):
        mask = (subclone_arr == clone) & tumor_mask
        ax.scatter(x[mask], y[mask], c=[cmap_vals[i % len(cmap_vals)]],
                   s=10, alpha=0.6, rasterized=True, linewidths=0)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(f"{rep_name}: Subclone Spatial ({n_subclones} subclones)",
                 fontsize=13, fontweight="bold")
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor="#e0e0e0", markersize=8, label="Normal")]
    for i, clone in enumerate(unique_clones):
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=cmap_vals[i % len(cmap_vals)],
                              markersize=8, label=clone))
    ax.legend(handles=handles, fontsize=8, loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "subclone_spatial.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_cnv_events(x, y, n_total_cnv, cell_types, out_dir, rep_name):
    fig, ax = plt.subplots(figsize=(8, 7))
    normal_mask = np.char.lower(cell_types.astype(str)) != "tumor"
    ax.scatter(x[normal_mask], y[normal_mask], c="#e0e0e0", s=6,
               alpha=0.3, rasterized=True, linewidths=0)
    tumor_mask = ~normal_mask
    if tumor_mask.any():
        sc = ax.scatter(x[tumor_mask], y[tumor_mask], c=n_total_cnv[tumor_mask],
                        cmap="YlOrRd", s=10, alpha=0.6, rasterized=True,
                        linewidths=0, vmin=0)
        plt.colorbar(sc, ax=ax, shrink=0.7, label="# total CNV events")
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(f"{rep_name}: Total CNV Events (initial + evolution)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "cnv_events_spatial.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def _compute_spot_subclone_info(st_data, cell_subclone):
    """For each spot, compute n_subclones and major subclone using cell_by_spot."""
    spot_coords = np.asarray(st_data["coordinates"])
    cell_by_spot = st_data["cell_by_spot"]
    n_spots = len(cell_by_spot)

    n_subclones_per_spot = np.zeros(n_spots, dtype=int)
    major_subclone_per_spot = np.full(n_spots, "Empty", dtype=object)

    for si in range(n_spots):
        spot_cell_ids = cell_by_spot[si]
        if not spot_cell_ids:
            continue
        labels = [cell_subclone.get(cid, "Unknown") for cid in spot_cell_ids]
        # Count only tumor subclones (exclude "Normal" and "Unknown")
        tumor_labels = [lb for lb in labels if lb not in ("Normal", "Unknown")]
        unique_tumor = set(tumor_labels)
        n_subclones_per_spot[si] = len(unique_tumor)
        if tumor_labels:
            from collections import Counter
            counts = Counter(tumor_labels)
            major_subclone_per_spot[si] = counts.most_common(1)[0][0]
        elif labels:
            major_subclone_per_spot[si] = "Normal"

    return spot_coords, n_subclones_per_spot, major_subclone_per_spot


def _plot_spot_subclone_count(st_data, cell_subclone, out_dir, rep_name):
    """ST-level plot: number of distinct tumor subclones per spot."""
    spot_coords, n_sub, _ = _compute_spot_subclone_info(st_data, cell_subclone)
    sx = spot_coords[:, 0].astype(float)
    sy = spot_coords[:, 1].astype(float)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(sx, sy, c=n_sub, cmap="YlGnBu", s=120, edgecolors="grey",
                    linewidths=0.5, vmin=0)
    plt.colorbar(sc, ax=ax, shrink=0.7, label="# tumor subclones in spot")
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(f"{rep_name}: Subclone Count per Spot (ST-level)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "st_subclone_count_spatial.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_spot_major_subclone(st_data, cell_subclone, out_dir, rep_name):
    """ST-level plot: major (most frequent) subclone per spot."""
    spot_coords, _, major_sub = _compute_spot_subclone_info(st_data, cell_subclone)
    sx = spot_coords[:, 0].astype(float)
    sy = spot_coords[:, 1].astype(float)

    unique_labels = sorted(set(major_sub))
    # Assign colours: Normal=grey, Empty=white, tumor subclones from Set1
    color_map = {}
    tumor_labels = [lb for lb in unique_labels if lb not in ("Normal", "Empty", "Unknown")]
    cmap_vals = plt.cm.Set1(np.linspace(0, 1, max(len(tumor_labels), 1)))
    for i, lb in enumerate(tumor_labels):
        color_map[lb] = cmap_vals[i % len(cmap_vals)]
    color_map["Normal"] = np.array([0.85, 0.85, 0.85, 1.0])
    color_map["Empty"] = np.array([1.0, 1.0, 1.0, 1.0])
    color_map["Unknown"] = np.array([0.95, 0.95, 0.95, 1.0])

    fig, ax = plt.subplots(figsize=(8, 7))
    for lb in unique_labels:
        mask = major_sub == lb
        c = color_map.get(lb, np.array([0.5, 0.5, 0.5, 1.0]))
        ax.scatter(sx[mask], sy[mask], c=[c], s=120, edgecolors="grey",
                   linewidths=0.5)

    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(f"{rep_name}: Major Subclone per Spot (ST-level)",
                 fontsize=13, fontweight="bold")
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=color_map.get(lb, [0.5]*3),
                      markeredgecolor="grey", markersize=10, label=lb)
               for lb in unique_labels]
    ax.legend(handles=handles, fontsize=8, loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "st_major_subclone_spatial.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
