#!/usr/bin/env python3
import os
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse

from . import formatting as f
from . import model as m
from . import utils as ut
from .cnv_logging import EventLogger


def run(
    n_initial_cells: int,
    chr_lengths,
    gene_table,
    common_gene_order,
    tumor_baseline_exp,
    normal_TME_exp,
    normal_celltype_proportions,
    lognorm_params_tumor,
    lognorm_params_normal,
    dirichlet_precision,
    action_rates: np.ndarray,
    mutation_rate: np.ndarray,
    transitions: np.ndarray,
    n_clusters: int = 10,
    st_eff=0.2,
    st_theta=10,
    max_iter: int = 500,
    max_cells: Union[int, float] = 1e4,
    domain_side_size: float = 200,
    init_coordinates: Optional[np.ndarray] = None,
    init_event_size: Optional[int] = None,
    fraction_drop: float = 0.05,
    fill_prob: float = 0.75,
    normal_action_rates: Optional[np.ndarray] = None,
    spot_radius: float = 2.5,
    n_spots_x: int = 10,
    n_spots_y: int = 10,
    output_dir: str = None,
    debug: bool = False,
    cnv_events_df: Optional[pd.DataFrame] = None,
    n_init_shared = 0,
    n_init_private=0,
    # --- Optional feature switches ---
    config: Dict = None,
    df_mapped_snps: Optional[pd.DataFrame] = None,
    generate_expression: bool = True,
    generate_st: bool = True,
    generate_baf: bool = True,
    skip_clustering: bool = False,
    n_jobs: int = 1,
) -> Dict[str, np.ndarray]:

    logger = EventLogger()
    #common_gene_order = np.sort(gene_table['gene'].unique())
    genome = m.Genome(gene_order=common_gene_order)
    genome_data = genome.load_real_data(chr_lengths, gene_table)
    genome_name=genome.gene_names

    # Read the optional config sections
    drip_conf   = (config or {}).get("drip", {})
    tls_conf    = (config or {}).get("tls", {})
    weight_conf = (config or {}).get("weights", {})
    celltype_groups_map = (config or {}).get("celltype_groups", {})
    perturb_conf = (config or {}).get("perturb", {})
    auto_conf    = (config or {}).get("auto_iter", {})
    mut_sched_conf = (config or {}).get("mutation_schedule", {})
    immune_soft_conf = (config or {}).get("immune_soft", {})
    auto_enable  = bool(auto_conf.get("enable", True))
    occ_target   = float(auto_conf.get("occupancy_target", 0.90))

    enable_drip      = bool(drip_conf.get("enable", True))
    drip_every       = int(drip_conf.get("every", 20))
    drip_fill_ratio  = float(drip_conf.get("fill_ratio", 0.05))
    immune_tls_share = float(drip_conf.get("immune_tls_share", 0.60))
    drip_per_cycle_max = drip_conf.get("per_cycle_max", None)
    drip_per_cycle_max = (int(drip_per_cycle_max) if drip_per_cycle_max is not None else None)

    tls_params = dict(
        n_tls           = int(tls_conf.get("n_tls", 2)),
        r_tls           = int(tls_conf.get("r_tls", 8)),
        r_block         = int(tls_conf.get("r_block", 2)),
        min_center_dist = float(tls_conf.get("min_center_dist", 15.0)),
        min_edge_gap    = float(tls_conf.get("min_edge_gap", 1.0)),
        border_margin   = int(tls_conf.get("border_margin", 0)),
        suppress_sigma  = float(tls_conf.get("suppress_sigma", 3.0)),
        tumor_block_spawn = float(tls_conf.get("tumor_block_spawn", 0.0)),  # soft suppression: lowers spawn
        tumor_block_death = float(tls_conf.get("tumor_block_death", 0.0)),  # soft suppression: raises death
        tls_celltypes     = tls_conf.get("tls_celltypes", ["T","B","DC","Plasma","T_reg"]),
    )

    weight_params = dict(
        mu_s = float(weight_conf.get("mu_s", 4.0)),
        w_s  = float(weight_conf.get("w_s", 5.0)),
        mu_i = weight_conf.get("mu_i", None),
        sigma_i = float(weight_conf.get("sigma_i", 2.5)),
        delta_epi = float(weight_conf.get("delta_epi", 2.0)),
        g_tls = float(weight_conf.get("g_tls", 7.0)),
    )

    # Three-mode immune placement config (inside/edge/outside probabilities)
    immune_place_conf = (config or {}).get("immune_placement", {})
    immune_place_probs = dict(
        inside=float(immune_place_conf.get("inside_prob", 0.33)),
        edge=float(immune_place_conf.get("edge_prob", 0.34)),
        outside=float(immune_place_conf.get("outside_prob", 0.33)),
    )
    # Optional shape parameters
    edge_sigma = float(immune_place_conf.get("edge_sigma", 1.5))
    inside_power = float(immune_place_conf.get("inside_power", 1.0))
    outside_sigma = float(immune_place_conf.get("outside_sigma", weight_params.get("sigma_i", 2.5)))
    # Whether to redraw the mixture proportions from a Dirichlet each time
    immune_mix_sample = bool(immune_place_conf.get("sample_mixture", False))
    _alpha = immune_place_conf.get("dirichlet_alpha", None)
    if _alpha is None:
        # Default alpha is probs*10; larger alpha means less variation
        immune_mix_alpha = (
            max(1e-6, immune_place_probs["inside"] * 10.0),
            max(1e-6, immune_place_probs["edge"]   * 10.0),
            max(1e-6, immune_place_probs["outside"]* 10.0),
        )
    else:
        immune_mix_alpha = (
            float(_alpha[0]),
            float(_alpha[1]),
            float(_alpha[2]),
        )

    print("Initialize : Tissue Growth")
    population, lineage, cell_map = m.grow_tissue(
        n_initial_cells,
        genome=genome,
        action_rates=action_rates,
        mutation_rate=mutation_rate,
        transitions=transitions,
        max_iter=max_iter,
        max_cells=max_cells,
        domain_side_size=domain_side_size,
        init_coordinates=init_coordinates,
        init_event_size=init_event_size,
        debug = debug,
        cnv_events_df=cnv_events_df,
        event_logger=logger,
        n_init_shared=n_init_shared,        # read from the design file
        n_init_private=n_init_private,
        enable_drip=enable_drip,
        drip_every=drip_every,
        drip_fill_ratio=drip_fill_ratio,
        immune_tls_share=immune_tls_share,
        drip_per_cycle_max=drip_per_cycle_max,
        tls_params=tls_params,
        weight_params=weight_params,
        normal_type_prop_map=normal_celltype_proportions,
        celltype_groups_map=celltype_groups_map,
        tls_celltypes=tls_params.get("tls_celltypes", []),
        forbid_tumor_on_tls=True,
        immune_place_probs=immune_place_probs,
        edge_sigma=edge_sigma,
        inside_power=inside_power,
        outside_sigma=outside_sigma,
        immune_mix_sample=immune_mix_sample,
        immune_mix_alpha=immune_mix_alpha,
        perturb_params=perturb_conf,
        auto_iter_enable=auto_enable,
        occupancy_target=occ_target,
        immune_soft_params=immune_soft_conf,
        mutation_schedule=mut_sched_conf,
    )
    n_aberrant_cells = len(population)

    print("Completed : Tissue growth | Total Cells : {}".format(n_aberrant_cells))

    if skip_clustering:
        print("Skip : Clustering (skip_clustering=True)")
        cluster_idx = np.zeros(len(population), dtype=int)
        cluster_lineage = np.zeros((1, 1), dtype=int)
    else:
        print("Initialized : Clustering")
        cluster_idx, cluster_genome = f.add_cluster(population, n_clusters)
        id_parent_map = {c.id: c.parent for c in population}
        cluster_lineage = f.get_cluster_lineage(population, cluster_idx, id_parent_map)
        print("Completed : Clustering")

    # Note: a one-shot add_background(...) call would conflict with the incremental drip below.
    # print("Initialized : Add normal cells")
    # population, dropped = m.add_background(
    #     population,
    #     genome,
    #     cell_map,
    #     normal_celltype_proportions,
    #     fraction_drop=fraction_drop,
    #     fill_prob=fill_prob,
    #     normal_action_rates=normal_action_rates
    # )
    # n_aberrant_cells -= len(dropped)
    dropped = []  # kept for interface compatibility
    n_normal_cells = len(population) - n_aberrant_cells

    cluster_idx = np.array([x for k, x in enumerate(cluster_idx) if k not in dropped])
    cluster_idx = np.append(
        cluster_idx,
        (max(cluster_idx) + 1) * np.ones(len(population) - len(cluster_idx)),
    )

    print("Completed : Add normal cells | Total Cells : {}".format(len(population)))
    if generate_st:
        print("Initialized : Tissue Assembly")
        tissue_data = f.assemble_tissue_data(
            population,
            tumor_baseline_exp,
            normal_TME_exp,
            lognorm_params_tumor,
            lognorm_params_normal,
            dirichlet_precision,
            common_gene_order,
            output_dir=output_dir,
            generate_expression=generate_expression,
            n_jobs=n_jobs,
        )
        tissue_data["cluster_index"] = cluster_idx
        print("Completed : Tissue Assembly")

        print("Initialized : In silico Spatial Transcriptomics")
        _cell_ids_for_st = np.array([c.id for c in population])
        st_data = f.in_silico_st(
            tissue_data,
            common_gene_order,
            domain_side_size,
            eff=st_eff,
            theta=st_theta,
            spot_radius=spot_radius,
            n_spots_x=n_spots_x,
            n_spots_y=n_spots_y,
            cell_ids=_cell_ids_for_st,
        )
        print("Completed : In silico Spatial Transcriptomics")
    else:
        print("Skip : Tissue Assembly / Spatial Transcriptomics (generate_st=False)")
        clone_ids = np.array([cell.clone_id for cell in population], dtype=object)
        tissue_data = dict(
            coordinates=np.array([cell.crd for cell in population]),
            # Fast proxy for benign status in quick mode:
            # normal injected cells have clone_id=None, tumor cells have clone_id!=None.
            cell_status=np.array([1 if cid is None else 0 for cid in clone_ids], dtype=np.int8),
            clone_id=clone_ids,
            cell_type=np.array([cell.cell_type for cell in population], dtype=object),
            cluster_index=cluster_idx,
        )
        st_data = {}

    # ==========================================
    # BAF matrix generation
    # ==========================================

    # 1. SNP-to-gene mapping table (df_mapped_snps)
    #    Loaded by the caller from the file named in [files].snp_file.
    #    It must carry a gene identifier column.
    # df_mapped_snps = pd.read_csv(snp_file)


    if generate_baf and generate_st and (df_mapped_snps is not None):
        print("Starting BAF generation pipeline...")

        # 2. Build gene_names_dict for utils.get_sc_gene_count_df
        #    Keys must look like "chr1A", "chr1B", ...
        gene_names_dict = {}
        unique_chrs = gene_table['chr'].unique()

        for chrom in unique_chrs:
            # Genes on this chromosome
            genes_on_chrom = gene_table[gene_table['chr'] == chrom]['gene'].values

            # Normalise the chromosome key
            # 1. Cast to string and strip any 'chr' prefix, leaving "1", "2", ...
            chrom_clean = str(chrom).replace("chr", "")

            # 2. Reassemble in the Genome class format: "chr" + number + "A"/"B"
            key_A = f"chr{chrom_clean}A"
            key_B = f"chr{chrom_clean}B"

            # Assign
            gene_names_dict[key_A] = genes_on_chrom
            gene_names_dict[key_B] = genes_on_chrom

        # Print the first few keys to confirm they match the Genome class
        print(f"DEBUG: Generated keys: {list(gene_names_dict.keys())[:4]}")
        # expected: ['chr1A', 'chr1B', 'chr2A', 'chr2B']

        # 3. Collect per-cell haplotype counts
        sc_genome_per_chr = []
        cell_names = []

        surviving_cells = [c for c in population]
        print(f"Extracting genome data for {len(surviving_cells)} cells...")

        for cell in population: # wrap in tqdm(population) for a progress bar
            cell_names.append(cell.id)
            sc_genome_per_chr.append(cell.genome.get_chr_gene_count())

        # 4. Build the wide haplotype table
        #    gene_names_dict maps the chrXA/chrXB counts back onto individual genes
        df_haplotypes = ut.get_sc_gene_count_df(sc_genome_per_chr, cell_names, gene_names_dict)
        # 5. Generate the AD/DP matrices
        print("Generating AD/DP matrices (this may take a while)...")
        AD, DP = f.generate_baf_matrix(
            st_data=st_data,
            df_haplotypes=df_haplotypes,
            df_mapped_snps=df_mapped_snps,
            common_gene_order=common_gene_order  # required to locate expression columns
        )

        # 6. Save the results in Matrix Market format
        out_baf_dir = os.path.join(output_dir, "synthetic", "baf")
        os.makedirs(out_baf_dir, exist_ok=True)
        print(f"Saving BAF matrices to {out_baf_dir}...")
        scipy.io.mmwrite(os.path.join(out_baf_dir, "AD.mtx"), scipy.sparse.csr_matrix(AD))
        scipy.io.mmwrite(os.path.join(out_baf_dir, "DP.mtx"), scipy.sparse.csr_matrix(DP))
        df_mapped_snps.to_csv(os.path.join(out_baf_dir, "snp_info.csv"), index=False)

        pd.Series(df_mapped_snps['snp_id']).to_csv(os.path.join(out_baf_dir, "snps.tsv"), sep='\t', header=False, index=False)
        # Spot names follow the row order of st_data['expression']; emit them as barcodes.tsv
        st_barcodes = [f"Spot_{i}" for i in range(st_data['expression'].shape[0])]
        pd.Series(st_barcodes).to_csv(os.path.join(out_baf_dir, "barcodes.tsv"), sep='\t', header=False, index=False)

        print("BAF matrices saved.")

    lineage = np.array(
        [(c, p) for k, (c, p) in enumerate(lineage.items()) if k not in dropped]
    )

    lineage = np.vstack((lineage, -1 * np.ones((n_normal_cells, 2))))

    surviving_ids = np.array([c.id for c in population])

    res = dict(
        tissue_data=tissue_data,
        lineage=lineage,
        cluster_lineage=cluster_lineage,
        st_data=st_data,
        genome_data=genome_data,
        gene_name=genome_name,
        cell_ids       = surviving_ids,
        event_log = logger.to_dataframe(),
    )

    return res
