from typing import Dict, List, Tuple

from . import model as m
from . import utils
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from joblib import Parallel, delayed
import os, tempfile
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats
from tqdm import tqdm

def add_cluster(
    population: List[m.Cell],
    n_clusters: int,
) -> Tuple[np.ndarray, np.ndarray]:

    genome_profile = np.vstack(
        [x.genome.get_total_gene_count()[np.newaxis, :] for x in population]
    )
    cluster = AgglomerativeClustering(
        n_clusters=n_clusters,linkage="ward"
    )

    idx = cluster.fit_predict(genome_profile)
    cluster_genome = np.zeros((n_clusters, population[0].genome.G))

    for clu in np.unique(idx):
        pos = idx == clu
        cluster_genome[clu, :] = genome_profile[pos, :].mean(axis=0).round()

    return idx, cluster_genome

def get_cluster_lineage(
    population: List[m.Cell],
    cluster_idx: np.ndarray,
    lineage: Dict[int, int],
) -> np.ndarray:

    n_cluster = np.unique(cluster_idx).shape[0]
    pos_to_id = {x: population[x].id for x in range(len(population))}
    id_to_pos = {v: k for k, v in pos_to_id.items()}
    cmat = np.zeros((n_cluster, n_cluster))

    for k1 in range(n_cluster):
        c_members = np.where(cluster_idx == k1)[0]
        for mb in c_members:
            i = pos_to_id[mb]
            k2 = k1
            while (k2 == k1) and (i > 0):
                # Defensive check: bail out if the key is missing from lineage, rather than raising KeyError
                if i not in lineage:
                    break
                i = lineage[i]
                if i in id_to_pos.keys():
                    k2 = cluster_idx[id_to_pos[i]]

            cmat[k1, k2] += 1

    return cmat

def assemble_tissue_data(
    population: List[m.Cell],
    tumor_baseline_exp,
    normal_TME_exp,
    lognorm_params_tumor,
    lognorm_params_normal,
    dirichlet_precision,
    common_gene_order,
    # Default 1: the worker is GIL-bound (pandas reindex dominates), so threading
    # measured no speedup, while sharing the global numpy RNG across threads made
    # the expression counts non-reproducible for a fixed seed.
    n_jobs: int = 1,
    output_dir: str = None,
    generate_expression: bool = True,
) -> Dict[str, np.ndarray]:
    n_cells = len(population)
    G       = population[0].genome.G

    # 1) Build a temporary memmap with dtype uint16 (a single gene's count is always < 65k)
    if output_dir is None:
        output_dir = os.getcwd()
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    fname   = os.path.join(tmp_dir, "expr_{}x{}.uint16".format(n_cells, G))
    expr_mm = np.memmap(fname, dtype=np.uint16, mode="w+", shape=(n_cells, G))

    if generate_expression:
        # 2) Define the worker: compute one row, write one row
        def _write_row(i, cell):
            cnt = cell.sample_expression_with_random_count(
                tumor_baseline_exp=tumor_baseline_exp,
                normal_TME_exp=normal_TME_exp,
                lognorm_params_tumor=lognorm_params_tumor,
                lognorm_params_normal=lognorm_params_normal,
                dirichlet_precision=dirichlet_precision,
            ).astype(np.uint16)
            expr_mm[i] = cnt
        # 3) Write in parallel (n_jobs=1 for a simple single-threaded run)
        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_write_row)(i, cell) for i, cell in enumerate(population)
        )
        expr_mm.flush()           # make sure it is flushed to disk
    else:
        # Fast mode: skip expression generation and write a zero matrix
        expr_mm[:] = 0
        expr_mm.flush()

    crd = np.array([x.crd for x in population])
    genome = np.vstack([x.genome.get_total_gene_count()[np.newaxis, :] for x in population]).astype(
        np.uint8
    )

    genome_per_chr = [cell.genome.get_chr_gene_count() for cell in population]

    benign = np.array([
        int(all(np.all(counts == 1) for counts in cell.genome.get_chr_gene_count().values()))
        for cell in population
    ])

    clone_ids = np.array([cell.clone_id for cell in population])
    cell_types = np.array([cell.cell_type for cell in population])

    res = dict(
        expression=expr_mm,
        coordinates=crd,
        genome_profile=genome,
        cell_status=benign,
        clone_id=clone_ids,
        cell_type=cell_types,
        genome_per_chr=genome_per_chr
    )

    return res

def in_silico_st(
    tissue_data: Dict[str, np.ndarray],
    common_gene_order,
    domain_side_size: float,
    eff=0.2,
    theta=10,
    spot_radius: float = 2.5,
    n_spots_x: int = 10,
    n_spots_y: int = 10,
    output_dir: str = None,
    cell_ids: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    # 1) Build a temporary memmap with dtype uint16 (a single gene's count is always < 65k)
    if output_dir is None:
        output_dir = os.getcwd()
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    radius_squared = spot_radius ** 2

    xx = np.linspace(spot_radius, domain_side_size - spot_radius, n_spots_x)
    yy = np.linspace(spot_radius, domain_side_size - spot_radius, n_spots_y)

    xx, yy = np.meshgrid(xx, yy)
    xx = xx.flatten()
    yy = yy.flatten()

    n_genes = tissue_data["expression"].shape[1]

    exp_matrix = np.zeros((xx.shape[0], n_genes))
    gen_matrix = np.zeros((xx.shape[0], n_genes))
    ben_matrix = np.zeros(xx.shape[0])
    cell_count = np.zeros(xx.shape[0])
    cell_id_list = []
    clone_id_by_spot = []
    cell_type_by_spot = []

    #mrna_pool_list = []

    make_cluster_matrix = "cluster_index" in tissue_data.keys()
    if make_cluster_matrix:
        n_clusters = np.unique(tissue_data["cluster_index"]).shape[0]
        cluster_matrix = np.zeros((xx.shape[0], n_clusters))

    for spot, (x, y) in enumerate(zip(xx, yy)):
        # Squared Euclidean distance from every cell to the spot centre
        distances_squared = np.sum((tissue_data["coordinates"] - np.array([x, y]))**2, axis=1)
        in_spot = np.where(distances_squared < radius_squared)[0]

        if cell_ids is not None:
            cell_id_list.append([int(cell_ids[i]) for i in in_spot])
        else:
            cell_id_list.append(in_spot.tolist())

        if "clone_id" in tissue_data and "cell_type" in tissue_data:
            clone_id_by_spot.append(tissue_data["clone_id"][in_spot].tolist())
            cell_type_by_spot.append(tissue_data["cell_type"][in_spot].tolist())
        else:
            clone_id_by_spot.append([])
            cell_type_by_spot.append([])

        # Skip spots that contain no cells (their output stays zero)
        if in_spot.size == 0:
            exp_matrix[spot, :] = 0
            gen_matrix[spot, :] = 0
            ben_matrix[spot] = 0
            cell_count[spot] = 0
            if make_cluster_matrix:
                cluster_matrix[spot, :] = 0
            continue
        # Aggregate expression: sum the expression vectors of all cells in the spot
        joint_expr = tissue_data["expression"][in_spot, :].sum(axis=0)
        exp_matrix[spot, :] = joint_expr.astype(int)

        cell_count[spot] = len(in_spot)

        # Aggregate genotype: mean genome_profile over the cells in the spot
        av_gen = tissue_data["genome_profile"][in_spot].mean(axis=0)
        gen_matrix[spot, :] = av_gen

        # Call a spot benign when the mean benign status is 0.9 or above
        state_ben = int(tissue_data["cell_status"][in_spot].mean() >= 0.9)
        ben_matrix[spot] = state_ben

        if make_cluster_matrix:
            clu, cnt = np.unique(
                tissue_data["cluster_index"][in_spot], return_counts=True
            )

            cluster_matrix[spot, clu.astype(int)] = cnt / cnt.sum()

    crd = np.hstack((xx[:, np.newaxis], yy[:, np.newaxis]))

    res = dict(
        expression=exp_matrix,
        coordinates=crd,
        genome_profile=gen_matrix,
        spot_status=ben_matrix,
        cluster_index=(cluster_matrix if make_cluster_matrix else None),
        cell_count=cell_count,
        cell_by_spot=cell_id_list,
        clone_id_by_spot=np.array(clone_id_by_spot, dtype=object),
        cell_type_by_spot=np.array(cell_type_by_spot, dtype=object)
    )

    return res

def load_and_map_snps(snp_file_path, gene_table, common_gene_order):
    """
    Read a CSV that already carries the SNP-to-gene mapping.
    No expensive coordinate mapping is performed here.
    """
    print(f"Loading pre-mapped SNPs from {snp_file_path}...")

    # 1. Read the CSV directly
    try:
        df = pd.read_csv(snp_file_path)
    except Exception as e:
        print(f"Error reading SNP file: {e}")
        return None

    # 2. Check that the required columns are present
    required_cols = ['snp_id', 'gene_id', 'chrom', 'pos']
    # accept 'chr' as an alias for 'chrom'
    if 'chr' in df.columns and 'chrom' not in df.columns:
        df.rename(columns={'chr': 'chrom'}, inplace=True)

    if not set(required_cols).issubset(df.columns):
        print(f"Error: CSV file must contain columns: {required_cols}")
        print(f"Found columns: {df.columns}")
        return None

    # 3. Clean up: enforce a consistent format
    # Cast chromosomes to string and strip the 'chr' prefix (e.g. "chr1" -> "1")
    # This keeps them consistent with the Genome class and gene_table downstream
    df['chrom'] = df['chrom'].astype(str).str.replace('chr', '', regex=False)

    # 4. Filter: keep only the genes of interest (common_gene_order)
    # This matters: dropping SNPs on genes outside the simulated set saves memory
    print(f"Filtering SNPs for {len(common_gene_order)} target genes...")
    original_count = len(df)

    # Build a set for fast lookup
    target_genes = set(common_gene_order)
    df_final = df[df['gene_id'].isin(target_genes)].copy()

    # 5. Sort (optional, but recommended)
    # Sort by chromosome then position
    # Note: sorting '1', '2', '10' correctly would need special handling
    # a plain value sort is enough here
    df_final.sort_values(by=['chrom', 'pos'], inplace=True)

    print(f"Done! Loaded {len(df_final)} SNPs (filtered from {original_count}).")

    return df_final

def prepare_and_run_haplotype_matrix(population, gene_table, df_mapped_snps):
    """
    Build the haplotype matrix using utils.get_sc_gene_count_df.
    """

    # --- 1. Prepare gene_names_dict ---
    # get_sc_gene_count_df expects keys "1A"/"1B" mapping to the gene names on that chromosome copy
    gene_names_dict = {}

    # Assumes gene_table['chr'] holds "1", "2", ... (mind the match if it carries a chr prefix)
    unique_chrs = gene_table['chr'].unique()

    for chrom in unique_chrs:
        # Keep the gene order identical to the order inside the Genome class
        # if gene_table is already sorted, take it as is
        genes = gene_table[gene_table['chr'] == chrom]['gene'].values

        # The helper reads copy = chrom[-1], so keys must end in A or B
        # Build "1A", "1B", "2A", "2B", ...
        gene_names_dict[f"{chrom}A"] = genes
        gene_names_dict[f"{chrom}B"] = genes

    # --- 2. Prepare sc_genome_per_chr and cell_names ---
    sc_genome_per_chr = []
    cell_names = []

    print("Extracting genome counts from population...")
    for cell in tqdm(population):
        cell_names.append(cell.id)
        # Call the per-cell accessor on the genome view
        sc_genome_per_chr.append(cell.genome.get_chr_gene_count())

    # --- 3. Delegate to the shared helper ---
    print("Building Haplotype DataFrame (using existing utils function)...")
    # reuse utils.get_sc_gene_count_df unchanged
    df_haplotypes = utils.get_sc_gene_count_df(sc_genome_per_chr, cell_names, gene_names_dict)

    return df_haplotypes


def generate_baf_matrix(
    st_data: dict,
    df_haplotypes: pd.DataFrame,
    df_mapped_snps: pd.DataFrame,
    common_gene_order: list,  # needed to look up column indices in the expression matrix
    concentration: float = 100.0
):
    """
    Generate the AD (alt depth) and DP (total depth) matrices needed for BAF analysis.
    This version expects:
    1. a 'gene_id' column in df_mapped_snps,
    2. 'Gene_A' / 'Gene_B' style columns in df_haplotypes,
    3. st_data['expression'] as the source of the observed depth.
    """

    print("Generating BAF matrices (AD/DP)...")

    n_spots = st_data['expression'].shape[0]
    n_snps = len(df_mapped_snps)
    cell_by_spot = st_data['cell_by_spot']

    # Initialise the matrices
    AD = np.zeros((n_spots, n_snps), dtype=int)
    DP = np.zeros((n_spots, n_snps), dtype=int)

    # -------------------------------------------------------------
    # 1. Build the lookup maps
    # -------------------------------------------------------------

    # A. SNP -> Gene Name
    # load_and_map_snps emits a 'gene_id' column
    snp_genes = df_mapped_snps['gene_id'].values

    # B. Gene Name -> Haplotype Matrix Columns
    col_A_names = [f"{g}_A" for g in snp_genes]
    col_B_names = [f"{g}_B" for g in snp_genes]

    # Check the gene is present in the haplotype table (tolerant)
    # df_haplotypes columns look like "TP53_A", "TP53_B"
    valid_mask = [
        (a in df_haplotypes.columns) and (b in df_haplotypes.columns)
        for a, b in zip(col_A_names, col_B_names)
    ]

    valid_indices = np.where(valid_mask)[0]

    # Only process valid SNPs
    final_cols_A = np.array(col_A_names)[valid_indices]
    final_cols_B = np.array(col_B_names)[valid_indices]
    final_snp_genes = snp_genes[valid_indices] # gene names for these SNPs

    print(f"Valid SNPs for BAF: {len(valid_indices)} / {n_snps}")

    # C. Gene name -> expression matrix column index (used to read the depth)
    # Build a lookup: gene_name -> column index in st_data['expression']
    gene_to_idx = {g: i for i, g in enumerate(common_gene_order)}

    # Find the expression column for each valid SNP
    # Genes absent from common_gene_order (e.g. filtered out) get -1
    expr_col_indices = np.array([gene_to_idx.get(g, -1) for g in final_snp_genes])

    # Filter again: keep only genes that are also in the expression matrix
    expr_valid_mask = expr_col_indices != -1

    # Final valid SNP indices (row numbers in the original df_mapped_snps)
    final_valid_snp_indices = valid_indices[expr_valid_mask]
    # matching expression matrix column numbers
    final_expr_col_indices = expr_col_indices[expr_valid_mask]
    # matching haplotype matrix column names
    final_cols_A = final_cols_A[expr_valid_mask]
    final_cols_B = final_cols_B[expr_valid_mask]

    # -------------------------------------------------------------
    # 2. Set up fast lookups
    # -------------------------------------------------------------

    # Map cell_id -> row number in the matrix
    cell_id_map = {cid: i for i, cid in enumerate(df_haplotypes.index)}

    # Pre-extract the haplotype data we need into numpy, shape (n_cells, n_final_snps)
    # costs some memory, but is much faster than looping
    haplo_vals_A = df_haplotypes[final_cols_A].values
    haplo_vals_B = df_haplotypes[final_cols_B].values

    # -------------------------------------------------------------
    # 3. Compute spot by spot
    # -------------------------------------------------------------

    for s_idx in tqdm(range(n_spots), desc="Calculating BAF"):
        spot_cell_ids = cell_by_spot[s_idx]
        if not spot_cell_ids:
            continue

        # Row numbers of every cell in this spot within the haplotype matrix
        row_indices = [cell_id_map[cid] for cid in spot_cell_ids if cid in cell_id_map]

        if not row_indices:
            continue

        # --- Compute theta (BAF) ---
        # Pull the A/B copy numbers of these cells
        spot_haplo_A = haplo_vals_A[row_indices, :] # (n_cells_in_spot, n_snps)
        spot_haplo_B = haplo_vals_B[row_indices, :]

        sum_A = spot_haplo_A.sum(axis=0)
        sum_B = spot_haplo_B.sum(axis=0)
        total_CN = sum_A + sum_B

        with np.errstate(divide='ignore', invalid='ignore'):
            theta = sum_B / total_CN
            theta[total_CN == 0] = 0.5 # no copies present: fall back to 0.5

        # --- Read DP (depth) ---
        # Take the read count for this spot's genes straight from the expression matrix
        # final_expr_col_indices holds each gene's position in the expression matrix
        current_DP = st_data['expression'][s_idx, final_expr_col_indices]

        # Depth 0: nothing to draw, leave AD at 0
        has_reads = current_DP > 0
        if not np.any(has_reads):
            continue

        # --- Draw AD (alt count) ---
        # only draw where reads exist
        active_theta = theta[has_reads]
        active_DP = current_DP[has_reads].astype(int)

        # Clip theta away from the boundaries
        active_theta = np.clip(active_theta, 1e-4, 1-1e-4)

        # Beta-binomial sampling
        rvs = stats.betabinom.rvs(
            n=active_DP, # total reads
            a=active_theta * concentration, # alpha
            b=(1 - active_theta) * concentration
        )

        # Write into the result matrices
        # note: final_valid_snp_indices[has_reads] gives the original SNP positions
        target_snp_indices = final_valid_snp_indices[has_reads]

        AD[s_idx, target_snp_indices] = rvs
        DP[s_idx, target_snp_indices] = active_DP

    return AD, DP
