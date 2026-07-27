# Output reference

Layout of `<output>/` after `stcnv-simulate`. Tables are written as Parquet by default and
as TSV with `--no-parquet`; the schemas below are identical either way. In the Parquet
files the row label becomes a leading column named `index`.

```
<output>/
├── res.pkl
├── event_log.tsv
├── quick_purity_summary.tsv        (--quick-purity-only only)
├── baf/
├── qc/
└── synthetic/
    ├── genome.tsv
    ├── lineage.tsv
    ├── lineage_all.tsv
    ├── all_cell_ids.tsv
    ├── cluster_lineage.tsv
    ├── single_cell_data/
    ├── spatial_data/
    └── visual/
```

## Ground truth: what a benchmark should score against

| Question | File |
|---|---|
| Which segments are altered, in which cell, to what allelic state? | `event_log.tsv` |
| What is each cell's copy number, per gene? | `single_cell_data/sc-genome_profile` |
| What is each spot's average copy number, per gene? | `spatial_data/st-genome_profile` |
| Which cells belong to which clone? | `single_cell_data/sc-meta_data` (`clone_id`) |
| How are clones related? | `lineage.tsv`, `lineage_all.tsv` |
| Which spots are tumour vs benign? | `spatial_data/st-annotation` |
| Which cells landed in which spot? | `spatial_data/cell_by_spot.txt` |
| Where does each gene sit on the genome? | `genome.tsv` |

## `event_log.tsv`

One row per CNV event applied during the run — the primary ground truth.

| Column | Meaning |
|---|---|
| `iter` | iteration the event occurred at; `0` for founder events |
| `cell_id` | cell that acquired the event |
| `parent_id` | that cell's parent (`-1` for a founder) |
| `clone_id` | founder clone the cell descends from |
| `cell_type` | always `Tumor` (normal cells do not mutate) |
| `source` | `catalog` when drawn from the CNV catalogue |
| `context` | `init_shared`, `init_private`, or an evolution event |
| `row_id` | row in the input CNV catalogue the event came from |
| `Chromosome`, `Start`, `End` | segment, `chr`-prefixed, 1-based inclusive |
| `Major_Copy_Number`, `Minor_Copy_Number` | allelic state applied |
| `length_mb` | segment length in megabases |
| `hap_assigned` | resulting per-haplotype counts, e.g. `chr7B:2;chr7A:0` |
| `affected_genes` | `;`-separated genes inside the segment |

`init_shared` events are applied identically to every founder clone; `init_private` events
are drawn per clone. Which haplotype receives the major copy number is decided
reproducibly from the segment's coordinates and state.

## `synthetic/single_cell_data/`

| File | Shape | Contents |
|---|---|---|
| `sc-expression` | cells × genes | raw integer counts; row label `Cell_<id>` |
| `sc-genome_profile` | cells × genes | total copy number per gene (0–255) |
| `sc-meta_data.tsv` | cells × 6 | see below |

`sc-meta_data.tsv` columns:

| Column | Meaning |
|---|---|
| `x`, `y` | integer lattice coordinates |
| `cluster` | cluster index from hierarchical clustering of genotypes |
| `cell_status` | `1` if the genome is entirely diploid (benign), else `0` |
| `clone_id` | founder clone for tumour cells; empty for infused normal cells |
| `cell_type` | `Tumor`, or a normal type such as `Hepatocyte`, `T`, `Fibroblast` |

Row labels are the simulator's own cell IDs, not row numbers — IDs are not contiguous,
because cells die during growth. Cross-reference `all_cell_ids.tsv`.

## `synthetic/spatial_data/`

| File | Shape | Contents |
|---|---|---|
| `st-expression` | **genes × spots** | spot counts after capture and overdispersion |
| `st-genome_profile` | **genes × spots** | mean copy number over cells in the spot |
| `st-meta` | spots × 3 | `x`, `y`, `cell_count` |
| `st-annotation` | spots × 1 | `benign` or `aberrant` |
| `cluster_data.tsv` | spots × clusters | per-spot cluster composition |
| `cell_by_spot.txt` | one line per spot | `Spot_<k> : <cell_id>,<cell_id>,...` |

Note the transpose: the single-cell matrices are cells × genes, while the two spot
matrices are genes × spots. `st-meta` and `st-annotation` are spots × columns.

A spot is called `benign` when the mean benign status of its cells is ≥ 0.9. Spots
containing no cells are all-zero and annotated from an empty mean.

## `synthetic/` top level

| File | Contents |
|---|---|
| `genome.tsv` | `gene`, `chr`, `start`, `end`, `mid`. Row label equals the `gene` column, and rows are in the same order as the expression matrix columns. |
| `lineage.tsv` | `child_id`, `parent_id` for **surviving tumour cells**. `parent_id` is a cell ID, or `initial_seed` for a founder, or `dropped_cell` if the parent is unknown. |
| `lineage_all.tsv` | the same for **every** cell ever created, including those that died; `parent_id` is numeric, `-1` for a founder. |
| `all_cell_ids.tsv` | `cell_id`, `cell_name`, `is_survived` — the map from IDs to the final tables. |
| `cluster_lineage.tsv` | clusters × clusters ancestry counts. |

`lineage.tsv` is the tree you can compare a clone-inference method against;
`lineage_all.tsv` is the true generative history, which no method can recover in full
because dead cells leave no data.

## `baf/`

Allele counts for BAF-based methods. Rows are **spots**, columns are **SNPs**.

| File | Contents |
|---|---|
| `AD.mtx` | Matrix Market: alt-allele depth |
| `DP.mtx` | Matrix Market: total depth, taken from the spot's expression of the SNP's gene |
| `snp_info.csv` | `snp_id`, `gene_id`, `chrom`, `pos` — rows align with the matrix columns |
| `barcodes.tsv` | spot names, headerless, in matrix row order |
| `snps.tsv` | SNP IDs, headerless |

AD is drawn beta-binomially around the true allelic ratio implied by each cell's
haplotype-specific copy number, so allelic imbalance tracks the simulated CNVs. Depth 0
means the gene had no reads in that spot.

Convert to Numbat's long format with:

```bash
stcnv-export-numbat --baf-dir <output>/baf \
  --snp-file <output>/baf/snp_info.csv \
  --meta-file <metadata with a cell_id column> \
  --genetic-map genetic_map_hg38_withX.txt.gz \
  --output <output>/numbat_allele_counts.tsv.gz
```

The export emits `cell, snp_id, CHROM, POS, cM, REF, ALT, AD, DP, GT, gene`, restricted to
`DP > 0` and sorted by coordinate then cell. `CHROM` is written without the `chr` prefix,
as Numbat expects.

## `res.pkl`

The complete in-memory result: `tissue_data`, `st_data`, `lineage`, `lineage_all`,
`cluster_lineage`, `genome_data`, `cell_ids`, `all_cell_ids`, `event_log`. Everything the
TSV/Parquet files hold, plus `genome_per_chr` (per-cell, per-haplotype gene counts) which
is not otherwise exported.

Pickles are version-fragile: read them with the same package version that wrote them, and
prefer the tabular files for anything you intend to keep.

## `qc/` and `synthetic/visual/`

Diagnostic figures, skipped entirely with `--skip-visual`.

| File | Shows |
|---|---|
| `qc/sc_celltype_spatial.pdf` | single cells coloured by cell type |
| `qc/st_celltype_proportions.pdf` | per-spot proportion map, one panel per cell type |
| `synthetic/visual/tissue.png` | aberrant-gene burden across the tissue |
| `synthetic/visual/tissue_w_array.png` | the same, with spot centres overlaid |

`stcnv-batch` writes further per-replicate QC figures covering subclone spatial layout,
per-spot subclone counts and CNV burden.

## Scratch files

Expression generation is backed by a memmap under `<output>/tmp/`, which is deleted once
the run completes. If a run is killed mid-way the directory may survive and can be removed
safely.