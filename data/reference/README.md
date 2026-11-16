# Reference data

The simulator needs six reference inputs, named under `[files]` in a design file. Four are
small enough to ship here; two are not, and must be supplied.

## Bundled

| File | Columns | Notes |
|---|---|---|
| `chr_lengths.csv` | `chr,length` | GRCh38 chromosome lengths, chr1–22 plus X and Y |
| `gene_order.csv` | `gene,chr,start,end` | GRCh38 gene coordinates, 36,558 rows |
| `celltype.tsv` | rowname `<barcode>`, one cell-type column | Cell-type labels for the reference cells |
| `mapped_snps.csv` | `snp_id,gene_id,chrom,pos` | Heterozygous SNPs assigned to genes |

Notes on the annotation:

- `gene_order.csv` contains **10 duplicated gene symbols** (e.g. `GGT1`, `CYB561D2`) that
  appear on two overlapping intervals. The simulator de-duplicates the gene table itself,
  but your reference expression matrix must have a **unique** gene index or expression
  generation will fail with `cannot reindex on an axis with duplicate labels`.
- `chr_lengths.csv` and `gene_order.csv` must use the same genome build as your CNV
  catalogue, and the same chromosome naming (`chr`-prefixed).
- `mapped_snps.csv` was built with `sptrace-prep-snps`; regenerate it if you change the gene
  annotation.

## Not bundled — you must supply these

### `refexp_file` — reference expression matrix

Tab-separated, **genes as rows, cells as columns**, raw counts, with a unique gene index:

```
        Cell_1  Cell_2  Cell_3
GENE_A       12       0       7
GENE_B        3      41       0
```

The simulator aggregates it by cell type (see `generate_normal_expression`) to derive one
mean profile per normal cell type, and the profile of `normal_original_celltype` seeds the
tumour baseline. Any single-cell or single-nucleus RNA-seq reference from the tissue you
want to mimic will do; it must share gene identifiers with `gene_order.csv` and cell
barcodes with `celltype.tsv`.

Not shipped because it is a few hundred megabytes.

### `cnv_catalog_file` — allele-specific CNV catalogue

Comma-separated, one row per observed segment. CNV events are drawn from this table, which
is what makes the simulated copy-number landscape realistic rather than uniform:

```
Chromosome,Start,End,Copy_Number,Major_Copy_Number,Minor_Copy_Number
chr1,1000000,5400000,3,2,1
chr7,55000000,61000000,2,2,0
```

`Copy_Number` may also be named `Total_Copy_Number`; column matching is
case-insensitive. Balanced diploid segments (`Major=2, Minor=2`) are dropped on load, and
segments covering fewer than 5 gene midpoints are filtered out.

We derive ours from TCGA allele-specific copy-number calls (ASCAT). Those require
dbGaP-controlled access, so the table cannot be redistributed. Build your own from any
allele-specific caller — ASCAT, Sequenza, FACETS, or a published pan-cancer segment table.

### `--genetic-map` for the Numbat export

`sptrace-export-numbat` annotates each SNP with its position in centimorgans. Use the map
that ships with [Eagle](https://alkesgroup.broadinstitute.org/Eagle/) v2.4.1,
`tables/genetic_map_hg38_withX.txt.gz`. Any table with chromosome, position and cM columns
works; column names are detected case-insensitively.

## Generating a runnable stand-in

To try the pipeline without either missing input:

```bash
sptrace-make-example-reference        # writes data/reference/example/
sptrace-simulate --design configs/design_example.toml --output out/example
```

This subsamples the real annotation above and **synthesises** the count matrix and the CNV
catalogue. It is intended for smoke tests, tutorials and CI. It is not a substitute for real
reference data in a benchmark: the count matrix has no biological covariance structure, and
the CNV segments come from a toy length/state distribution rather than from observed
tumours.
