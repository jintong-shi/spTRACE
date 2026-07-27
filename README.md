# st-cnv-simulator

Agent-based simulator of spatial transcriptomics data with allele-specific copy-number
variation, for benchmarking spatial CNV inference methods.

The simulator grows a tumour cell population on a 2-D lattice; each division can draw a
new allele-specific CNV event from a real copy-number catalogue, so subclones accumulate
a genuine phylogeny. Normal (TME) cells are infused *during* growth rather than pasted in
afterwards, and are placed by cell-type-specific spatial weight maps, which yields a
realistic tumour/stroma interface and optional tertiary lymphoid structures. The final
tissue is rendered three ways:

| Layer | What you get |
|---|---|
| **Single cell** | per-cell counts, per-gene copy number, coordinates, cell type, clone ID |
| **Spot (ST)** | Visium-style spot counts with capture efficiency and overdispersion, per-spot cell composition |
| **Allele** | AD/DP matrices over SNPs, exportable to [Numbat](https://github.com/kharchenkolab/numbat) allele-count format |

Because the ground truth is known exactly — every event, every clone, every cell's
genotype — the output can score a CNV caller's segment boundaries, copy-number states,
clone assignments, and allelic imbalance.

## Install

```bash
git clone https://github.com/jintongshi/st-cnv-simulator.git
cd st-cnv-simulator
pip install -e ".[parquet]"        # drop [parquet] and pass --no-parquet to write TSV
```

Python 3.10 or newer. No GPU required.

## Quickstart

Two of the reference inputs cannot be redistributed (see
[data/reference/README.md](data/reference/README.md)), so generate a small stand-in first:

```bash
stcnv-make-example-reference
stcnv-simulate --design configs/design_example.toml --output out/example
```

That completes in well under a minute and produces ~2,500 cells over 100 spots. Inspect
`out/example/synthetic/`, then move to `configs/design.toml` with your own reference data
for a full-size run.

> The example reference contains a **synthetic** expression matrix and a **synthetic** CNV
> catalogue. It exists so the pipeline runs out of the box — do not benchmark against it.

## Commands

| Command | Purpose |
|---|---|
| `stcnv-simulate` | one simulation from one design file |
| `stcnv-batch` | many replicates, sampling parameters from `[batch_ranges]` |
| `stcnv-export-numbat` | AD/DP matrices → Numbat allele-count TSV |
| `stcnv-prep-snps` | map a SNP list onto genes to build `snp_file` |
| `stcnv-make-example-reference` | build the small runnable example reference set |

Every command takes `--help`. The Python API is a single entry point:

```python
import st_cnv_simulator
res = st_cnv_simulator.run(...)   # see src/st_cnv_simulator/pipeline.py for the signature
```

## Reproducibility

A seed fixes the entire run. Same seed, same machine, same versions → **byte-identical
output files**, which the test suite asserts across every table the simulator writes.

```bash
stcnv-simulate --design configs/design_example.toml --output out/a --seed 42
stcnv-simulate --design configs/design_example.toml --output out/b --seed 42
# out/a and out/b agree file for file
```

Set the seed in the design file (`[reproducibility].seed`) or with `--seed`, which wins.
`0` is a valid seed. With no seed at all the simulator prints a warning and the run is not
reproducible.

For batches, `[batch].seed` (or `stcnv-batch --seed`) seeds the parameter sampler, and each
replicate's own simulation seed is derived from it — so one number reproduces a whole batch.

Two caveats worth knowing:

- `--jobs > 1` parallelises expression generation with threads that share the global numpy
  RNG, which makes counts depend on thread scheduling. The default is `--jobs 1`, which is
  reproducible and, measured at 13.6k cells, also *faster* (85 s vs 95 s) — the worker is
  GIL-bound, so threading bought nothing.
- The simulator seeds numpy's global RNG rather than threading an explicit `Generator`
  through each call site. This is enough for run-to-run reproducibility, but it means you
  cannot run two independently-seeded simulations concurrently in one process. Use separate
  processes (which is what `stcnv-batch` does).

## Configuration

`configs/design.toml` is the annotated reference; every section is documented in
[docs/configuration.md](docs/configuration.md). Paths under `[files]` may be absolute or
relative to the design file itself, and `$VAR` and `~` are expanded — so a checkout works
wherever it is cloned, and configs stay portable between machines.

A missing reference file fails immediately with every offending path listed, rather than
part-way through a long run.

## Outputs

```
<output>/
├── event_log.tsv                    every CNV event: iteration, cell, clone, segment, haplotype
├── baf/                             AD.mtx, DP.mtx, snp_info.csv, barcodes.tsv
├── qc/                              all diagnostic figures, incl. subclone maps
└── synthetic/
    ├── genome.tsv                   gene → chromosome, start, end
    ├── lineage.tsv                  surviving tumour cells: child → parent
    ├── lineage_all.tsv              full ancestry, including cells that died
    ├── all_cell_ids.tsv             every cell ID ever created, with a survival flag
    ├── cluster_lineage.tsv          cluster-level ancestry matrix
    ├── single_cell_data/            sc-expression, sc-genome_profile, sc-meta_data
    └── spatial_data/                st-expression, st-genome_profile, st-meta,
                                     st-annotation, cluster_data, cell_by_spot
```

Column-by-column definitions are in [docs/outputs.md](docs/outputs.md).

## Batch runs

```bash
stcnv-batch --design configs/design.toml --genetic-map /path/to/genetic_map_hg38_withX.txt.gz
```

Each replicate draws its parameters from `[batch_ranges]`, runs in its own subprocess, and
lands in its own folder alongside the exact `config.toml` that produced it, a Numbat export,
and QC figures. `examples/run_slurm.sh` is a cluster template.

Omit `--genetic-map` to skip the Numbat export; the map ships with Eagle and is not
redistributable here.

## Development

```bash
pip install -e ".[dev]"
pytest -m "not slow"     # fast unit tests
pytest                   # adds real end-to-end simulations (~1 min)
ruff check .
```

## Citation

If this simulator supports a publication, please cite it — see [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
