# Changelog

## 0.1.0 — first public release

Initial open-source release.

The simulation model is otherwise unchanged from the internal version. Comparing abstract
syntax trees with comments and docstrings stripped, `cnv_utils.py` and `cnv_logging.py` are
identical, and across `model.py`, `formatting.py`, `utils.py` and `pipeline.py` exactly four
definitions differ — each a deliberate fix listed below:

| Module | Definition changed | Why |
|---|---|---|
| `model.py` | `deterministic_hap_flip` | seeded from a process-salted `hash()` |
| `formatting.py` | `assemble_tissue_data` | `n_jobs` default 8 → 1 |
| `utils.py` | `plot_heatmap_by_cluster_and_chrom` | `mpl.cm.get_cmap` removed in matplotlib 3.9 |
| `pipeline.py` | `run` | gained an `n_jobs` parameter |

Import blocks also changed: `hashlib` added to `model.py`, a missing `utils` import added to
`formatting.py`, and five unused imports removed from `pipeline.py`.

### Packaging

- Installable package under `src/st_cnv_simulator/` with a `pyproject.toml`; console
  entry points `stcnv-simulate`, `stcnv-batch`, `stcnv-export-numbat`, `stcnv-prep-snps`
  and `stcnv-make-example-reference`.
- **Removed the top-level `toml.py`**, which shadowed the PyPI `toml` package for anything
  run from the repository root. TOML handling now lives in `st_cnv_simulator._toml` and
  reads via `tomllib` (3.11+) or `tomli` (3.10).
- Scripts became modules: `run.py` → `cli/simulate.py`, `batch_runner.py` → `cli/batch.py`,
  `export_numbat.py` and `prep_snps.py` → `cli/`, `main.py` → `pipeline.py`.
  `formating.py` was renamed to `formatting.py`.
- `batch.py` now invokes replicates as `python -m st_cnv_simulator.cli.simulate` instead of
  a bare `run.py` in the working directory, so batches work from any directory.
- Dropped an unused `from calicost import ...` import, removing a phantom dependency on
  CalicoST.
- Added `.gitignore`, MIT `LICENSE`, `CITATION.cff`, `CONTRIBUTING.md`, a GitHub Actions CI
  workflow, and a test suite (39 tests).

### Bug fixes

- **`genome.tsv` was mislabelled.** `extract_results` overwrote the index of the gene
  coordinate table with the alphabetical gene order while the table itself was in genomic
  order, so *every* row carried a gene name that did not own its coordinates. Rows are now
  reindexed by gene symbol and reordered to match the expression matrix columns, so label,
  coordinates and expression column agree. Any analysis that joined on `genome.tsv`'s index
  was reading the wrong coordinates.
- **A fixed seed did not reproduce a run.** Two causes, both fixed:
  - `deterministic_hap_flip` seeded a `RandomState` from Python's builtin `hash()` of a
    string. That hash is salted per interpreter by `PYTHONHASHSEED`, so which haplotype
    received an event's major copy number varied between processes. It now uses a blake2b
    digest. *This changes output relative to the internal version* — but the internal
    version had no stable output to preserve.
  - Expression generation ran eight threads sharing the global numpy RNG, making counts
    depend on thread scheduling. The default is now single-threaded, exposed as `--jobs`.
    Measured at 13,620 cells, single-threaded is also faster (85 s vs 95 s): the worker is
    GIL-bound, so the threading bought nothing.

  With both fixed, two runs at the same seed produce byte-identical output across every
  table the simulator writes, which the test suite asserts.
- **`mpl.cm.get_cmap` / `plt.cm.get_cmap`** were removed in matplotlib 3.9; three call
  sites would crash on any current install. Replaced with `mpl.colormaps[...]`.
- **Batch runs were not reproducible**: replicate seeds came from an unseeded
  `random.randint`. `[batch].seed` (or `--seed`) now seeds the sampler, so one number
  reproduces a whole batch including each replicate's own seed.
- **`seed = 0` silently disabled seeding**, because the guard was `if seed != 0`. Zero is
  now a valid seed, and a run with no seed at all warns.
- The temporary expression memmap under `<output>/tmp/` was never deleted, leaving scratch
  behind — over 16 GB on a full-size run. It is now removed once outputs are written.
- `prefilter_cnv_by_gene_mid` returning zero rows now raises with a diagnostic instead of
  letting the run continue with an empty catalogue.
- `export_numbat` no longer swallows every exception in a bare `except:`, and its
  `.A1`-based dense conversion was replaced with `np.asarray(...).ravel()`.
- `formatting.prepare_and_run_haplotype_matrix` raised `NameError: name 'utils' is not
  defined` on any call — the module never imported `utils`. It is not reached by the
  pipeline (which inlines the equivalent), so this was latent. Import added.

### Known lint debt

`ruff check` passes, but the numerical core is exempted from several rules via
`per-file-ignores` in `pyproject.toml`: legacy typing spellings (`Dict`, `Optional`),
around 20 unused imports, and multiple statements per line. Two of those unused imports
(`statsmodels`, several `scipy.stats` functions) are the only reason those packages are
declared dependencies — dropping them would lighten the install.

`ruff format` is likewise not enforced repo-wide. Both were left alone to keep the release
free of behaviour-neutral churn through the simulation code, so it stays diffable against
the internal version. New code is expected to be clean.

### Portability

- **Removed all 13 hardcoded absolute paths.** `[files]` entries may be absolute or
  relative to the design file, with `$VAR` and `~` expansion.
- The Eagle genetic map path is now the required `--genetic-map` argument to
  `stcnv-export-numbat` rather than a module-level constant.
- `prep_snps.py`'s three hardcoded paths became `--snp-ids`, `--gene-table` and `--output`;
  it now accepts a text/CSV SNP list as well as `.npy`.
- The site-specific SLURM script was replaced with a generic template in `examples/`.
- Missing reference files are reported up front, all at once, with a pointer to
  `stcnv-make-example-reference`.

### Outputs no longer written

- **`res.pkl` is gone.** It reached ~10 GB per replicate (measured on a 28k-cell run), which
  dominated the on-disk size of a batch, and pickles are version-fragile besides. Everything
  downstream code read from it is available in the TSV/Parquet tables. `stcnv-batch`'s QC
  step was rewired to load those tables via a new `load_replicate_tables()` instead of
  unpickling. `st_cnv_simulator.run()` still returns the full result object in memory, so a
  Python caller can serialise whatever it needs.

  `synthetic/lineage_all.tsv` is still written. It is only ~1 MB, and dropping it turned out
  to cost correctness: `lineage.tsv` covers surviving tumour cells only, so an ancestry walk
  over it breaks at the first cell that died. Measured across eight replicates, that
  understated the evolved-subclone count in six of them (3 major founders became 1 in one
  case). `stcnv-batch` now prefers `lineage_all.tsv` and reports
  `n_broken_lineage_links` / `subclone_counts_exact` if it ever has to fall back, so
  `n_subclones` can never be silently understated.

### QC available from a single run

- The subclone figures (`subclone_spatial.png`, `cnv_events_spatial.png`,
  `st_subclone_count_spatial.png`, `st_major_subclone_spatial.png`,
  `celltype_proportions.png`, `sc_celltype_spatial.png`) used to be produced **only** by the
  batch runner, so a plain single simulation never got them. The QC analysis and plotting
  moved out of `cli/batch.py` into a new library module `st_cnv_simulator.qc`, and
  `stcnv-simulate` now runs it too. A single run and a batch replicate produce the same ten
  figures, and the run prints a one-line subclone summary.

  `st_cnv_simulator.qc.analyze_replicate()` and `generate_qc_plots()` are importable, so QC
  can also be re-run over an existing output directory without re-simulating.

  Two consequences. `stcnv-batch` no longer runs the QC itself (it would have run twice per
  replicate); figure titles now carry the replicate folder name rather than `Rep_<i>`. And
  `--skip-visual` now suppresses the subclone figures too, where before the batch runner
  drew them regardless.

### Figures consolidated

- `synthetic/visual/` is gone. Its two figures moved into `qc/` and were renamed to say
  what they plot: `tissue.png` → `cnv_burden_spatial.png`, `tissue_w_array.png` →
  `cnv_burden_with_spot_array.png`. There is now one figure directory instead of two, and
  in batch runs `qc/cnv_events_spatial.png` no longer sits in a different folder from the
  near-identical `visual/tissue.png`. `--skip-visual` still suppresses all of them.

### Behaviour changes to be aware of

- `STCNV_NO_PARQUET` and `STCNV_SKIP_VISUAL` environment variables were replaced by
  `--no-parquet` and `--skip-visual`.
- `extract_results` used to `rmtree` an existing `<output>/synthetic` without asking. It now
  fails unless `--overwrite` is passed. `stcnv-batch` passes it automatically.
- `[files].genome_file` was removed from the shipped config: nothing read it.
- `[files].tcga_all_seg` is now `cnv_catalog_file`; the old name still works.

### Documentation

- README, a full configuration reference (`docs/configuration.md`) and an output schema
  reference (`docs/outputs.md`) documenting every column of every file.
- `data/reference/README.md` describes all six reference inputs, which two cannot be
  redistributed and why, and how to build your own.
- Roughly 380 lines of Chinese comments and docstrings were translated to English.
