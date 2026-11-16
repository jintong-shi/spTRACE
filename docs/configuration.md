# Configuration reference

A design file is TOML. `configs/design.toml` is the annotated reference;
`configs/design_example.toml` is a small, fast variant. Every section other than `[files]`,
`[population]`, `[expression]`, `[spatial]`, `[initial_cnv]` and
`[normal_celltype_proportions]` is optional and falls back to the defaults listed here.

## `[files]`

Paths may be absolute or relative to the design file's own directory; `$VAR` and `~` are
expanded. A missing file aborts the run immediately, listing every offending path.

| Key | Purpose |
|---|---|
| `chr_lengths_file` | chromosome lengths, `chr,length` |
| `gene_table_file` | gene coordinates, `gene,chr,start,end` |
| `refexp_file` | reference count matrix, genes × cells, TSV |
| `refcelltype_file` | cell-type labels, TSV |
| `snp_file` | SNP-to-gene map, `snp_id,gene_id,chrom,pos` |
| `cnv_catalog_file` | allele-specific CNV catalogue events are drawn from |
| `normal_celltypes` | list of cell types to treat as normal |
| `normal_original_celltype` | the normal type the tumour derives from |

`cnv_catalog_file` was previously called `tcga_all_seg`; the old name is still accepted.
Formats are documented in [data/reference/README.md](../data/reference/README.md).

Only chr1–22, X and Y are carried through the simulation; other contigs are dropped from
both the gene table and the chromosome lengths.

## `[population]`

Tissue growth on a `domain_side_size` × `domain_side_size` lattice, one cell per site.

| Key | Default | Meaning |
|---|---|---|
| `n_initial_cells` | required | founder tumour clones, each with its own baseline expression |
| `domain_side_size` | 200 | lattice side length |
| `max_cells` | 1e4 | stop once this many cells exist |
| `max_iterations` | 500 | stop after this many iterations |
| `n_clusters` | 10 | clusters for the genotype clustering step |
| `background_fill_prob` | 0.75 | passed to the normal-cell fill as `fill_prob` |
| `fraction_drop` | 0.05 | fraction of cells dropped during the fill |
| `normal_action_rates` | – | `[stay, spawn, move, death]` for infused normal cells |
| `init_event_size` | – | optional fixed size for founder events |
| `initial_coordinates` | – | list of `{x, y}` tables to place founders explicitly |

Growth stops at whichever of `max_cells`, `max_iterations` or `[auto_iter].occupancy_target`
comes first.

### `[population.transitions]`

Two-state model per tumour cell: state 0 proliferative, state 1 mutator.

| Key | Meaning |
|---|---|
| `s0s1` | per-iteration probability of 0 → 1 |
| `s1s0` | per-iteration probability of 1 → 0 |

### `[population.rates_state_0]` / `[population.rates_state_1]`

Action propensities for a tumour cell in that state; normalised to a distribution before
sampling, so they need not sum to 1.

| Key | Meaning |
|---|---|
| `stay_rate` | do nothing |
| `spawn_rate` | divide into a free neighbouring site |
| `move_rate` | migrate to a free neighbouring site |
| `death_rate` | die and free its site |
| `mutation_rate` | probability of acquiring a new CNV event at division |

Typically state 0 has `mutation_rate = 0` and `death_rate = 0`, so mutation and death are
confined to state 1. Both sections must be present together, or neither.

## `[initial_cnv]`

CNV events carried by the founder clones before any evolution.

| Key | Default | Meaning |
|---|---|---|
| `n_shared` | 1 | events applied identically to every founder — the truncal CNVs |
| `n_private` | 2 | events private to each founder; an int, or one entry per founder |

## `[expression]`

| Key | Default | Meaning |
|---|---|---|
| `lognorm_params_tumor` | – | `[mu, sigma]` of the lognormal for a tumour cell's total count |
| `lognorm_params_normal` | – | the same for a normal cell |
| `dirichlet_precision` | – | Dirichlet concentration for gene-level counts; higher is less noisy |

Per cell: total count is drawn lognormal-then-Poisson, gene proportions from a Dirichlet
around the cell's baseline profile scaled by per-gene copy number, and counts are
multinomial given the total.

## `[spatial]`

| Key | Default | Meaning |
|---|---|---|
| `n_spots_x`, `n_spots_y` | 10, 10 | spot grid dimensions |
| `spot_radius` | 2.5 | capture radius in lattice units |
| `st_eff` | 0.2 | capture efficiency — fraction of transcripts observed |
| `st_theta` | 10 | negative-binomial dispersion of capture |

Spots are laid out on a regular grid over the lattice. A cell contributes to a spot when its
centre falls inside `spot_radius`; overlapping spots are possible, and cells outside every
spot contribute to none.

## `[drip]`

Normal cells are infused incrementally during growth rather than in one pass at the end,
which is what produces a realistic tumour/stroma interface.

| Key | Default | Meaning |
|---|---|---|
| `enable` | true | |
| `every` | 20 | infuse every N iterations |
| `per_cycle_max` | – | cap on cells added per cycle |
| `fill_ratio` | 0.05 | fraction of free sites to fill per cycle |
| `immune_tls_share` | 0.60 | share of TLS-eligible cells seeded inside a TLS |

## `[tls]`

Tertiary lymphoid structures: dense immune aggregates placed far from tumour by
farthest-point sampling.

| Key | Default | Meaning |
|---|---|---|
| `n_tls` | 2 | number of structures |
| `r_tls` | 8 | core radius |
| `r_block` | 2 | dilation of the core that tumour cells may not enter |
| `min_center_dist` | 15.0 | minimum separation between centres |
| `min_edge_gap` | 1.0 | minimum gap from the tumour edge |
| `border_margin` | 0 | keep-out margin at the lattice border |
| `suppress_sigma` | 3.0 | Gaussian blur radius of the soft suppression field |
| `tls_celltypes` | `["T","B","DC","Plasma","T_reg"]` | types allowed inside a TLS |
| `spawn_mod` | – | per tumour state, additive change to spawn near a TLS (negative lowers) |
| `death_mod` | – | per tumour state, additive change to death near a TLS (positive raises) |
| `tumor_block_spawn` / `tumor_block_death` | 0.0 | scalar legacy form of the two above |

## `[immune_soft]`

The same suppression, weaker and diffuse, around scattered immune cells rather than TLS.

| Key | Default | Meaning |
|---|---|---|
| `enable` | true | |
| `labels` | – | cell types that generate the field |
| `sigma` | 2.0 | blur radius of the density field |
| `spawn_mod`, `death_mod` | – | per-state modifiers, as in `[tls]` |
| `tumor_block_spawn` | 0.10 | scalar fallback |
| `tumor_block_death` | 0.05 | scalar fallback |

## `[weights]`

Spatial weight maps deciding where each cell-type group is placed, as a function of signed
distance from the tumour boundary (negative inside).

| Key | Default | Meaning |
|---|---|---|
| `mu_s` | 4.0 | stromal rim: distance from the tumour edge |
| `w_s` | 5.0 | stromal rim: width |
| `mu_i` | – | immune band centre; defaults from the placement mixture |
| `sigma_i` | 2.5 | immune band width outside the tumour |
| `delta_epi` | 2.0 | epithelial offset, further out than stroma |
| `g_tls` | 7.0 | TLS peak gain |

## `[immune_placement]`

Immune cells are placed from a three-way mixture: inside the tumour, at its margin, or
outside.

| Key | Default | Meaning |
|---|---|---|
| `inside_prob`, `edge_prob`, `outside_prob` | 0.33 / 0.34 / 0.33 | mixture weights |
| `edge_sigma` | 1.5 | width of the margin band |
| `inside_power` | 1.0 | decay exponent inside the tumour |
| `outside_sigma` | `sigma_i` | width of the outside band |
| `sample_mixture` | false | redraw the mixture from a Dirichlet each cycle |
| `dirichlet_alpha` | `probs × 10` | concentration when `sample_mixture` is on |

## `[perturb]`

Roughens the weight maps so replicates are not near-identical.

| Key | Default | Meaning |
|---|---|---|
| `weight_temperature` | 1.0 | sharpens (>1) or flattens (<1) the maps |
| `uniform_tail` | 0.0002 | uniform floor, so no site is impossible |
| `noise_gain`, `noise_sigma` | 0.02, 6.0 | amplitude and scale of low-frequency noise |
| `micro_blobs_per_group` | 0 | number of small hotspots per group |
| `micro_blob_radius`, `micro_blob_gain` | 0, 1.0 | their size and strength |
| `jitter_prob`, `jitter_radius` | 0.02, 1 | chance and range of displacing a placed cell |

## `[auto_iter]`

| Key | Default | Meaning |
|---|---|---|
| `enable` | true | stop early on occupancy |
| `occupancy_target` | 0.90 | target fraction of occupied sites |

If `max_iterations` runs out below target, a final top-up places normal cells to reach it.

## `[mutation_schedule]`

| Key | Default | Meaning |
|---|---|---|
| `enable_state1_age_decay` | – | decay the state-1 mutation rate with cell age |
| `half_life_iter` | – | iterations for the rate to halve |
| `min_factor` | – | floor on the decay multiplier |

With decay on, events concentrate early in growth, giving a few large clones rather than
uniformly scattered private events.

## `[celltype_groups]`

Maps each cell type to one of three placement groups — `Immune`, `Stromal`, `Epithelial` —
which select the weight map used. Values are matched case-insensitively and by prefix.

## `[normal_celltype_proportions]`

Relative weights for the composition of infused normal cells. Keys must be cell types
present in `refcelltype_file`. Values need not sum to 1.

## `[reproducibility]`

| Key | Meaning |
|---|---|
| `seed` | seeds numpy and Python RNGs. `0` is valid. Omit for an unseeded run (warned about). |

`--seed` on the command line overrides this. See the README's reproducibility section for
the guarantees and the two caveats.

## `[batch]` and `[batch_ranges]`

Read only by `sptrace-batch`; ignored by `sptrace-simulate`.

`[batch]`:

| Key | Meaning |
|---|---|
| `replicates` | number of replicates |
| `output_folder` | batch output directory |
| `seed` | seeds the parameter sampler, making the whole batch reproducible |

`[batch_ranges]`: each entry is a `[min, max]` pair sampled once per replicate. Integer
bounds give an integer draw, float bounds a float rounded to 3 decimals. Keys are dot-paths
into the sections above, e.g.

```toml
population.max_cells = [300000, 550000]
population.rates_state_1.spawn_rate = [0.45, 0.65]
tls.n_tls = [2, 3]
```

Two keys are special-cased because they feed `[initial_cnv]`:

| Key | Effect |
|---|---|
| `n_shared_count` | range for `initial_cnv.n_shared` |
| `n_private_count` | range for each entry of `initial_cnv.n_private`, drawn once per founder |

`normal_celltype_proportions` is additionally perturbed per replicate by a Dirichlet
around the base values. Each replicate's own simulation seed is drawn from the batch
sampler, so it lands in the replicate's saved `config.toml` and the run can be reproduced
on its own.
