# Contributing

Thanks for your interest. Issues and pull requests are welcome.

## Setup

```bash
git clone https://github.com/jintongshi/st-cnv-simulator.git
cd st-cnv-simulator
pip install -e ".[dev,parquet]"
stcnv-make-example-reference     # needed by the end-to-end tests
```

## Tests

```bash
pytest -m "not slow"     # unit tests, a few seconds
pytest                   # adds real end-to-end simulations, about a minute
ruff check .
```

CI runs the tests on Python 3.10, 3.11 and 3.12, plus `ruff check` and a dedicated
same-seed reproducibility job.

`ruff format` is **not** enforced repo-wide: applying it would rewrite the layout of the
numerical core, making this release hard to diff against the internal version it came from.
Format the files you touch (`ruff format <paths>`) rather than the whole tree.

## Things to know before changing the simulation

**Reproducibility is a hard requirement.** A fixed seed must produce byte-identical output.
`tests/test_smoke.py::test_same_seed_reproduces_every_output` enforces this and will catch a
regression, but it is easy to introduce one:

- The simulator seeds numpy's **global** RNG. Any new randomness must come from
  `np.random.*` on that global stream, drawn in a deterministic order.
- Do not introduce threads or processes that draw from the global RNG. Expression
  generation is single-threaded for exactly this reason — see `--jobs` and the note in
  `formatting.assemble_tissue_data`.
- Do not key any RNG off Python's builtin `hash()` of a string. It is salted per
  interpreter by `PYTHONHASHSEED`. `deterministic_hap_flip` uses blake2b;
  `tests/test_determinism.py` guards this.
- Do not rely on set iteration order for anything that reaches the output.

If a change legitimately alters simulation output, say so explicitly in the pull request and
add a CHANGELOG entry — downstream benchmarks may be pinned to a version.

## Style

- `ruff` for linting and formatting; line length 100.
- Comments and docstrings in English.
- Type hints on new public functions. The existing numerical core is only partly annotated;
  adding hints there is welcome but keep it separate from behaviour changes.
- The simulation core is dense numerical code. Match the surrounding style rather than
  reformatting neighbouring lines.

## Adding a configuration option

1. Read it in `pipeline.run` from the `config` dict with a documented default.
2. Add it to `configs/design.toml` with an inline comment.
3. Document it in `docs/configuration.md`.
4. Keep it optional — an existing design file must keep working unchanged.

## Adding an output

Document every column in `docs/outputs.md`, and extend `EXPECTED_OUTPUTS` in
`tests/test_smoke.py` so the reproducibility test covers the new file.

## Reporting a bug

Please include the design file (or the relevant sections), the exact command, the seed, the
package version, and the full traceback. For anything reproducibility-related, the output of
two same-seed runs is the most useful thing you can attach.