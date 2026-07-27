"""End-to-end smoke test on the small example reference set.

Marked ``slow``: it runs a real (tiny) simulation, which takes tens of seconds.
Deselect with ``pytest -m "not slow"``.
"""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow

EXPECTED_OUTPUTS = [
    "synthetic/event_log.tsv",
    "synthetic/genome.tsv",
    "synthetic/lineage.tsv",
    "synthetic/lineage_all.tsv",
    "synthetic/all_cell_ids.tsv",
    "synthetic/cluster_lineage.tsv",
    "synthetic/single_cell_data/sc-expression.tsv",
    "synthetic/single_cell_data/sc-genome_profile.tsv",
    "synthetic/single_cell_data/sc-meta_data.tsv",
    "synthetic/spatial_data/st-expression.tsv",
    "synthetic/spatial_data/st-genome_profile.tsv",
    "synthetic/spatial_data/st-meta.tsv",
    "synthetic/spatial_data/st-annotation.tsv",
    "synthetic/spatial_data/cell_by_spot.txt",
    "synthetic/baf/AD.mtx",
    "synthetic/baf/DP.mtx",
    "synthetic/baf/snp_info.csv",
    "synthetic/baf/barcodes.tsv",
]


def run_simulation(repo_root, out_dir, seed):
    cmd = [
        sys.executable, "-m", "st_cnv_simulator.cli.simulate",
        "--design", str(repo_root / "configs" / "design_example.toml"),
        "--output", str(out_dir),
        "--seed", str(seed),
        "--skip-visual",
        "--no-parquet",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
    assert proc.returncode == 0, f"simulation failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return proc


@pytest.fixture(scope="module")
def completed_run(repo_root, example_reference, tmp_path_factory):
    out = tmp_path_factory.mktemp("run_a")
    run_simulation(repo_root, out, seed=42)
    return out


def test_all_expected_outputs_exist(completed_run):
    missing = [p for p in EXPECTED_OUTPUTS if not (completed_run / p).exists()]
    assert not missing, f"missing outputs: {missing}"


def test_scratch_memmap_is_cleaned_up(completed_run):
    assert not (completed_run / "tmp").exists()


def test_output_root_holds_only_data_and_figures(completed_run):
    """Everything simulated goes under synthetic/; every figure under qc/.

    ``completed_run`` passes --skip-visual, so qc/ is absent here; the point is
    that nothing *else* is left at the top level.
    """
    entries = {p.name for p in completed_run.iterdir()}
    assert entries <= {"qc", "synthetic"}, f"unexpected entries at the output root: {entries}"
    assert "synthetic" in entries
    assert (completed_run / "synthetic" / "baf").is_dir()
    assert (completed_run / "synthetic" / "event_log.tsv").is_file()


def test_all_figures_live_in_one_place(repo_root, example_reference, tmp_path):
    """Figures go to qc/ only; the old synthetic/visual/ directory is gone."""
    out = tmp_path / "figs"
    cmd = [
        sys.executable, "-m", "st_cnv_simulator.cli.simulate",
        "--design", str(repo_root / "configs" / "design_example.toml"),
        "--output", str(out), "--seed", "42", "--no-parquet",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
    assert proc.returncode == 0, proc.stderr[-2000:]

    assert not (out / "synthetic" / "visual").exists()
    figures = sorted(p.name for p in out.rglob("*.png")) + sorted(
        p.name for p in out.rglob("*.pdf")
    )
    # A single run gets the same figure set as a batch replicate, including the
    # subclone views, which used to be produced only by stcnv-batch.
    assert figures == [
        "celltype_proportions.png",
        "cnv_burden_spatial.png",
        "cnv_burden_with_spot_array.png",
        "cnv_events_spatial.png",
        "sc_celltype_spatial.png",
        "st_major_subclone_spatial.png",
        "st_subclone_count_spatial.png",
        "subclone_spatial.png",
        "sc_celltype_spatial.pdf",
        "st_celltype_proportions.pdf",
    ]
    assert all(p.parent.name == "qc" for p in out.rglob("*.png"))
    assert all(p.parent.name == "qc" for p in out.rglob("*.pdf"))


def test_default_run_writes_parquet(repo_root, example_reference, tmp_path):
    """The default output format is Parquet, so pyarrow is a required dependency.

    Every other test passes --no-parquet, which is how pyarrow being declared an
    optional extra went unnoticed: a plain install could not complete a default run.
    """
    import pandas as pd

    out = tmp_path / "pq"
    cmd = [
        sys.executable, "-m", "st_cnv_simulator.cli.simulate",
        "--design", str(repo_root / "configs" / "design_example.toml"),
        "--output", str(out), "--seed", "42", "--skip-visual",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
    assert proc.returncode == 0, f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"

    expr = out / "synthetic" / "single_cell_data" / "sc-expression.parquet"
    assert expr.is_file()
    assert not expr.with_suffix(".tsv").exists()
    assert len(pd.read_parquet(expr)) > 0


def test_pickle_is_not_written(completed_run):
    """res.pkl reached ~10 GB per replicate and is no longer emitted."""
    assert not (completed_run / "res.pkl").exists()


def test_full_ancestry_is_written_and_covers_dead_cells(completed_run):
    """lineage_all.tsv must be a strict superset of lineage.tsv.

    Without it, walking a founder to its surviving descendants stops at the first
    cell that died, which understates subclone counts.
    """
    import pandas as pd

    all_lin = pd.read_csv(completed_run / "synthetic/lineage_all.tsv", sep="\t", index_col=0)
    surviving = pd.read_csv(completed_run / "synthetic/lineage.tsv", sep="\t", index_col=0)

    assert len(all_lin) >= len(surviving)
    assert set(surviving["child_id"]) <= set(all_lin["child_id"])
    # Founders are encoded as -1 here, not as the 'initial_seed' label lineage.tsv uses.
    assert pd.to_numeric(all_lin["parent_id"], errors="coerce").notna().all()

    # Every numeric parent must itself be present, so no chain can break.
    children = set(all_lin["child_id"].astype(int))
    parents = all_lin["parent_id"].astype(int)
    dangling = [p for p in parents if p >= 0 and p not in children]
    assert not dangling, f"{len(dangling)} ancestry links point at an absent cell"


def test_single_cell_and_spot_tables_are_consistent(completed_run):
    import pandas as pd

    meta = pd.read_csv(completed_run / "synthetic/single_cell_data/sc-meta_data.tsv", sep="\t")
    expr = pd.read_csv(
        completed_run / "synthetic/single_cell_data/sc-expression.tsv", sep="\t", nrows=5
    )
    assert len(meta) > 0
    # sc-expression is written cells x genes with the cell name in the first column.
    assert expr.columns[0] in {"index", "Unnamed: 0"}

    st_meta = pd.read_csv(completed_run / "synthetic/spatial_data/st-meta.tsv", sep="\t")
    assert {"x", "y", "cell_count"} <= set(st_meta.columns)
    assert (st_meta["cell_count"] >= 0).all()

    annotation = pd.read_csv(completed_run / "synthetic/spatial_data/st-annotation.tsv", sep="\t")
    assert set(annotation["annotation"].dropna()) <= {"benign", "aberrant"}


def test_genome_table_labels_match_its_coordinates(completed_run, example_reference):
    """Regression: the index used to be overwritten with an unrelated gene order.

    genome.tsv comes back from the simulator in genomic order while the expression
    matrix is alphabetical. Relabelling the index instead of reordering the rows
    gave every row a gene name that did not own those coordinates.
    """
    import pandas as pd

    genome = pd.read_csv(completed_run / "synthetic/genome.tsv", sep="\t", index_col=0)
    assert (genome.index == genome["gene"]).all(), "genome.tsv index disagrees with its gene column"

    # Coordinates must match the annotation the run was given.
    source = pd.read_csv(example_reference / "gene_order.csv").drop_duplicates(
        subset=["gene"], keep="first"
    ).set_index("gene")
    shared = genome.index.intersection(source.index)
    assert len(shared) == len(genome)
    assert (genome.loc[shared, "start"].astype(int) == source.loc[shared, "start"].astype(int)).all()
    assert (genome.loc[shared, "chr"] == source.loc[shared, "chr"]).all()


def test_genome_table_row_order_matches_expression_columns(completed_run):
    """genome.tsv row i must describe the gene in expression column i."""
    import pandas as pd

    genome = pd.read_csv(completed_run / "synthetic/genome.tsv", sep="\t", index_col=0)
    header = pd.read_csv(
        completed_run / "synthetic/single_cell_data/sc-expression.tsv", sep="\t", nrows=0
    )
    assert list(genome.index) == list(header.columns[1:])


def test_event_log_records_cnv_events(completed_run):
    import pandas as pd

    log = pd.read_csv(completed_run / "synthetic/event_log.tsv", sep="\t")
    assert len(log) > 0
    assert {"iter", "cell_id", "Chromosome", "Major_Copy_Number", "Minor_Copy_Number"} <= set(
        log.columns
    )
    assert (log["Major_Copy_Number"] >= log["Minor_Copy_Number"]).all()


def test_same_seed_reproduces_every_output(repo_root, example_reference, tmp_path_factory):
    """The property that makes the simulator usable as a benchmark."""
    import hashlib

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    out_a = tmp_path_factory.mktemp("rep_a")
    out_b = tmp_path_factory.mktemp("rep_b")
    run_simulation(repo_root, out_a, seed=99)
    run_simulation(repo_root, out_b, seed=99)

    compared = 0
    for rel in EXPECTED_OUTPUTS:
        a, b = out_a / rel, out_b / rel
        if a.exists() and b.exists():
            assert digest(a) == digest(b), f"{rel} differs between identical-seed runs"
            compared += 1
    assert compared >= 15


def test_different_seeds_produce_different_data(repo_root, example_reference, tmp_path_factory):
    out = tmp_path_factory.mktemp("rep_c")
    run_simulation(repo_root, out, seed=1234)
    a = (out / "synthetic/single_cell_data/sc-expression.tsv").read_bytes()

    other = tmp_path_factory.mktemp("rep_d")
    run_simulation(repo_root, other, seed=4321)
    b = (other / "synthetic/single_cell_data/sc-expression.tsv").read_bytes()

    assert a != b


def test_rerun_without_overwrite_fails_cleanly(repo_root, completed_run):
    cmd = [
        sys.executable, "-m", "st_cnv_simulator.cli.simulate",
        "--design", str(repo_root / "configs" / "design_example.toml"),
        "--output", str(completed_run),
        "--seed", "42", "--skip-visual", "--no-parquet",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
    assert proc.returncode != 0
    assert "already exists" in proc.stdout + proc.stderr


def test_quick_purity_mode(repo_root, example_reference, tmp_path):
    import pandas as pd

    out = tmp_path / "quick"
    cmd = [
        sys.executable, "-m", "st_cnv_simulator.cli.simulate",
        "--design", str(repo_root / "configs" / "design_example.toml"),
        "--output", str(out),
        "--seed", "42", "--quick-purity-only", "--skip-visual",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
    assert proc.returncode == 0, proc.stderr[-2000:]

    summary = pd.read_csv(out / "quick_purity_summary.tsv", sep="\t")
    assert 0.0 <= summary["overall_tumor_purity"].iloc[0] <= 1.0
    assert summary["n_tumor_cells"].iloc[0] <= summary["n_cells"].iloc[0]
    # Quick mode must not have produced the expensive layers.
    assert not (out / "synthetic" / "single_cell_data").exists()
    assert not (out / "synthetic" / "spatial_data").exists()
    assert not (out / "synthetic" / "baf").exists()
