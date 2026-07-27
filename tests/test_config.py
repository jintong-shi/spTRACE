"""Tests for design-file loading and path resolution."""

from pathlib import Path

import pytest

from st_cnv_simulator import _toml as toml
from st_cnv_simulator.cli.simulate import PATH_KEYS, load_design, resolve_paths


def write_design(tmp_path: Path, files: dict, name="design.toml") -> Path:
    path = tmp_path / name
    path.write_text(toml.dumps({"files": files}), encoding="utf-8")
    return path


def touch_all(directory: Path, names) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for n in names:
        (directory / n).write_text("x", encoding="utf-8")


def test_relative_paths_resolve_against_the_design_file(tmp_path):
    refs = tmp_path / "refs"
    touch_all(refs, [f"{k}.csv" for k in PATH_KEYS])
    design = write_design(tmp_path, {k: f"refs/{k}.csv" for k in PATH_KEYS})

    resolved = load_design(design)
    for k in PATH_KEYS:
        assert Path(resolved["files"][k]).is_absolute()
        assert Path(resolved["files"][k]) == refs / f"{k}.csv"


def test_design_file_can_be_loaded_from_any_cwd(tmp_path, monkeypatch):
    refs = tmp_path / "refs"
    touch_all(refs, [f"{k}.csv" for k in PATH_KEYS])
    design = write_design(tmp_path, {k: f"refs/{k}.csv" for k in PATH_KEYS})

    monkeypatch.chdir(tmp_path.parent)
    resolved = load_design(design)
    assert Path(resolved["files"]["gene_table_file"]).exists()


def test_absolute_paths_are_left_alone(tmp_path):
    refs = tmp_path / "refs"
    touch_all(refs, [f"{k}.csv" for k in PATH_KEYS])
    design = write_design(tmp_path, {k: str(refs / f"{k}.csv") for k in PATH_KEYS})

    resolved = load_design(design)
    assert Path(resolved["files"]["snp_file"]) == refs / "snp_file.csv"


def test_environment_variables_are_expanded(tmp_path, monkeypatch):
    refs = tmp_path / "refs"
    touch_all(refs, [f"{k}.csv" for k in PATH_KEYS])
    monkeypatch.setenv("STCNV_TEST_REFS", str(refs))
    design = write_design(tmp_path, {k: f"$STCNV_TEST_REFS/{k}.csv" for k in PATH_KEYS})

    resolved = load_design(design)
    assert Path(resolved["files"]["chr_lengths_file"]) == refs / "chr_lengths_file.csv"


def test_legacy_tcga_all_seg_key_is_accepted(tmp_path):
    refs = tmp_path / "refs"
    names = [f"{k}.csv" for k in PATH_KEYS if k != "cnv_catalog_file"] + ["seg.csv"]
    touch_all(refs, names)

    files = {k: f"refs/{k}.csv" for k in PATH_KEYS if k != "cnv_catalog_file"}
    files["tcga_all_seg"] = "refs/seg.csv"
    design = write_design(tmp_path, files)

    resolved = load_design(design)
    assert Path(resolved["files"]["cnv_catalog_file"]) == refs / "seg.csv"
    assert "tcga_all_seg" not in resolved["files"]


def test_new_key_wins_over_legacy_key(tmp_path):
    refs = tmp_path / "refs"
    touch_all(refs, [f"{k}.csv" for k in PATH_KEYS] + ["old.csv"])
    files = {k: f"refs/{k}.csv" for k in PATH_KEYS}
    files["tcga_all_seg"] = "refs/old.csv"
    design = write_design(tmp_path, files)

    resolved = load_design(design)
    assert Path(resolved["files"]["cnv_catalog_file"]).name == "cnv_catalog_file.csv"


def test_missing_reference_raises_with_all_offending_paths(tmp_path):
    refs = tmp_path / "refs"
    present = [k for k in PATH_KEYS if k != "refexp_file"]
    touch_all(refs, [f"{k}.csv" for k in present])
    design = write_design(tmp_path, {k: f"refs/{k}.csv" for k in PATH_KEYS})

    with pytest.raises(FileNotFoundError) as exc:
        load_design(design)
    message = str(exc.value)
    assert "refexp_file" in message
    assert "stcnv-make-example-reference" in message


def test_resolve_paths_tolerates_absent_optional_keys(tmp_path):
    design = {"files": {"normal_original_celltype": "Hepatocyte"}}
    out = resolve_paths(design, tmp_path / "design.toml")
    assert out["files"]["normal_original_celltype"] == "Hepatocyte"


def test_shipped_designs_declare_every_path_key(repo_root):
    for name in ("design.toml", "design_example.toml"):
        design = toml.load(repo_root / "configs" / name)
        assert set(PATH_KEYS) <= set(design["files"]), f"{name} is missing path keys"
