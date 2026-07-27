"""The local TOML helpers must read the shipped configs and round-trip them."""

import pytest

from st_cnv_simulator import _toml as toml


def test_loads_shipped_design(repo_root):
    design = toml.load(repo_root / "configs" / "design.toml")
    assert design["population"]["n_initial_cells"] == 2
    assert design["reproducibility"]["seed"] == 20260414
    assert "cnv_catalog_file" in design["files"]


def test_round_trip_preserves_values(repo_root):
    original = toml.load(repo_root / "configs" / "design.toml")
    assert toml.loads(toml.dumps(original)) == original


def test_round_trip_example_design(repo_root):
    original = toml.load(repo_root / "configs" / "design_example.toml")
    assert toml.loads(toml.dumps(original)) == original


def test_dumps_handles_nesting_and_types():
    data = {
        "top": 1,
        "flag": True,
        "ratio": 0.25,
        "name": "hello",
        "items": [1, 2, 3],
        "section": {"a": 1, "nested": {"b": "two"}},
    }
    assert toml.loads(toml.dumps(data)) == data


def test_dumps_rejects_unsupported_type():
    with pytest.raises(TypeError, match="Cannot serialise"):
        toml.dumps({"bad": {1, 2, 3}})


def test_dumps_rejects_non_dict():
    with pytest.raises(TypeError, match="must be a dict"):
        toml.dumps([1, 2, 3])


def test_module_does_not_shadow_pypi_toml():
    """Regression guard: the shim used to live at top level as `toml.py`."""
    import importlib.util

    spec = importlib.util.find_spec("st_cnv_simulator._toml")
    assert spec is not None
    assert spec.name == "st_cnv_simulator._toml"
