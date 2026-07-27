from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def reference_dir(repo_root) -> Path:
    return repo_root / "data" / "reference"


@pytest.fixture(scope="session")
def example_reference(reference_dir) -> Path:
    """The generated example reference set, skipping if it has not been built."""
    example = reference_dir / "example"
    needed = ["counts.tsv", "cnv_catalog.csv", "gene_order.csv", "celltype.tsv"]
    if not all((example / f).exists() for f in needed):
        pytest.skip(
            "Example reference not generated. Run `stcnv-make-example-reference` first."
        )
    return example
