"""spTRACE: agent-based simulation of spatial transcriptomics data with CNVs.

The simulator grows a tumour cell population on a 2-D lattice, mutates each
cell's diploid genome by drawing allele-specific copy-number events from a
real CNV catalogue, infuses normal (TME) cells, then renders the resulting
tissue as single-cell expression, spot-level spatial transcriptomics, and
allele (AD/DP) count matrices.

Typical use is through the command line:

    sptrace-simulate --design configs/design.toml --output out/
    sptrace-batch    --design configs/design.toml

The Python API entry point is :func:`sptrace.run`.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .pipeline import run

__all__ = ["run", "__version__"]
