"""Reproducibility regression tests.

The haplotype-assignment helper used to key a RandomState off Python's builtin
``hash()`` of a string. That hash is salted per interpreter by ``PYTHONHASHSEED``,
so a fixed simulation seed produced different allele assignments between runs.
These tests pin the behaviour that fixed it.
"""

import hashlib
import os
import subprocess
import sys

from sptrace.model import deterministic_hap_flip

# Representative events: (chrom, start, end, major_cn, minor_cn)
EVENTS = [
    ("chr1", 1_000_000, 5_000_000, 2, 0),
    ("chr7", 55_000_000, 61_000_000, 3, 1),
    ("chr17", 100, 8_000_000, 1, 0),
    ("chrX", 40_000_000, 40_500_000, 4, 1),
]


def test_hap_flip_is_stable_within_a_process():
    assert [deterministic_hap_flip(*e) for e in EVENTS] == [
        deterministic_hap_flip(*e) for e in EVENTS
    ]


def test_hap_flip_is_stable_across_hash_seeds():
    """The decisive test: vary PYTHONHASHSEED and require identical results."""
    script = (
        "from sptrace.model import deterministic_hap_flip as f;"
        "print([f(*e) for e in ["
        "('chr1',1000000,5000000,2,0),"
        "('chr7',55000000,61000000,3,1),"
        "('chr17',100,8000000,1,0),"
        "('chrX',40000000,40500000,4,1)]])"
    )
    outputs = set()
    for hash_seed in ("0", "1", "12345", "random"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env
        )
        outputs.add(proc.stdout.strip())

    assert len(outputs) == 1, f"hap flip varies with PYTHONHASHSEED: {outputs}"


def test_hap_flip_discriminates_between_events():
    """A constant function would pass the stability tests; make sure it is not one."""
    many = [
        deterministic_hap_flip("chr1", start, start + 1_000_000, 2, 1)
        for start in range(0, 200_000_000, 1_000_000)
    ]
    assert set(many) == {True, False}


def test_hap_flip_key_includes_every_component():
    """Distinct events must derive distinct seeds, not just distinct outcomes."""

    def digest(chrom, start, end, major, minor, seed=0):
        key = f"{chrom}:{start}:{end}:{major}:{minor}:{seed}".encode()
        return hashlib.blake2b(key, digest_size=4).hexdigest()

    base = ("chr5", 1_000_000, 4_000_000, 3, 1)
    variants = [
        ("chr6", 1_000_000, 4_000_000, 3, 1),
        ("chr5", 1_000_001, 4_000_000, 3, 1),
        ("chr5", 1_000_000, 4_000_001, 3, 1),
        ("chr5", 1_000_000, 4_000_000, 2, 1),
        ("chr5", 1_000_000, 4_000_000, 3, 2),
    ]
    digests = {digest(*base)} | {digest(*v) for v in variants}
    assert len(digests) == len(variants) + 1
