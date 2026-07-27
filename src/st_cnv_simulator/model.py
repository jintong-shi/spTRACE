from __future__ import annotations

import copy
import hashlib
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from . import utils as ut
import pickle, gzip
import os
from .cnv_utils import sample_uniform_event
from .cnv_logging import EventLogger
from scipy.ndimage import distance_transform_edt, binary_dilation, gaussian_filter

def _signed_distance(mask: np.ndarray) -> np.ndarray:
    """Signed distance field: negative inside the tumour, positive outside."""
    out = ~mask
    d_out = distance_transform_edt(out)
    d_in  = distance_transform_edt(mask)
    return d_out - d_in

def _disk_mask(shape, center, radius):
    rr, cc = np.ogrid[:shape[0], :shape[1]]
    return (rr - center[0])**2 + (cc - center[1])**2 <= radius**2

def _sample_positions_by_weight(W: np.ndarray, n: int):
    """Sample n points from a weight map without replacement; W<=0 is never selected."""
    W = np.clip(W.astype(float), 0, None)
    s = W.sum()
    if s <= 0 or n <= 0:
        return np.empty((0, 2), dtype=int)
    idx = np.flatnonzero(W > 0)
    p   = W.flat[idx] / s
    choose = np.random.choice(idx, size=min(n, len(idx)), replace=False, p=p)
    return np.column_stack(np.unravel_index(choose, W.shape))  # (k,2)

def _normalize_groups_map(celltype_groups_map: dict) -> dict:
    """
    Normalise design.toml's [celltype_groups] to {'t': 'Immune', ...}: keys lower-cased, values collapsed to the three canonical group names.
    Values may be written with any capitalisation or prefix of immune/stromal/epithelial.
    """
    norm = {}
    for k, v in (celltype_groups_map or {}).items():
        key = (k or "").strip().lower()
        val = (v or "").strip().lower()
        if val.startswith("immu"):
            G = "Immune"
        elif val.startswith("stro"):
            G = "Stromal"
        elif val.startswith("epi"):
            G = "Epithelial"
        else:
            G = "Stromal"
        if key:
            norm[key] = G
    return norm

def _alloc_by_proportions(total: int, weights: dict) -> dict:
    """Split `total` into integers according to a weight dict (largest-remainder method; the parts sum to total)."""
    items = [(k, float(w)) for k, w in (weights or {}).items() if float(w) > 0]
    if total <= 0 or not items:
        return {k: 0 for k, _ in items}
    S = sum(w for _, w in items)
    if S <= 0:
        return {items[0][0]: total}
    raw = [(k, total * w / S) for k, w in items]
    flo = {k: int(np.floor(x)) for k, x in raw}
    rem = total - sum(flo.values())
    frac = sorted(((k, x - flo[k]) for k, x in raw), key=lambda kv: kv[1], reverse=True)
    for i in range(rem):
        flo[frac[i % len(frac)][0]] += 1
    return flo

def _jitter_place(x, y, cell_map, radius=1, attempts=6, rng=None):
    if radius <= 0:
        return x, y
    rng = rng or np.random
    H, Wd = cell_map.shape
    for _ in range(int(attempts)):
        # Support both the Generator and the legacy RandomState interface
        dx = rng.integers(-radius, radius + 1) if hasattr(rng, "integers") else rng.randint(-radius, radius + 1)
        dy = rng.integers(-radius, radius + 1) if hasattr(rng, "integers") else rng.randint(-radius, radius + 1)
        nx = int(x + dx)
        ny = int(y + dy)
        if nx < 0 or ny < 0 or nx >= H or ny >= Wd:
            continue
        if cell_map[nx, ny] != 0:
            continue
        return nx, ny
    return x, y

def _drip_seed_one_type(population, cell_map, label_name: str, W: np.ndarray, n_new: int,
                        genome, counter_like=None,
                        normal_action_rates=(0.93,0.01,0.05,0.01),
                        jitter_prob: float = 0.0,
                        jitter_radius: int = 1):
    """Seed n_new normal cells from a weight map (label=label_name, clone_id=None)."""
    if n_new <= 0:
        return population, cell_map
    free = (cell_map == 0)
    pos = _sample_positions_by_weight(W * free, n_new)
    zero_mut_rate = np.zeros(2, dtype=float)
    rng = np.random
    for x, y in pos:
        # jitter after selection
        if jitter_prob > 0.0 and (rng.random() < float(jitter_prob)):
            x, y = _jitter_place(int(x), int(y), cell_map, radius=int(jitter_radius), rng=rng)
        kwargs = dict(
            x=int(x), y=int(y),
            genome=genome.create_view() if hasattr(genome, "create_view") else copy.deepcopy(genome),
            action_rates=np.repeat(np.array(normal_action_rates).reshape(4,1), 2, axis=1),
            mutation_rate=zero_mut_rate,
            clone_id=None,                 # no clone_id means a normal cell
            cell_type=label_name,
        )
        if counter_like is not None:
            kwargs["counter"] = counter_like
        population.append(Cell(**kwargs))
        cell_map[int(x), int(y)] = 1
    return population, cell_map

def indices_in_window(genome, chrom: str, start: int, end: int):
    pos = genome.pos[chrom]
    return np.where((pos >= start) & (pos <= end))[0]

def deterministic_hap_flip(chrom, start, end, major_cn, minor_cn, seed=0):
    """Decide, reproducibly, whether an event's major CN goes to haplotype A or B.

    The digest must be stable across processes, so this uses blake2b rather than
    Python's built-in hash(), which is salted per interpreter by PYTHONHASHSEED.
    """
    key = f"{chrom}:{start}:{end}:{major_cn}:{minor_cn}:{seed}".encode()
    digest = hashlib.blake2b(key, digest_size=4).digest()
    rng = np.random.RandomState(int.from_bytes(digest, "big"))
    return rng.rand() < 0.5  # True means swap the two haplotypes once

def apply_catalog_event_to_genome(
    genome,
    chrom_base: str, start: int, end: int,
    major_cn: int, minor_cn: int,
    *,
    event_logger=None,
    cell=None,
    iter_t: int = None,
    source=None,          # passed through unchanged (catalog/random)
    context: str = None   # "init_shared" / "init_private" / "evolution"
):
    if context == "init_shared":
        flip = deterministic_hap_flip(chrom_base, start, end, major_cn, minor_cn)
    else:
        flip = (np.random.rand() < 0.5)

    if not flip:
        major_hap, minor_hap = chrom_base + "A", chrom_base + "B"
    else:
        major_hap, minor_hap = chrom_base + "B", chrom_base + "A"

    affected_gene_indices = set()

    # minor
    idx = genome.indices_in_window(minor_hap, int(start), int(end)) if hasattr(genome, "indices_in_window") else indices_in_window(genome, minor_hap, int(start), int(end))
    if len(idx) > 0:
        if hasattr(genome, "base"):
            affected_gene_indices.update(genome.base.gene_id[minor_hap][idx].tolist())
        else:
            affected_gene_indices.update(genome.gene_id[minor_hap][idx].tolist())
    if minor_cn == 0:
        if len(idx) > 0: genome.delete(minor_hap, idx)
    elif minor_cn > 1:
        for _ in range(int(minor_cn) - 1):
            idx = genome.indices_in_window(minor_hap, int(start), int(end)) if hasattr(genome, "indices_in_window") else indices_in_window(genome, minor_hap, int(start), int(end))
            if len(idx) > 0:
                if hasattr(genome, "base"):
                    affected_gene_indices.update(genome.base.gene_id[minor_hap][idx].tolist())
                else:
                    affected_gene_indices.update(genome.gene_id[minor_hap][idx].tolist())
                genome.add(minor_hap, idx)

    # major
    idx = genome.indices_in_window(major_hap, int(start), int(end)) if hasattr(genome, "indices_in_window") else indices_in_window(genome, major_hap, int(start), int(end))
    if len(idx) > 0:
        if hasattr(genome, "base"):
            affected_gene_indices.update(genome.base.gene_id[major_hap][idx].tolist())
        else:
            affected_gene_indices.update(genome.gene_id[major_hap][idx].tolist())
    if major_cn == 0:
        if len(idx) > 0: genome.delete(major_hap, idx)
    elif major_cn > 1:
        for _ in range(int(major_cn) - 1):
            idx = genome.indices_in_window(major_hap, int(start), int(end)) if hasattr(genome, "indices_in_window") else indices_in_window(genome, major_hap, int(start), int(end))
            if len(idx) > 0:
                if hasattr(genome, "base"):
                    affected_gene_indices.update(genome.base.gene_id[major_hap][idx].tolist())
                else:
                    affected_gene_indices.update(genome.gene_id[major_hap][idx].tolist())
                genome.add(major_hap, idx)

    if event_logger is not None and cell is not None:
        try:
            gene_names = [
                (genome.base.all_genes_list if hasattr(genome, "base") else genome.all_genes_list)[g]
                for g in sorted(affected_gene_indices)
            ]
        except Exception:
            gene_names = list(sorted(affected_gene_indices))
        length_mb = (int(end) - int(start) + 1) / 1e6
        row_id = -1
        if isinstance(source, (dict, pd.Series)) and "row_id" in source:
            row_id = int(source["row_id"])

        event_logger.log({
            "iter": (int(iter_t) if iter_t is not None else -1),
            "cell_id": getattr(cell, "id", None),
            "parent_id": getattr(cell, "parent", None),
            "clone_id": getattr(cell, "clone_id", None),
            "cell_type": getattr(cell, "cell_type", None),
            "source": "catalog" if source is not None else "random",
            "context": (context or ("evolution" if iter_t not in (None, 0) else "init_private")),
            "row_id": row_id,
            "Chromosome": chrom_base,
            "Start": int(start),
            "End": int(end),
            "Major_Copy_Number": int(major_cn),
            "Minor_Copy_Number": int(minor_cn),
            "length_mb": length_mb,
            "hap_assigned": f"{major_hap}:{int(major_cn)};{minor_hap}:{int(minor_cn)}",
            "affected_genes": gene_names
        })

class Genome:
    def __init__(
        self,
        gene_order,
        sigma: int = 2,
        expr_ub: int = 1000,
        expr_lb: int = 100,
        min_event_size: int = 5e5, #500 kb
        max_event_size: int = 5e6 #5mb
    ):
        """
        Each chromosome carries two copies, A and B.
        
        Parameters
        ----------
        sigma : int
            Reserved: random size distribution for amplification/deletion events (e.g. a truncated normal).
        expr_ub, expr_lb : int
            Lower/upper bounds for expression simulation (optional feature).
        min_event_size, max_event_size : float
            Minimum and maximum width of an amplification/deletion event (e.g. 1 kb to 5 Mb).
        
        Attributes
        ----------
        chromosomes : List[str]
            e.g. ["chr1A","chr1B","chr2A","chr2B", ..., "chr22A","chr22B"].
        pos : dict[str, np.ndarray]
            Per-chromosome-copy coordinate array for every gene (start/mid/end, depending on use).
        gene_id : dict[str, np.ndarray]
            Gene names (or indices) matching `pos`.
        genome_length : dict[str, float]
            Current length of each chromosome copy in this simulation (changes with add/delete).
        """
        self.pos = {}
        self.gene_id = {}
        self.genome_length = {}
        self.chromosomes: List[str] = []

        self.sigma_raw = sigma
        self.expr_ub = expr_ub
        self.expr_lb = expr_lb

        self.min_event_size = min_event_size
        self.max_event_size = max_event_size
        self.gene_order = gene_order


    def load_real_data(self, df_chrlen, df_gene):
        """
        Read the chr_lengths and gene_order tables and build per-chromosome gene positions.
        Load chr1-22 lengths and gene annotation, and create an A and a B copy of each chromosome.
        """

        # 1) Put the chromosome lengths into a dict
        chrlen_dict = dict(zip(df_chrlen["chr"], df_chrlen["length"]))
        # Intersect the chromosomes present in df_gene with those in chrlen_dict
        genes_chroms = df_gene["chr"].unique()
        lengths_chroms = list(chrlen_dict.keys())
        valid_chroms = np.sort(np.intersect1d(genes_chroms, lengths_chroms))
        df_gene = df_gene[df_gene["chr"].isin(valid_chroms)].copy()


        # 2) Read the gene positions
        df_gene["chr"] = df_gene["chr"].astype(str)

        # Use the gene midpoint as its representative position:
        df_gene["mid"] = (df_gene["start"] + df_gene["end"]) // 2

        # Sort
        df_gene.sort_values(by=["chr", "mid"], inplace=True)

        # 3) Iterate over chromosomes; each yields an A and a B copy

        for base_chr_name in valid_chroms:
            # Take the genes on this chromosome
            sub = df_gene[df_gene["chr"] == base_chr_name]
            sub_pos = sub["mid"].values  # use the midpoint as the gene position
            sub_genes = sub["gene"].values
            # Read the chromosome length
            length_val = float(chrlen_dict[base_chr_name])

            # Copy A
            chrA = base_chr_name + "A"
            self.chromosomes.append(chrA)
            self.pos[chrA] = sub_pos.copy()
            self.gene_id[chrA] = sub_genes.copy()
            self.genome_length[chrA] = length_val

            # Copy B
            chrB = base_chr_name + "B"
            self.chromosomes.append(chrB)
            self.pos[chrB] = sub_pos.copy()
            self.gene_id[chrB] = sub_genes.copy()
            self.genome_length[chrB] = length_val

        # Full gene set
        self.all_genes_list = list(self.gene_order)
        self.G = len(self.gene_order)

        # Also need a gene_name -> index map (in case sub_genes holds strings)
        self.gene_index_map = {gname: i for i,gname in enumerate(self.all_genes_list)}

        self.gene_names = {}
        for chrom in self.chromosomes:
            self.gene_names[chrom] = self.gene_id[chrom].copy()

        # Then convert self.gene_id[chrom] back into numeric indices
        for chrom in self.chromosomes:
            new_ids = [self.gene_index_map[gene] for gene in self.gene_names[chrom]]
            self.gene_id[chrom] = np.array(new_ids, dtype=int)

        self.gene_id_original = {chrom: self.gene_id[chrom].copy() for chrom in self.chromosomes}

        print("Load data done. Chromosomes:", self.chromosomes)
        return df_gene

    def create_view(self) -> "GenomeView":
        return GenomeView(self)

    def sample_event(self, chrom: Optional[str] = None) -> tuple:
        """
        Pick a chromosome at random from self.chromosomes (when chrom=None) and draw
        a random (center, width) inside it, where
        width is drawn from [min_event_size, min(max_event_size, chromosome length)].
        Chromosomes shorter than min_event_size are handled as a special case.
        """
        # 1) If the caller did not name a chromosome, pick one at random from self.chromosomes
        if chrom is None:
            chrom = np.random.choice(self.chromosomes)

        chrom_len = self.genome_length[chrom]
        center = np.random.uniform(0, chrom_len)

        # First compare the chromosome length against self.max_event_size
        event_upper_bound = min(self.max_event_size, chrom_len)

        if event_upper_bound <= self.min_event_size:
            # For a very short chromosome (e.g. <1 kb) let the half-width be half its length
            half_width = event_upper_bound / 2.0
        else:
            # Sample the full event length, then convert it to the half-width in_range expects
            event_len = np.random.uniform(self.min_event_size, event_upper_bound)
            half_width = event_len / 2.0
        return (chrom, center, half_width)

    def in_range(self, chrom: str, center: float, width: float) -> np.ndarray:
        """
        Return the indices of every gene falling inside [center-width, center+width].
        """
        chrom_pos = self.pos[chrom]
        mask = np.abs(chrom_pos - center) < width
        idx = np.where(mask)[0]
        return idx

    def add(self, chrom: str, idx: np.ndarray) -> None:
        """
        Duplicate the genes at indices `idx` on chromosome `chrom`.
        """
        if len(idx) == 0:
            return
        start_pos = idx[0]
        end_pos = idx[-1]

        amp_lens = self.pos[chrom][start_pos:end_pos + 1] # amplified positions
        delta = amp_lens[-1] - amp_lens[0] # amplified length

        p1_pos = self.pos[chrom][0:end_pos + 1] # coordinates before the amplified segment
        p2_pos = amp_lens + delta # amplified segment, shifted by delta
        p3_pos = self.pos[chrom][end_pos + 1:] + delta # genes downstream of the segment shift by delta
        self.pos[chrom] = np.hstack((p1_pos, p2_pos, p3_pos)) # stitch together the new chromosome coordinates

        p1_gen = self.gene_id[chrom][0:end_pos + 1]
        p2_gen = self.gene_id[chrom][idx]
        p3_gen = self.gene_id[chrom][end_pos + 1:]
        self.gene_id[chrom] = np.hstack((p1_gen, p2_gen, p3_gen))

        self.genome_length[chrom] += delta # update genome_length

    def delete(self, chrom: str, idx: np.ndarray) -> None:
        """
        Delete the genes at indices `idx` on chromosome `chrom`.
        """
        if len(idx) == 0:
            return
        start_pos = idx[0]
        end_pos   = idx[-1]

        amp_lens = self.pos[chrom][start_pos : end_pos + 1]
        delta = amp_lens[-1] - amp_lens[0]

        p1_pos = self.pos[chrom][0:start_pos]
        p2_pos = self.pos[chrom][end_pos + 1 : ] - delta
        new_pos = np.hstack((p1_pos, p2_pos))

        p1_gen = self.gene_id[chrom][0:start_pos]
        p2_gen = self.gene_id[chrom][end_pos + 1 : ]
        new_gen = np.hstack((p1_gen, p2_gen))

        self.pos[chrom] = new_pos
        self.gene_id[chrom] = new_gen
        self.genome_length[chrom] -= delta

    def get_total_gene_count(self) -> np.ndarray:
        """
        Sum each gene's copy number over the whole genome (chr1-22, copies A and B).
        Returns a counts array of shape (self.G,).
        """
        counts = np.zeros(self.G, dtype=int)
        for chrom in self.chromosomes:
            expected = self.gene_id_original[chrom]  # genes expected to be present (numeric index array)
            current = self.gene_id[chrom]            # genes actually present now (numeric index array)
            chrom_counts = np.bincount(current, minlength=self.G)
            counts[expected] += chrom_counts[expected]
        return counts

    def get_chr_gene_count(self) -> np.ndarray:
        """
        Sum each gene's copy number over the genome, copy A only (chr1-22).
        Returns a counts array of shape (self.G,).
        """
        counts_dict = {}
        # Iterate over the chromosomes
        for chrom in self.chromosomes:
            # Expected gene set, kept in the original order
            expected = self.gene_id_original[chrom]
            # Current gene list (after add/delete operations)
            current = self.gene_id[chrom]
            # For each expected gene, count how many times it occurs in `current`
            counts = np.array([np.sum(current == gene) for gene in expected], dtype=int)
            counts_dict[chrom] = counts
        return counts_dict

class GenomeView:
    """
    Per-cell genome view. Built on a read-only Genome base, it maintains per-haplotype copy counts incrementally as events are applied.
    """
    def __init__(self, base: Genome):
        self.base = base
        self.events = []
        self._hap_counts_cache: Optional[Dict[str, np.ndarray]] = None
        self._total_counts_cache: Optional[np.ndarray] = None

    # ---- Pass-through / compatibility ----
    @property
    def chromosomes(self):
        return self.base.chromosomes

    @property
    def all_genes_list(self):
        return self.base.all_genes_list

    @property
    def G(self):
        return self.base.G

    @property
    def genome_length(self):
        return self.base.genome_length

    def sample_event(self, chrom: Optional[str] = None):
        return self.base.sample_event(chrom=chrom)

    def in_range(self, chrom: str, center: float, width: float) -> np.ndarray:
        return self.base.in_range(chrom, center, width)

    def indices_in_window(self, chrom: str, start: int, end: int) -> np.ndarray:
        return indices_in_window(self.base, chrom, start, end)

    # ---- Internals ----
    def _ensure_hap_cache(self):
        if self._hap_counts_cache is None:
            self._hap_counts_cache = {}
            for chrom in self.base.chromosomes:
                n = len(self.base.gene_id_original[chrom])
                self._hap_counts_cache[chrom] = np.ones(n, dtype=np.uint16)

    def _invalidate_caches(self):
        self._total_counts_cache = None

    # ---- Event application: add / delete ----
    def add(self, chrom: str, idx: np.ndarray) -> None:
        if len(idx) == 0:
            return
        self._ensure_hap_cache()
        self._hap_counts_cache[chrom][idx] += 1
        self._invalidate_caches()

    def delete(self, chrom: str, idx: np.ndarray) -> None:
        if len(idx) == 0:
            return
        self._ensure_hap_cache()
        arr = self._hap_counts_cache[chrom]
        arr[idx] = np.maximum(arr[idx] - 1, 0)
        self._invalidate_caches()

    # ---- Statistics ----
    def get_total_gene_count(self) -> np.ndarray:
        if self._hap_counts_cache is None:
            return np.full(self.base.G, 2, dtype=np.uint16)
        if self._total_counts_cache is not None:
            return self._total_counts_cache
        counts = np.zeros(self.base.G, dtype=np.uint16)
        for chrom, hap_counts in self._hap_counts_cache.items():
            expected = self.base.gene_id_original[chrom]
            counts[expected] += hap_counts.astype(np.uint16)
        self._total_counts_cache = counts
        return counts

    def get_chr_gene_count(self) -> dict:
        if self._hap_counts_cache is None:
            return {chrom: np.ones(len(self.base.gene_id_original[chrom]), dtype=np.uint16)
                    for chrom in self.base.chromosomes}
        return self._hap_counts_cache

    # ---- Lightweight fork, used at cell division ----
    def fork(self) -> "GenomeView":
        v = GenomeView(self.base)
        if self._hap_counts_cache is not None:
            v._hap_counts_cache = {k: arr.copy() for k, arr in self._hap_counts_cache.items()}
        v._total_counts_cache = None
        return v

class Cell:
    def __init__(
        self,
        genome: Genome,
        counter: ut.Counter,
        cell_type: Optional[str] = None,
        mutation_rate: Optional[np.ndarray] = None,
        action_rates: Optional[np.ndarray] = None,
        x: int = 0,
        y: int = 0,
        active_state: int = 5,
        p_move: Optional[np.ndarray] = None,
        tmat: Optional[np.ndarray] = None,
        parent: int = -1,
        clone_id = None, #None:normal cells
        cnv_events_df: Optional[pd.DataFrame] = None,
        event_logger: Optional[EventLogger] = None,
        mut_sched: Optional[Dict] = None,
    ) -> None:
        self.birth_iter = 0
        self.counter = counter
        self.id = self.counter()
        self.parent = parent
        self.children = []
        self.clone_id=clone_id
        self.cell_type=cell_type
        self.genome = genome

        self.N = self.genome.G # total number of genes

        self.baseline_expr_id = f"tumor_expr_{clone_id}"

        self.cnv_events_df = cnv_events_df
        self.event_logger  = event_logger
        self._iter = 0  # current iteration step, used for logging
        self.mut_sched = (mut_sched or {})

        if mutation_rate is not None:
            self.mutation_rate = mutation_rate
        else:
            self.mutation_rate = np.random.uniform(0, 1, size=2)

        if action_rates is not None:
            self.action_rates = action_rates

            self.action_rates /= self.action_rates.sum(axis=0, keepdims=True)
        else:
            self.action_rates = np.random.dirichlet(np.ones(4), size=2).T

        if tmat is None: # tmat: cell-state transition matrix
            self.tmat = np.array([[0.8, 0.2], [0.4, 0.6]]).T
        else:
            self.tmat = tmat

        self.active = active_state

        assert np.all(self.action_rates >= 0), "rates must be non-negative than zero"
        self.crd = np.array([int(x), int(y)]) # cell coordinates on the lattice

        if p_move is None: # p_move: cell movement probability distribution
            self.p_move = np.random.dirichlet(0.1 * np.ones(4))
        else:
            self.p_move = p_move

        #self.occupied_positions = [(x, y)]  # list of lattice coordinates actually occupied by this cell
        #self.cell_size = cell_size   # record cell size / volume


        self.gamma = 0.05

        self.dx = [-1, 0, 1, 0]
        self.dy = [0, -1, 0, 1]
        self.step_scale = [1, 2]



    def search_nbrhd(
        self,
        cell_map: np.ndarray,
        forbidden_mask: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Look for free neighbouring positions

        free_space = []
        pvals = []
        scale_x = np.random.choice(self.step_scale)
        scale_y = np.random.choice(self.step_scale)
        for k, (dx, dy) in enumerate(zip(self.dx, self.dy)):
            new_x = int(self.crd[0] + dx * scale_x)
            new_y = int(self.crd[1] + dy * scale_y)

            if (new_x < 0) | (new_x >= cell_map.shape[0]):
                continue
            elif (new_y < 0) | (new_y >= cell_map.shape[1]):
                continue

            # Mutual exclusion: tumour cells may not enter TLS_block (forbidden_mask)
            if forbidden_mask is not None and (self.clone_id is not None):
                if forbidden_mask[new_x, new_y]:
                    continue

            if cell_map[new_x, new_y] == 0:
                free_space.append((new_x, new_y))
                pvals.append(self.p_move[k])

        if len(pvals) > 0:
            pvals = [x / sum(pvals) for x in pvals]
        return free_space, pvals

    # def genome_update(
    #     self,
    # ) -> None:
    #     # gene mutation
    #     # check if mutation should occur
    #     if np.random.random() < self.mutation_rate[self.state]:
    #         self.mutation_rate[self.state] *= 0.95
    #         chrom, center, length = self.genome.sample_event()
    #         idx = self.genome.in_range(chrom, center, length)

    #         if np.random.random() < 0.5:
    #             self.genome.add(chrom,idx)
    #         else:
    #             self.genome.delete(chrom,idx)

    def genome_update(self, iter_t=None, event_logger=None):
        # Effective mutation rate with age decay (applies to tumour state 1 only; configurable)
        base_rate = float(self.mutation_rate[self.state])
        eff_rate = base_rate
        try:
            if (self.clone_id is not None) and (self.state == 1):
                cfg = self.mut_sched or {}
                enable = bool(cfg.get("enable_state1_age_decay", True))
                if enable and iter_t is not None:
                    birth = int(getattr(self, "birth_iter", 0))
                    age = max(0, int(iter_t) - birth)
                    half_life = float(cfg.get("half_life_iter", 40.0))
                    min_factor = float(cfg.get("min_factor", 0.0))
                    if half_life > 0:
                        factor = 0.5 ** (age / half_life)
                        factor = max(min_factor, factor)
                        eff_rate = base_rate * factor
        except Exception:
            eff_rate = base_rate

        if np.random.random() < eff_rate:
            self.mutation_rate[self.state] *= 0.95

            logger = event_logger or self.event_logger  # use the logger passed in, else our own

            if self.cnv_events_df is not None:
                evt = sample_uniform_event(self.cnv_events_df)
                apply_catalog_event_to_genome(
                    self.genome,
                    evt["Chromosome"], int(evt["Start"]), int(evt["End"]),
                    int(evt["Major_Copy_Number"]), int(evt["Minor_Copy_Number"]),
                    event_logger=logger,
                    cell=self,
                    iter_t=iter_t,
                    source=evt,
                    context="evolution"
                )
            else:
                chrom, center, length = self.genome.sample_event()
                idx = self.genome.in_range(chrom, center, length)
                if np.random.random() < 0.5:
                    self.genome.add(chrom, idx)
                else:
                    self.genome.delete(chrom, idx)

    def sample_expression_with_random_count(
        self,
        tumor_baseline_exp: Optional[Dict[str, np.ndarray]] = None,
        normal_TME_exp: Optional[Dict[str, np.ndarray]] = None,
        copy2fold_func=ut.copy2fold_func,
        lognorm_params_tumor=(9.0, 0.5),   # mean, sigma for tumor cell
        lognorm_params_normal=(8.0, 0.5), # mean, sigma for normal cell
        dirichlet_precision=100,
    ) -> np.ndarray:
        """
        Generate a discrete expression count vector for this cell.

        Tumour cells (clone_id is not None):
        - use the tumour baseline expression stored in baseline_expr_dict (key: tumor_expr_{clone_id}),
        - derive fold changes from the genome copy number,
        - draw the total count from a lognormal+Poisson, then gene-level counts from a Dirichlet.
    
        Normal cells (clone_id is None):
        - take the baseline for this cell_type from normal_TME_exp (a dict keyed by normal cell type),
        - ignore copy number,
        - otherwise follow the same steps as a tumour cell.
        """
        EPS=1e-8
        #1. generate total_count
        if self.clone_id is not None:
            mu, sigma = lognorm_params_tumor
        else:
            mu, sigma = lognorm_params_normal
        lam = np.random.lognormal(mean=mu, sigma=sigma)
        total_count = np.random.poisson(lam)
        #2. calculate raw_expr
        if self.clone_id is not None:
            if tumor_baseline_exp is None:
                raise ValueError("For tumor cells，the 'tumor_baseline_exp' parameter is required.")
            # 1) Fetch the matching baseline_expr
            baseline_series = tumor_baseline_exp[self.baseline_expr_id]
            baseline_vec = baseline_series.reindex(self.genome.all_genes_list).values
            copy_counts = self.genome.get_total_gene_count()
            fold_factors = copy2fold_func(copy_counts)
            raw_expr = baseline_vec * fold_factors
        else:
            if normal_TME_exp is None:
                raise ValueError("For normal cells，the 'normal_TME_exp' parameter is required.")
            baseline_series = normal_TME_exp[self.cell_type].copy()
            raw_expr = baseline_series.reindex(self.genome.all_genes_list).values.copy()
        #3. normalize raw_expr
        s = raw_expr.sum()
        base = raw_expr / s
        base = base + EPS
        base = base / base.sum()

        #4. sample p
        if dirichlet_precision is None or dirichlet_precision <= 0:
            p = base
        else:
            alpha = base * float(dirichlet_precision)
            alpha = np.where(alpha <= 0, EPS, alpha)
            p = np.random.dirichlet(alpha)

        #5. generate expr_counts
        counts = np.random.multinomial(int(total_count), p).astype(np.int64)
        return counts

    @property
    def state(
        self,
    ) -> int:
        return int(self.active < 0)

    def action(
        self,
        cell_map: np.ndarray,
        iter_t: int = 0,
        event_logger=None,
        forbidden_mask: np.ndarray = None,
        tls_soft: np.ndarray = None,
        tls_cfg: dict = None,
        imm_soft: np.ndarray = None,
        imm_cfg: dict = None
    ) -> Union[List[None], List[Cell]]:

        if self.active < 0:
            new_state = np.random.multinomial(1, self.tmat[:, 1]).argmax()
            if new_state == 0:
                self.active = np.random.geometric(p=self.tmat[1, 0])
        else:
            self.active -= 1

        # Take the "local" action distribution (by current state, or a single column)
        base = self.action_rates
        col = 0 if base.shape[1] == 1 else self.state
        local = base[:, col].astype(float).copy()

        # --- TLS suppression (soft, optional, can be raised or lowered per state) ---
        # grow_tissue supplies tls_soft (0..1, larger the closer to a TLS) and tls_cfg.
        # Config priority: spawn_mod/death_mod (length-2 arrays applied per state; negative lowers, positive raises).
        # Legacy fallback: without spawn_mod/death_mod, use the scalar tumor_block_spawn/death (equivalent to spawn_mod=[-beta_s,-beta_s], death_mod=[+beta_d,+beta_d]).
        if self.clone_id is not None and tls_soft is not None:  # tumour cells only
            xx, yy = int(self.crd[0]), int(self.crd[1])
            s_here = float(tls_soft[xx, yy]) if 0 <= xx < tls_soft.shape[0] and 0 <= yy < tls_soft.shape[1] else 0.0
            if s_here > 0.0:
                col = 0 if self.action_rates.shape[1] == 1 else self.state
                cfg = (tls_cfg or {})
                sm = cfg.get("spawn_mod", None)
                dm = cfg.get("death_mod", None)
                if sm is None:
                    beta_s = float(cfg.get("tumor_block_spawn", 0.0))
                    sm = [-beta_s, -beta_s]
                if dm is None:
                    beta_d = float(cfg.get("tumor_block_death", 0.0))
                    dm = [beta_d, beta_d]
                try:
                    spawn_scale = max(0.0, 1.0 + float(sm[col]) * s_here)
                except Exception:
                    spawn_scale = max(0.0, 1.0 + float(sm if np.isscalar(sm) else 0.0) * s_here)
                try:
                    death_scale = max(0.0, 1.0 + float(dm[col]) * s_here)
                except Exception:
                    death_scale = max(0.0, 1.0 + float(dm if np.isscalar(dm) else 0.0) * s_here)
                local[1] *= spawn_scale
                local[3] *= death_scale

        # --- Immune soft suppression (weaker than TLS, optional, per-state up/down) ---
        if self.clone_id is not None and imm_soft is not None:
            xx, yy = int(self.crd[0]), int(self.crd[1])
            s_here = float(imm_soft[xx, yy]) if 0 <= xx < imm_soft.shape[0] and 0 <= yy < imm_soft.shape[1] else 0.0
            if s_here > 0.0:
                col = 0 if self.action_rates.shape[1] == 1 else self.state
                cfg = (imm_cfg or {})
                sm = cfg.get("spawn_mod", None)
                dm = cfg.get("death_mod", None)
                if sm is None:
                    beta_s = float(cfg.get("tumor_block_spawn", 0.0))
                    sm = [-beta_s, -beta_s]
                if dm is None:
                    beta_d = float(cfg.get("tumor_block_death", 0.0))
                    dm = [beta_d, beta_d]
                try:
                    spawn_scale = max(0.0, 1.0 + float(sm[col]) * s_here)
                except Exception:
                    spawn_scale = max(0.0, 1.0 + float(sm if np.isscalar(sm) else 0.0) * s_here)
                try:
                    death_scale = max(0.0, 1.0 + float(dm[col]) * s_here)
                except Exception:
                    death_scale = max(0.0, 1.0 + float(dm if np.isscalar(dm) else 0.0) * s_here)
                local[1] *= spawn_scale
                local[3] *= death_scale

        # Normalise, then draw an action
        local = np.maximum(local, 1e-12)
        local = local / local.sum()
        action = np.random.multinomial(1, local).argmax()

        if action == 3 and cell_map.sum() > 50:
            cell_map[self.crd[0], self.crd[1]] = 0
            return []
        else:
            free_space, pvals = self.search_nbrhd(cell_map, forbidden_mask=forbidden_mask)
            if len(free_space) > 0:
                new_crd_idx = np.random.choice(len(free_space), p=pvals)
                new_crd = np.array(free_space[new_crd_idx])

                if action == 1:
                    #self.genome_update()
                    self.genome_update(iter_t=iter_t, event_logger=event_logger)

                    new_genome = self.genome.fork() if hasattr(self.genome, "fork") else copy.deepcopy(self.genome)
                    new_cell = Cell(
                        genome=new_genome,
                        mutation_rate=self.mutation_rate,
                        action_rates=self.action_rates,
                        counter=self.counter,
                        x=new_crd[0],
                        y=new_crd[1],
                        active_state=self.active,
                        p_move=(self.p_move if self.active >= 0 else None),
                        tmat=self.tmat,
                        parent=self.id,
                        clone_id = self.clone_id,
                        cell_type = self.cell_type,
                        cnv_events_df=self.cnv_events_df,
                        event_logger=self.event_logger,  # important: hand the logger to the daughter cell
                        mut_sched=getattr(self, "mut_sched", None),
                    )
                    try:
                        new_cell.birth_iter = int(iter_t)
                    except Exception:
                        pass
                    self.children.append(new_cell.id)

                    cell_map[new_crd[0], new_crd[1]] = 1

                    return [self, new_cell]

                elif action == 2:
                    cell_map[self.crd[0], self.crd[1]] = 0
                    self.crd = new_crd
                    cell_map[self.crd[0], self.crd[1]] = 1
                    return [self]
                else:
                    return [self]
            else:
                return [self]

def grow_tissue(
    n_initial_cells: int,
    genome: Genome,
    action_rates: Optional[np.ndarray] = None,
    mutation_rate: Optional[np.ndarray] = None,
    transitions: Optional[np.ndarray] = None,
    max_iter: int = 500,
    max_cells: Union[int, float] = 1e4,
    domain_side_size: float = 200,
    init_coordinates: Optional[np.ndarray] = None,
    init_event_size: Optional[int] = None,
    debug: bool = False,
    output_dir: str = None,
    cnv_events_df: Optional[pd.DataFrame] = None,
    event_logger: Optional[EventLogger] = None,
    n_init_shared: int = 0,                  # number of events shared by all initial clones
    n_init_private: Union[int, List[int]] = 0,  # private events per clone (int, or a list of length n_initial_cells)
    # --- Normal-cell drip / TLS / weight maps / cell-type grouping ---
    enable_drip: bool = True,
    drip_every: int = 20,
    drip_fill_ratio: float = 0.05,
    drip_per_cycle_max: Optional[int] = None,
    immune_tls_share: float = 0.60,
    tls_params: Dict = None,              # design.toml [tls]
    weight_params: Dict = None,           # design.toml [weights]
    normal_type_prop_map: Dict[str, float] = None,  # [normal_celltype_proportions]
    celltype_groups_map: Dict[str, str] = None,     # [celltype_groups]
    tls_celltypes: List[str] = None,                # [tls].tls_celltypes (the concrete types allowed inside a TLS)
    forbid_tumor_on_tls: bool = True,
    # --- Three-mode immune placement parameters ---
    immune_place_probs: Dict[str, float] = None,
    edge_sigma: float = 1.5,
    inside_power: float = 1.0,
    outside_sigma: float = 2.5,
    immune_mix_sample: bool = False,
    immune_mix_alpha: Tuple[float, float, float] = (3.3, 3.4, 3.3),
    # Perturbation parameters accepted for interface parity; unused in this variant
    perturb_params: Dict = None,
    # --- Automatic iteration: stop once target occupancy is reached ---
    auto_iter_enable: bool = True,
    occupancy_target: float = 0.90,
    immune_soft_params: Dict = None,
    mutation_schedule: Dict = None,
) -> Tuple[List[Cell], Dict[int, int], np.ndarray]:

    event_records = []   # global event list for this simulation
    n_genes = genome.G

    if action_rates is None:
        action_rates = np.array(((0.1, 0.3), (0.4, 0.45), (0.4, 0.2), (0.0, 0.00)))
    else:
        action_rates = action_rates

    if transitions is None:
        tmat = np.array([[0.8, 0.2], [0.2, 0.8]]).T
    else:
        tmat = transitions

    if mutation_rate is None:
        mutation_rate = np.array([0.0, 0.8])
    else:
        mutation_rate = mutation_rate

    if init_coordinates is None:
        init_coordinates = []
        for i in range(n_initial_cells):
            xy = np.random.uniform(
                0.1 * domain_side_size,
                0.9 * domain_side_size,
                size=2,
            )
            init_coordinates.append(xy)


    else:
        if isinstance(init_coordinates[0], int) or isinstance(
            init_coordinates[0], float
        ):
            init_coordinates = [init_coordinates]

            assert (
                len(init_coordinates) == n_initial_cells
            ), "incorrect number of initial coordinate tuples"
            assert len(init_coordinates[0]) == 2, "coordinate _tuple_ is required"

            for ic in init_coordinates:
                for ii in range(len(ic)):
                    assert (
                        (ic[ii] >= 0) and (ic[ii] < domain_side_size)
                    ), "Initial coordinate pair ({},{}) is out of bounds".format(
                        ic[0], ic[1]
                    )

    cell_map = np.zeros((int(domain_side_size), int(domain_side_size)))
    counter = ut.Counter()
    lineage = {}
    population = []

    # ---- Pre-draw the shared initial events (from cnv_events_df) ----
    shared_events = []
    if cnv_events_df is not None and len(cnv_events_df) > 0 and n_init_shared > 0:
        n_pick = min(n_init_shared, len(cnv_events_df))
        shared_idx = np.random.choice(len(cnv_events_df), size=n_pick, replace=False)
        shared_events = [cnv_events_df.iloc[j] for j in shared_idx]

    for i in range(n_initial_cells):
        seed_cell = Cell(
            genome=genome.create_view() if hasattr(genome, "create_view") else copy.deepcopy(genome),
            counter=counter,
            mutation_rate=mutation_rate,
            action_rates=action_rates,
            tmat=tmat,
            x=init_coordinates[i][0],
            y=init_coordinates[i][1],
            clone_id = i,
            cell_type = 'Tumor',
            cnv_events_df=cnv_events_df,
            event_logger=event_logger,
            mut_sched=(mutation_schedule or {}),
        )


        if cnv_events_df is not None and len(cnv_events_df) > 0:
            # Apply the shared initial events first (the same batch for every initial clone)
            if shared_events:
                for evt in shared_events:
                    apply_catalog_event_to_genome(
                        seed_cell.genome,
                        evt["Chromosome"], int(evt["Start"]), int(evt["End"]),
                        int(evt["Major_Copy_Number"]), int(evt["Minor_Copy_Number"]),
                        event_logger=event_logger, cell=seed_cell, iter_t=0, source=evt,
                        context="init_shared"
                    )

            if isinstance(n_init_private, list):
                n_priv = int(n_init_private[i]) if i < len(n_init_private) else 0
            else:
                n_priv = int(n_init_private)

            for _ in range(max(0, n_priv)):
                evt = sample_uniform_event(cnv_events_df)
                apply_catalog_event_to_genome(
                    seed_cell.genome,
                    evt["Chromosome"], int(evt["Start"]), int(evt["End"]),
                    int(evt["Major_Copy_Number"]), int(evt["Minor_Copy_Number"]),
                    event_logger=event_logger, cell=seed_cell, iter_t=0, source=evt,
                    context="init_private"
                )
        else:
            pass

        cell_map[int(init_coordinates[i][0]), int(init_coordinates[i][1])] = 1
        lineage.update({seed_cell.parent: [seed_cell.id]})
        population.append(seed_cell)

    # ==== One-off TLS placement, away from the tumour ====
    if tls_params is None: tls_params = {}
    if weight_params is None: weight_params = {}
    groups_map = _normalize_groups_map(celltype_groups_map or {})
    tls_set = set([ (s or "").strip().lower() for s in (tls_celltypes or []) ])

    # Tumour mask (clone_id != None)
    tumor_mask = np.zeros_like(cell_map, dtype=bool)
    for c in population:
        if c.clone_id is not None:
            tumor_mask[int(c.crd[0]), int(c.crd[1])] = True

    # TLS parameters (away-from-tumour strategy)
    n_tls           = int(tls_params.get('n_tls', 2))
    r_tls           = int(tls_params.get('r_tls', 8))
    r_block         = int(tls_params.get('r_block', 2))
    min_center_dist = float(tls_params.get('min_center_dist', 15.0))
    min_edge_gap    = float(tls_params.get('min_edge_gap', 1.0))
    border_margin   = int(tls_params.get('border_margin', 0))
    suppress_sigma  = float(tls_params.get('suppress_sigma', 3.0))  # soft-suppression blur radius

    # Distance to the nearest tumour cell
    dist_map = distance_transform_edt(~tumor_mask)
    center_thresh = max(min_center_dist, r_tls + r_block + min_edge_gap)
    cand_mask = (~tumor_mask) & (dist_map >= center_thresh)
    if border_margin > 0:
        H, W = cand_mask.shape
        cand_mask[:border_margin, :] = False
        cand_mask[-border_margin:, :] = False
        cand_mask[:, :border_margin] = False
        cand_mask[:, -border_margin:] = False
    cand = np.column_stack(np.where(cand_mask))
    if len(cand) == 0:
        for scale in (0.75, 0.5, 0.0):
            thr = center_thresh * scale
            cand_mask2 = (~tumor_mask) & (dist_map >= thr)
            if border_margin > 0:
                cand_mask2[:border_margin, :] = False
                cand_mask2[-border_margin:, :] = False
                cand_mask2[:, :border_margin] = False
                cand_mask2[:, -border_margin:] = False
            cand = np.column_stack(np.where(cand_mask2))
            if len(cand) > 0:
                break

    TLS_core = np.zeros_like(tumor_mask, dtype=bool)
    if len(cand) > 0 and n_tls > 0:
        chosen = []
        # the first centre is random
        c0 = cand[np.random.randint(len(cand))]
        chosen.append(tuple(c0))
        # the rest: farthest-point sampling
        for _ in range(1, n_tls):
            dmin = np.full(len(cand), np.inf, dtype=float)
            for i, p in enumerate(cand):
                for q in chosen:
                    d = (p[0]-q[0])**2 + (p[1]-q[1])**2
                    if d < dmin[i]:
                        dmin[i] = d
            idx = int(np.argmax(dmin))
            chosen.append(tuple(cand[idx]))
        for cc in chosen:
            TLS_core |= _disk_mask(tumor_mask.shape, cc, r_tls)
    TLS_core &= (~tumor_mask)  # belt and braces: never overlap the tumour
    TLS_block = binary_dilation(TLS_core, iterations=r_block)  # hard exclusion mask
    tls_soft  = gaussian_filter(TLS_core.astype(float), sigma=suppress_sigma)  # soft suppression field (0 to peak)
    if tls_soft.max() > 0:
        tls_soft = tls_soft / tls_soft.max()

    # ==== Initial weight maps: Stromal rim + tumour-internal edge bias; Immune = TLS peak + outer band; Epithelial further out ====
    # ---- Fix the immune mixing coefficients once (inside/edge/outside) ----
    if immune_mix_sample:
        _mix = np.random.dirichlet(np.array(immune_mix_alpha, dtype=float))
        p_in_fixed, p_edge_fixed, p_out_fixed = float(_mix[0]), float(_mix[1]), float(_mix[2])
    else:
        _p_in   = float((immune_place_probs or {}).get('inside', 1/3))
        _p_edge = float((immune_place_probs or {}).get('edge',   1/3))
        _p_out  = float((immune_place_probs or {}).get('outside',1/3))
        _P = max(1e-8, (_p_in + _p_edge + _p_out))
        p_in_fixed, p_edge_fixed, p_out_fixed = _p_in / _P, _p_edge / _P, _p_out / _P

    def _apply_temperature_and_tail(W, temp=1.0, eps=0.0):
        W = np.clip(W, 0, None).astype(float)
        if temp != 1.0:
            W = np.power(W, max(1e-6, float(temp)))
        if eps > 0.0:
            W = W + float(eps)
        return np.clip(W, 0, None)

    def _lowfreq_noise(shape, sigma=6.0, seed=None):
        # Use the global numpy RNG so the global seed controls this
        x = np.random.random(shape)
        n = gaussian_filter(x, sigma=sigma)
        n -= n.min(); d = (n.max() - n.min()) + 1e-8
        return n / d

    def _apply_noise(W, gain=0.0, sigma=6.0, seed=None):
        W = np.clip(W, 0, None).astype(float)
        m = W.max()
        if m <= 0 or gain <= 0:
            return W
        Wn = W / m
        N  = _lowfreq_noise(W.shape, sigma=sigma, seed=seed)
        Wn = (1.0 - gain) * Wn + gain * N
        return np.clip(Wn, 0, None) * m

    def _add_micro_blobs(W, k=0, radius=3, gain=1.2, seed=None):
        if k <= 0:
            return W
        H, Wd = W.shape
        out = W.astype(float).copy()
        for _ in range(int(k)):
            cx = np.random.randint(radius, max(radius+1, H - radius)) if H > 2*radius else H//2
            cy = np.random.randint(radius, max(radius+1, Wd - radius)) if Wd > 2*radius else Wd//2
            rr, cc = np.ogrid[:H, :Wd]
            disk = ((rr - cx)**2 + (cc - cy)**2) <= int(radius)**2
            out[disk] += float(gain) * (out.max() if out.max() > 0 else 1.0) * 0.05
        return out

    def _build_weights(D: np.ndarray, TLS_core_local: np.ndarray,
                       mu_s: float = 4.0, w_s: float = 5.0,
                       mu_i: float = None, sigma_i: float = 2.5,
                       delta_epi: float = 2.0, g_tls: float = 7.0):
        if mu_i is None:
            mu_i = mu_s + 1.5
        ring  = np.exp(-((D - mu_s)**2) / (2.0 * (w_s/2.0)**2))
        inner = (D < 0).astype(float) * np.exp(-((D + 1.0)**2) / (2.0 * (1.5)**2))
        W_stromal = np.clip(ring + 0.6 * inner, 0, None)
        # TLS core + outer immune band
        W_tls   = gaussian_filter(TLS_core_local.astype(float), sigma=1.0) * g_tls
        imm_out = np.exp(-((D - mu_i)**2) / (2.0 * sigma_i**2))
        imm_out[D < 0] = 0.0
        # Three-mode immune weights: inside/edge/outside (using the fixed mixing coefficients)
        # inside: preference for the tumour interior (D<0, decaying with |D|)
        inside_raw = (D < 0).astype(float) * np.exp(-np.abs(D) ** max(1.0, float(inside_power)))
        # edge: Gaussian band at the tumour margin (D approx 0)
        edge_raw = np.exp(- (D ** 2) / (2.0 * (max(1e-6, float(edge_sigma)) ** 2)))
        # outside: outer immune band (already provided by imm_out)
        outside_raw = np.clip(imm_out, 0, None)
        W_immune = np.clip(p_in_fixed * inside_raw + p_edge_fixed * edge_raw + p_out_fixed * outside_raw, 0, None)
        epi_zone = (D >= (mu_s + w_s/2.0 + delta_epi)).astype(float)
        W_epi    = np.clip(epi_zone, 0, None)
        # ---- Apply perturbations (from perturb_params) ----
        pert = perturb_params or {}
        temp  = float(pert.get("weight_temperature", 1.0))
        eps   = float(pert.get("uniform_tail", 0.0))
        ngain = float(pert.get("noise_gain", 0.0))
        nsig  = float(pert.get("noise_sigma", 6.0))
        k_blob = int(pert.get("micro_blobs_per_group", 0))
        blob_r = int(pert.get("micro_blob_radius", 3))
        blob_g = float(pert.get("micro_blob_gain", 1.2))
        rng_seed = None

        def perturb_one(Wmat, add_k):
            Wmat = _apply_temperature_and_tail(Wmat, temp=temp, eps=eps)
            Wmat = _apply_noise(Wmat, gain=ngain, sigma=nsig, seed=rng_seed)
            Wmat = _add_micro_blobs(Wmat, k=add_k, radius=blob_r, gain=blob_g, seed=rng_seed)
            return np.clip(Wmat, 0, None)

        W_stromal = perturb_one(W_stromal, k_blob)
        W_immune  = perturb_one(W_immune,  int(max(0, k_blob//2)))
        W_epi     = perturb_one(W_epi,     k_blob)
        return W_stromal, W_immune, W_epi
    # ---- Optional: immune soft-suppression config (weaker than TLS) ----
    imm_soft = immune_soft_params or {}
    imm_soft_enable = bool(imm_soft.get("enable", True))
    imm_soft_labels = set([s.lower() for s in imm_soft.get("labels", ["T","B","NK","Plasma"])])
    imm_soft_sigma  = float(imm_soft.get("sigma", 2.0))
    imm_block_spawn = float(imm_soft.get("tumor_block_spawn", 0.10))  # weaker than TLS
    imm_block_death = float(imm_soft.get("tumor_block_death", 0.05))  # weaker than TLS
    imm_soft_map = np.zeros_like(cell_map, dtype=float)

    D0 = _signed_distance(tumor_mask)
    W_stromal, W_immune, W_epi = _build_weights(
        D0, TLS_core,
        mu_s   = float(weight_params.get('mu_s', 4.0)),
        w_s    = float(weight_params.get('w_s', 5.0)),
        mu_i   = weight_params.get('mu_i', None),
        sigma_i= float(weight_params.get('sigma_i', 2.5)),
        delta_epi = float(weight_params.get('delta_epi', 2.0)),
        g_tls  = float(weight_params.get('g_tls', 7.0)),
    )

    print()
    stopped_by = None  # "occupancy" | "max_cells" | None (hit max_iter)
    for t in range(max_iter):
        print("\r Iteration : {} | Cells : {}".format(t, len(population)), end="")
        new_cells = []

        # ==== Update the immune soft-suppression field (blurred T/B/NK/Plasma density) ====
        if imm_soft_enable:
            imm_soft_map[:] = 0.0
            for c in population:
                if c.clone_id is None and (c.cell_type or "").strip().lower() in imm_soft_labels:
                    x, y = int(c.crd[0]), int(c.crd[1])
                    if 0 <= x < imm_soft_map.shape[0] and 0 <= y < imm_soft_map.shape[1]:
                        imm_soft_map[x, y] = 1.0
            if imm_soft_sigma > 0:
                imm_soft_map[:] = gaussian_filter(imm_soft_map, sigma=float(imm_soft_sigma))
                if imm_soft_map.max() > 0:
                    imm_soft_map[:] = imm_soft_map / imm_soft_map.max()

        # ==== Normal-cell drip: every drip_every iterations ====
        if enable_drip and drip_every and (t > 0) and (t % int(drip_every) == 0):
            # Refresh the tumour mask and distance field
            tumor_mask = np.zeros_like(cell_map, dtype=bool)
            for c in population:
                if c.clone_id is not None:
                    tumor_mask[int(c.crd[0]), int(c.crd[1])] = True
            D = _signed_distance(tumor_mask)
            # Refresh the weight maps (TLS positions stay fixed)
            W_stromal, W_immune, W_epi = _build_weights(
                D, TLS_core,
                mu_s   = float(weight_params.get('mu_s', 4.0)),
                w_s    = float(weight_params.get('w_s', 5.0)),
                mu_i   = weight_params.get('mu_i', None),
                sigma_i= float(weight_params.get('sigma_i', 2.5)),
                delta_epi = float(weight_params.get('delta_epi', 2.0)),
                g_tls  = float(weight_params.get('g_tls', 7.0)),
            )
            # Cells added this cycle = free positions x ratio
            n_free  = int(np.sum(cell_map == 0))
            n_total = int(max(0, round(n_free * float(drip_fill_ratio))))
            if drip_per_cycle_max is not None:
                n_total = min(n_total, int(drip_per_cycle_max))
            if n_total > 0:
                # Read the existing counter (if the constructor needs it)
                counter_like = counter
                for c_ in population:
                    if hasattr(c_, "counter"):
                        counter_like = c_.counter
                        break
                # Allocate to concrete labels per [normal_celltype_proportions]
                per_label_counts = _alloc_by_proportions(n_total, normal_type_prop_map or {})
                # Only tls_celltypes may be seeded into a TLS, split by immune_tls_share
                for lab, n_lab in per_label_counts.items():
                    if n_lab <= 0:
                        continue
                    lab_l = (lab or "").strip().lower()
                    group = groups_map.get(lab_l, "Stromal")
                    if group == "Immune":
                        if lab_l in tls_set:
                            n_tls_seed = int(round(n_lab * float(immune_tls_share)))
                            n_out_seed = int(n_lab - n_tls_seed)
                            W_tls_core = gaussian_filter(TLS_core.astype(float), sigma=1.0)
                            # T/B fill the TLS first; the remainder is scattered by the three-mode immune weight map (with jitter)
                            jprob = float((perturb_params or {}).get("jitter_prob", 0.0))
                            jrad  = int((perturb_params or {}).get("jitter_radius", 1))
                            population, cell_map = _drip_seed_one_type(population, cell_map, lab, W_tls_core, n_tls_seed, genome, counter_like, jitter_prob=jprob, jitter_radius=jrad)
                            population, cell_map = _drip_seed_one_type(population, cell_map, lab, W_immune,   n_out_seed, genome, counter_like, jitter_prob=jprob, jitter_radius=jrad)
                        else:
                            # Other immune subtypes: placed purely by the three-mode immune weight map (inside/edge/outside)
                            jprob = float((perturb_params or {}).get("jitter_prob", 0.0))
                            jrad  = int((perturb_params or {}).get("jitter_radius", 1))
                            population, cell_map = _drip_seed_one_type(population, cell_map, lab, W_immune, n_lab, genome, counter_like, jitter_prob=jprob, jitter_radius=jrad)
                    elif group == "Epithelial":
                        population, cell_map = _drip_seed_one_type(population, cell_map, lab, W_epi, n_lab, genome, counter_like)
                    else:  # Stromal
                        population, cell_map = _drip_seed_one_type(population, cell_map, lab, W_stromal, n_lab, genome, counter_like)

        # SPEEDUP (claude/02): O(n^2) population.pop(0) -> O(n) direct iteration.
        # Iteration order over the existing cells is identical to popping from the
        # front of the list, so RNG consumption order is preserved and outputs
        # remain bit-identical.
        _current_cells = population
        population = []
        for cell in _current_cells:
            #cell._iter = t
            out = cell.action(
                cell_map,
                iter_t=t,
                event_logger=event_logger,
                forbidden_mask=(TLS_block if forbid_tumor_on_tls else None),
                tls_soft=tls_soft,
                tls_cfg=tls_params,
                imm_soft=imm_soft_map if imm_soft_enable else None,
                imm_cfg={"tumor_block_spawn": imm_block_spawn, "tumor_block_death": imm_block_death} if imm_soft_enable else None,
            )
            for c in out:
                if c.id not in lineage:
                    lineage.update({c.id: c.parent})

            new_cells += out

        population += new_cells
        # ---- Stop automatically once occupancy target or max_cells is reached ----
        n_occ = int(cell_map.sum())
        if auto_iter_enable:
            total_slots = int(domain_side_size * domain_side_size)
            if total_slots > 0 and (n_occ / total_slots) >= float(occupancy_target):
                print("\nReached occupancy target: {:.2f}%".format((n_occ/total_slots)*100.0))
                stopped_by = "occupancy"
                break
        if n_occ > max_cells:
            stopped_by = "max_cells"
            break
    # ---- If max_iter ran out below target occupancy, do one final top-up ----
    if auto_iter_enable and (stopped_by is None):
        total_slots = int(domain_side_size * domain_side_size)
        target_n = int(np.ceil(float(occupancy_target) * total_slots))
        n_occ = int(cell_map.sum())
        need = max(0, target_n - n_occ)
        # Never exceed the free positions, nor the max_cells guard
        free_slots = int(np.sum(cell_map == 0))
        if max_cells is not None:
            need = min(need, int(max(0, max_cells - n_occ)))
        need = min(need, free_slots)
        if need > 0:
            # Rebuild the weight maps from the current tumour distribution
            tumor_mask = np.zeros_like(cell_map, dtype=bool)
            for c in population:
                if c.clone_id is not None:
                    tumor_mask[int(c.crd[0]), int(c.crd[1])] = True
            D = _signed_distance(tumor_mask)
            W_stromal, W_immune, W_epi = _build_weights(
                D, TLS_core,
                mu_s   = float(weight_params.get('mu_s', 4.0)),
                w_s    = float(weight_params.get('w_s', 5.0)),
                mu_i   = weight_params.get('mu_i', None),
                sigma_i= float(weight_params.get('sigma_i', 2.5)),
                delta_epi = float(weight_params.get('delta_epi', 2.0)),
                g_tls  = float(weight_params.get('g_tls', 7.0)),
            )
            # Allocate across normal cell types; TLS-first and jitter identical to a regular drip
            counter_like = counter
            for c_ in population:
                if hasattr(c_, "counter"):
                    counter_like = c_.counter
                    break
            per_label_counts = _alloc_by_proportions(need, normal_type_prop_map or {})
            for lab, n_lab in per_label_counts.items():
                if n_lab <= 0:
                    continue
                lab_l = (lab or "").strip().lower()
                group = groups_map.get(lab_l, "Stromal")
                if group == "Immune":
                    if lab_l in tls_set:
                        n_tls_seed = int(round(n_lab * float(immune_tls_share)))
                        n_out_seed = int(n_lab - n_tls_seed)
                        W_tls_core = gaussian_filter(TLS_core.astype(float), sigma=1.0)
                        jprob = float((perturb_params or {}).get("jitter_prob", 0.0))
                        jrad  = int((perturb_params or {}).get("jitter_radius", 1))
                        population, cell_map = _drip_seed_one_type(population, cell_map, lab, W_tls_core, n_tls_seed, genome, counter_like, jitter_prob=jprob, jitter_radius=jrad)
                        population, cell_map = _drip_seed_one_type(population, cell_map, lab, W_immune,   n_out_seed, genome, counter_like, jitter_prob=jprob, jitter_radius=jrad)
                    else:
                        jprob = float((perturb_params or {}).get("jitter_prob", 0.0))
                        jrad  = int((perturb_params or {}).get("jitter_radius", 1))
                        population, cell_map = _drip_seed_one_type(population, cell_map, lab, W_immune, n_lab, genome, counter_like, jitter_prob=jprob, jitter_radius=jrad)
                elif group == "Epithelial":
                    population, cell_map = _drip_seed_one_type(population, cell_map, lab, W_epi, n_lab, genome, counter_like)
                else:  # Stromal
                    population, cell_map = _drip_seed_one_type(population, cell_map, lab, W_stromal, n_lab, genome, counter_like)
            # Report the final occupancy
            new_occ = int(cell_map.sum())
            print("\nFinal drip fill applied. Occupancy: {:.2f}%".format((new_occ/total_slots)*100.0))
    print()
    lineage.pop(-1, None)

    if debug:
        print("debug mode:debug_stage1_grow")
        if output_dir is None:
            output_dir = os.getcwd()
        tmp_dir = os.path.join(output_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        dbg_dir = os.path.join(tmp_dir, "debug_stage1_grow")
        os.makedirs(dbg_dir, exist_ok=True)

        # 1) Histogram of the per-cell number of genes with copy != 2
        cnv_counts = [
            sum((arr != 1).sum() for arr in cell.genome.get_chr_gene_count().values())
            for cell in population
        ]
        plt.hist(cnv_counts, bins=40)
        plt.xlabel("#genes CN≠2"); plt.ylabel("#cells")
        plt.title("Per-cell CNV load after grow_tissue()")
        plt.savefig(os.path.join(dbg_dir, "cnv_load_hist.png"))
        plt.close()

        # 2) Save a lightweight population pickle
        with gzip.open(os.path.join(dbg_dir, "population.pkl.gz"), "wb") as pkl:
            pickle.dump({"population": population,
                        "lineage": lineage}, pkl)


    return population, lineage, cell_map


def add_background(
    population: List[Cell],
    genome: Genome,
    cell_map: np.ndarray,
    normal_celltype_proportions: Dict[str, float] = None,
    fraction_drop: float = 0.05, #for tumor
    fill_prob: float = 0.7,
    normal_action_rates: Tuple[float, float, float, float] = (0.93, 0.01, 0.05, 0.01), #stay, spawn,  move,  death
) -> Tuple[List[Cell], np.ndarray]:

    pop_size = len(population)
    # If some cells are to be removed, pick them at random and update cell_map
    if fraction_drop > 0:
        drop = np.random.choice(
            pop_size, replace=False, size=int(pop_size * fraction_drop)
        )

        new_cell_map = copy.deepcopy(cell_map)
        for i in drop:
            new_cell_map[int(population[i].crd[0]), int(population[i].crd[1])] -= 1
        # Drop the removed cells from the population
        population = [c for k, c in enumerate(population) if k not in drop]
    else:
        drop = np.array([], dtype=int)
        new_cell_map = copy.deepcopy(cell_map)

    #counter = ut.Counter(max([x.id for x in population]))
    max_id = max(x.id for x in population)
    counter = ut.Counter(max_id + 1)

    # No longer share a single deep copy; each normal cell gets its own lightweight view
    zero_mut_rate = np.zeros(2, dtype=float)
    normal_population = []

    labels = list(normal_celltype_proportions.keys())
    weights = np.array(list(normal_celltype_proportions.values()))

    # Find the positions in cell_map with no cell (indices whose value is 0)
    row, col = np.where(new_cell_map == 0)
    n_empty      = len(row)
    keep_mask    = np.random.rand(n_empty) < fill_prob
    row, col     = row[keep_mask], col[keep_mask]

    for x, y in zip(row, col):
        label_choice = np.random.choice(labels, p=weights)
        normal_population.append(
            Cell(
                x=x,
                y=y,
                genome=genome.create_view() if hasattr(genome, "create_view") else copy.deepcopy(genome),
                counter=counter,
                action_rates=np.repeat(np.array(normal_action_rates).reshape(4, 1), 2, axis=1),
                mutation_rate=zero_mut_rate,
                clone_id=None,
                cell_type=label_choice,
            )
        )

    population = population + normal_population


    return population, drop
