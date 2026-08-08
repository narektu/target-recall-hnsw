import math
import random
import pytest
from target_recall_hnsw.level import LevelSampler


def test_level_sampler_reproducibility():
    """Ensure identical seeds produce the exact same sequence of levels."""
    sampler1 = LevelSampler(M=2, seed=42)
    sampler2 = LevelSampler(M=2, seed=42)

    seq1 = [sampler1.sample() for _ in range(50)]
    seq2 = [sampler2.sample() for _ in range(50)]

    assert seq1 == seq2


def test_level_distribution():
    """Verify that levels follow the expected geometric distribution based on M."""
    M = 2
    num_samples = 20_000
    sampler = LevelSampler(M=M, seed=123)

    counts = {}
    for _ in range(num_samples):
        lvl = sampler.sample()
        counts[lvl] = counts.get(lvl, 0) + 1

    # Level 0 should occur more frequently than level 1, level 1 more than level 2, etc.
    assert counts.get(0, 0) > counts.get(1, 0)
    assert counts.get(1, 0) > counts.get(2, 0)

    # Check that the ratio between level 0 and level 1 counts approximates M (2.0)
    if 1 in counts and 0 in counts:
        ratio = counts[0] / counts[1]
        assert 1.6 < ratio < 2.4
