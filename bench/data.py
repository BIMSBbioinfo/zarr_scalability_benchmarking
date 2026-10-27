"""Deterministic array content, so any step can regenerate the ground truth."""
import numpy as np

from bench import config


def make_data() -> np.ndarray:
    rng = np.random.default_rng(config.DATA_SEED)
    return rng.integers(0, 256, size=config.SHAPE, dtype=np.uint8)
