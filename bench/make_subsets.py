"""Generate the subset list once and save it, so every method reads the
exact same windows in the exact same order."""
import json

import numpy as np

from bench import config


def main() -> None:
    rng = np.random.default_rng(config.SUBSET_SEED)
    n_rows, n_cols = config.SHAPE
    subsets = []
    for i in range(config.N_SUBSETS):
        h = int(rng.integers(config.SUBSET_MIN_EDGE, config.SUBSET_MAX_EDGE + 1))
        w = int(rng.integers(config.SUBSET_MIN_EDGE, config.SUBSET_MAX_EDGE + 1))
        r0 = int(rng.integers(0, n_rows - h + 1))
        c0 = int(rng.integers(0, n_cols - w + 1))
        subsets.append({"id": i, "r0": r0, "r1": r0 + h, "c0": c0, "c1": c0 + w})

    payload = {
        "shape": list(config.SHAPE),
        "chunks": list(config.CHUNKS),
        "seed": config.SUBSET_SEED,
        "subsets": subsets,
    }
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.SUBSETS_PATH.write_text(json.dumps(payload, indent=1))
    print(f"Wrote {len(subsets)} subsets to {config.SUBSETS_PATH}")


def load_subsets() -> list[dict]:
    payload = json.loads(config.SUBSETS_PATH.read_text())
    if tuple(payload["shape"]) != config.SHAPE:
        raise RuntimeError("subsets.json was made for a different array shape; regenerate it")
    return payload["subsets"]


if __name__ == "__main__":
    main()
