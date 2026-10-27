"""Fixed benchmark configuration. Edit here; everything else reads from this."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
STORE_PATH = DATA_DIR / "array.zarr"
SUBSETS_PATH = DATA_DIR / "subsets.json"

# Array
SHAPE = (10_000, 10_000)
CHUNKS = (200, 200)
DTYPE = "uint8"
DATA_SEED = 0
ZSTD_LEVEL = 3

# Subsets (rectangular windows, generated once and saved to disk)
N_SUBSETS = 200
SUBSET_SEED = 1
SUBSET_MIN_EDGE = 100
SUBSET_MAX_EDGE = 1000

METHODS = ("zarr-python", "zarrs-python", "tensorstore")
