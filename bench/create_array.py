"""Write the benchmark array to disk as a Zarr v3 store (via zarr-python)."""
import shutil

import numpy as np
import zarr
from zarr.codecs import ZstdCodec

from bench import config
from bench.data import make_data


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if config.STORE_PATH.exists():
        shutil.rmtree(config.STORE_PATH)

    data = make_data()
    arr = zarr.create_array(
        store=str(config.STORE_PATH),
        shape=config.SHAPE,
        chunks=config.CHUNKS,
        dtype=config.DTYPE,
        compressors=ZstdCodec(level=config.ZSTD_LEVEL),
        fill_value=0,
        zarr_format=3,
    )
    arr[:] = data

    # Sanity check: read everything back and compare.
    back = zarr.open_array(str(config.STORE_PATH), mode="r")[:]
    if not np.array_equal(back, data):
        raise RuntimeError("Round-trip check failed: written array != source data")

    n_chunks = sum(1 for p in config.STORE_PATH.rglob("*") if p.is_file() and p.name != "zarr.json")
    print(f"Wrote {config.STORE_PATH}")
    print(f"  shape={config.SHAPE} chunks={config.CHUNKS} dtype={config.DTYPE} "
          f"codec=zstd(level={config.ZSTD_LEVEL}) chunk files={n_chunks}")


if __name__ == "__main__":
    main()
