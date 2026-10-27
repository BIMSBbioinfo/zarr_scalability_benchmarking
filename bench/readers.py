"""One reader per method. Each returns read(r0, r1, c0, c1) -> np.ndarray
plus a short description of what was actually opened.

Each method is run in its own process (see run_all.py), so global config such
as zarr's codec pipeline cannot leak between methods.
"""
from typing import Callable

import numpy as np

from bench import config
import importlib.metadata

ReadFn = Callable[[int, int, int, int], np.ndarray]


def _pipeline_name(arr) -> str:
    pipeline = getattr(getattr(arr, "_async_array", None), "codec_pipeline", None)
    return type(pipeline).__name__ if pipeline is not None else "unknown"


def open_zarr_python() -> tuple[ReadFn, str]:
    import zarr

    arr = zarr.open_array(str(config.STORE_PATH), mode="r")

    def read(r0, r1, c0, c1):
        return arr[r0:r1, c0:c1]

    return read, f"zarr {zarr.__version__}, pipeline={_pipeline_name(arr)}"


def open_zarrs_python() -> tuple[ReadFn, str]:
    import zarr
    import zarrs

    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
    arr = zarr.open_array(str(config.STORE_PATH), mode="r")

    name = _pipeline_name(arr)
    if name != "unknown" and "Zarrs" not in name:
        raise RuntimeError(f"Expected the zarrs codec pipeline, got {name}")

    def read(r0, r1, c0, c1):
        return arr[r0:r1, c0:c1]

    version = getattr(zarrs, "__version__", "?")
    return read, f"zarr {zarr.__version__} + zarrs {version}, pipeline={name}"


def open_tensorstore() -> tuple[ReadFn, str]:
    import tensorstore as ts

    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(config.STORE_PATH)},
        # No in-process chunk cache, so repeats don't hit TensorStore's cache.
        "context": {"cache_pool": {"total_bytes_limit": 0}},
    }
    arr = ts.open(spec, read=True, open=True).result()

    def read(r0, r1, c0, c1):
        return arr[r0:r1, c0:c1].read().result()

    return read, f"tensorstore {importlib.metadata.version("tensorstore")}, driver=zarr3"


OPENERS = {
    "zarr-python": open_zarr_python,
    "zarrs-python": open_zarrs_python,
    "tensorstore": open_tensorstore,
}
