# zarr-read-bench

Benchmarks reading rectangular subsets of a 2D Zarr v3 array with three readers:

- **zarr-python**: zarr-python v3 with its default codec pipeline
- **zarrs-python**: zarr-python with the Rust `zarrs.ZarrsCodecPipeline`
- **tensorstore**: TensorStore's `zarr3` driver

All methods read the identical list of subsets (saved to `data/subsets.json`) in the
identical order, and every read is checked against the ground-truth data.

## Setup

```bash
conda env create -f environment.yml
conda activate zarr-read-bench
```

## Run (from the repo root)

```bash
python -m bench.create_array   # 10000x10000 uint8, 1000x1000 chunks, zstd -> data/array.zarr
python -m bench.make_subsets   # 200 random windows -> data/subsets.json
python -m bench.run_all        # options: --repeats 3 --warmup 1 --methods zarr-python tensorstore
```

## Example Output

```python
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
```

```
Per-read time across all subsets and repeats:
              median_ms  mean_ms  p05_ms  p95_ms  total_s  n_reads  throughput_MB_s
method                                                                             
tensorstore       0.283    0.308   0.144   0.533    0.185      600          920.570
zarrs-python      0.896    0.985   0.487   1.660    0.591      600          287.903
zarr-python       2.834    3.179   1.104   6.116    1.908      600           89.170
```

## Outputs (`results/`)

| File | Contents |
|---|---|
| `raw_<method>.csv` | Every timed read: method, repeat, order, subset coords, bytes, `time_ns`, crc32 |
| `per_subset_ms.csv` | One row per subset, median time over repeats per method |
| `summary.csv` | Per method: median/mean/p05/p95 ms per read, total time, throughput |
| `run_info.json` | Package versions, platform, CPU count, and the config used |

## Configuration

All fixed parameters (shape, chunks, codec, seeds, number and size range of subsets)
live in `bench/config.py`. After changing them, rerun `create_array` and `make_subsets`.

## Methodology notes

- Each method runs in a **fresh subprocess**, so zarr's global codec-pipeline setting
  and any in-process caches can't leak between methods.
- The timed region is only the read call that returns a NumPy array. Shape checks
  and checksums happen outside it. GC is disabled during each timed pass.
- TensorStore's chunk cache is set to 0 bytes so repeats don't hit an in-process cache.
- The **OS page cache** is shared: after the warmup pass, all methods read from
  warm cache. For cold-cache numbers on Linux, use `--warmup 0 --repeats 1 --methods <one>`
  and drop caches before each method: `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`.
- Each library uses its default threading. zarrs and TensorStore are multithreaded
  in native code; zarr-python decodes via its async/thread pool.
- The data is uniform random bytes, which zstd can't compress. That makes this a
  worst case for decompression cost. Edit `bench/data.py` for more realistic content.
