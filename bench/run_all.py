"""Run every method in its own fresh process, then verify and summarise."""
import argparse
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

from bench import config, summarize


def record_environment(args) -> None:
    def v(pkg):
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None

    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": {p: v(p) for p in ("numpy", "pandas", "zarr", "zarrs", "tensorstore")},
        "array": {"shape": config.SHAPE, "chunks": config.CHUNKS, "dtype": config.DTYPE,
                  "zstd_level": config.ZSTD_LEVEL, "data_seed": config.DATA_SEED},
        "subsets": {"n": config.N_SUBSETS, "seed": config.SUBSET_SEED,
                    "edge_range": [config.SUBSET_MIN_EDGE, config.SUBSET_MAX_EDGE]},
        "run": {"methods": args.methods, "repeats": args.repeats, "warmup": args.warmup},
    }
    (config.RESULTS_DIR / "run_info.json").write_text(json.dumps(info, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(config.METHODS), choices=config.METHODS)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    if not config.STORE_PATH.exists() or not config.SUBSETS_PATH.exists():
        sys.exit("Missing data. Run: python -m bench.create_array && python -m bench.make_subsets")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for m in args.methods:
        subprocess.run([sys.executable, "-m", "bench.run_method", "--method", m,
                        "--repeats", str(args.repeats), "--warmup", str(args.warmup)],
                       check=True, cwd=config.ROOT)

    record_environment(args)
    summarize.main(args.methods)


if __name__ == "__main__":
    main()
