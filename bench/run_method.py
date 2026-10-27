"""Benchmark a single method. Normally invoked by run_all.py in a fresh process."""
import argparse
import csv
import gc
import time
import zlib

import numpy as np

from bench import config
from bench.make_subsets import load_subsets
from bench.readers import OPENERS

FIELDS = ["method", "repeat", "order", "subset_id", "r0", "r1", "c0", "c1",
          "n_bytes", "time_ns", "crc32"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(OPENERS))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1, help="untimed passes over all subsets")
    args = ap.parse_args()

    subsets = load_subsets()

    t0 = time.perf_counter_ns()
    read, desc = OPENERS[args.method]()
    open_ms = (time.perf_counter_ns() - t0) / 1e6
    print(f"[{args.method}] {desc} (open: {open_ms:.1f} ms)")

    for _ in range(args.warmup):
        for s in subsets:
            read(s["r0"], s["r1"], s["c0"], s["c1"])

    rows = []
    for rep in range(args.repeats):
        gc.collect()
        gc.disable()
        try:
            for order, s in enumerate(subsets):
                t_start = time.perf_counter_ns()
                out = read(s["r0"], s["r1"], s["c0"], s["c1"])
                t_end = time.perf_counter_ns()

                # Everything below is outside the timed region.
                expected = (s["r1"] - s["r0"], s["c1"] - s["c0"])
                if out.shape != expected or out.dtype != np.uint8:
                    raise RuntimeError(f"subset {s['id']}: got {out.shape}/{out.dtype}, "
                                       f"expected {expected}/uint8")
                rows.append({
                    "method": args.method, "repeat": rep, "order": order,
                    "subset_id": s["id"], "r0": s["r0"], "r1": s["r1"],
                    "c0": s["c0"], "c1": s["c1"], "n_bytes": out.nbytes,
                    "time_ns": t_end - t_start,
                    "crc32": zlib.crc32(np.ascontiguousarray(out)),
                })
                del out
        finally:
            gc.enable()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / f"raw_{args.method}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{args.method}] wrote {len(rows)} timings to {out_path}")


if __name__ == "__main__":
    main()
