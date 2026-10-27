"""Verify correctness across methods and summarise per-subset timings."""
import zlib

import numpy as np
import pandas as pd

from bench import config
from bench.data import make_data
from bench.make_subsets import load_subsets


def load_raw(methods) -> pd.DataFrame:
    frames = [pd.read_csv(config.RESULTS_DIR / f"raw_{m}.csv") for m in methods]
    return pd.concat(frames, ignore_index=True)


def verify(df: pd.DataFrame) -> None:
    """Every method's every read must match the ground-truth data."""
    data = make_data()
    ref = {s["id"]: zlib.crc32(np.ascontiguousarray(data[s["r0"]:s["r1"], s["c0"]:s["c1"]]))
           for s in load_subsets()}
    bad = df[df["crc32"] != df["subset_id"].map(ref)]
    if len(bad):
        print(bad[["method", "repeat", "subset_id"]].to_string(index=False))
        raise RuntimeError(f"{len(bad)} reads did not match the reference data")

    # Same subsets, same order, for every method and repeat.
    orders = df.groupby(["method", "repeat"])["subset_id"].apply(tuple)
    if orders.nunique() != 1:
        raise RuntimeError("Methods/repeats did not read the same subsets in the same order")
    print(f"Verified {len(df)} reads: all methods returned identical, correct data "
          f"in identical order.")


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.assign(time_ms=df["time_ns"] / 1e6)

    # Per subset: median over repeats, one column per method.
    per_subset = (df.groupby(["subset_id", "method"])["time_ms"].median()
                    .unstack("method"))
    meta = df.drop_duplicates("subset_id").set_index("subset_id")[["r0", "r1", "c0", "c1", "n_bytes"]]
    per_subset = meta.join(per_subset.add_suffix("_ms")).sort_index()

    summary = df.groupby("method").agg(
        median_ms=("time_ms", "median"),
        mean_ms=("time_ms", "mean"),
        p05_ms=("time_ms", lambda x: x.quantile(0.05)),
        p95_ms=("time_ms", lambda x: x.quantile(0.95)),
        total_s=("time_ms", lambda x: x.sum() / 1e3),
        n_reads=("time_ms", "size"),
    )
    total_mb = df.groupby("method")["n_bytes"].sum() / 1e6
    summary["throughput_MB_s"] = total_mb / summary["total_s"]
    summary = summary.sort_values("median_ms")
    return per_subset, summary


def main(methods=config.METHODS) -> None:
    df = load_raw(methods)
    verify(df)
    per_subset, summary = summarize(df)

    per_subset.to_csv(config.RESULTS_DIR / "per_subset_ms.csv")
    summary.to_csv(config.RESULTS_DIR / "summary.csv")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
        print("\nPer-read time across all subsets and repeats:")
        print(summary.to_string())
    print(f"\nPer-subset medians: {config.RESULTS_DIR / 'per_subset_ms.csv'}")


if __name__ == "__main__":
    main()
