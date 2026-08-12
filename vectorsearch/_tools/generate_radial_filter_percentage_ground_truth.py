"""
Generate a filtered radial search dataset with per-percentage ground truth.

Each document gets boolean filter attribute fields (e.g. filter10pct = 'true'
or 'false'). For each percentage, exactly that fraction of documents is marked
'true', chosen uniformly at random. Filtering with a term query
(e.g. {"term": {"filter10pct": "true"}}) therefore passes exactly that
percentage of documents, scattered randomly across the HNSW graph.

For each percentage, ground truth is computed by brute force over the passing
documents only: the top take_n nearest passing docs per query, plus per-query
radial search thresholds (distance to the k-th nearest passing neighbor,
converted per engine). This mirrors the unfiltered radial per-query benchmark:
at benchmark time query_k selects the ground truth depth (neighbors[:k]) and
the radial threshold (threshold[k-1]).

Usage:
    python generate_radial_filter_percentage_ground_truth.py \
        --input cohere-1m.hdf5 \
        --output cohere-1m-radial-filter-percentage.hdf5 \
        --space-type innerproduct \
        --percentages 0.1 1 10 60 \
        --take-n 1000

Output HDF5 (for --percentages 0.1 1 10 60):
    train:                       (N, dim)     float32  — corpus vectors (copied from input)
    test:                        (Q, dim)     float32  — query vectors (copied from input)
    attributes:                  (N, P)       |S8      — 'true'/'false' per percentage field
    neighbors_01pct:             (Q, take_n)  int64    — top take_n passing docs, nearest first
    distances_01pct:             (Q, take_n)  float32  — raw distances for those neighbors
    faiss_max_distance_01pct:    (Q, take_n)  float32  — engine threshold values
    faiss_min_score_01pct:       (Q, take_n)  float32
    lucene_max_distance_01pct:   (Q, take_n)  float32
    lucene_min_score_01pct:      (Q, take_n)  float32
    ... same six datasets per percentage, suffixed _1pct, _10pct, _60pct

Attribute column order matches --percentages order and maps to index fields:
    0.1 -> filter01pct, 1 -> filter1pct, 10 -> filter10pct, 60 -> filter60pct

Requires take_n <= passing docs at the smallest percentage.
"""

import argparse
import h5py
import numpy as np
import shutil
import sys
import time


def pct_suffix(pct):
    """0.1 -> '01pct', 1 -> '1pct', 10 -> '10pct', 60 -> '60pct'"""
    if pct < 1:
        return f"0{str(pct).replace('0.', '')}pct"
    return f"{int(pct)}pct"


def calculate_distances_batch(queries, corpus, space_type):
    if space_type == "l2":
        q_norms = np.sum(queries ** 2, axis=1, keepdims=True)
        c_norms = np.sum(corpus ** 2, axis=1, keepdims=True).T
        dots = queries @ corpus.T
        return q_norms + c_norms - 2 * dots
    elif space_type == "innerproduct":
        return -(queries @ corpus.T)
    elif space_type == "cosine":
        q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        c_norms = np.linalg.norm(corpus, axis=1, keepdims=True).T
        dots = queries @ corpus.T
        return 1 - dots / (q_norms * c_norms)
    else:
        raise ValueError(f"Unsupported space type: {space_type}")


def raw_distance_to_opensearch_score(distances, space_type):
    if space_type == "l2":
        return 1.0 / (1.0 + distances)
    elif space_type == "innerproduct":
        return np.where(distances >= 0, 1.0 / (1.0 + distances), -distances + 1.0)
    elif space_type == "cosine":
        return (2.0 - distances) / 2.0
    else:
        raise ValueError(f"Unsupported space type: {space_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-percentage filtered radial ground truth for vector search benchmarks")
    parser.add_argument("--input", required=True, help="Input HDF5 with train/test")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument("--space-type", required=True, choices=["l2", "innerproduct", "cosine"])
    parser.add_argument("--percentages", type=float, nargs="+", default=[0.1, 1, 10, 60],
                        help="Filter percentages (default: 0.1 1 10 60)")
    parser.add_argument("--take-n", type=int, default=1000,
                        help="Neighbors stored per query per percentage (default: 1000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-batch-size", type=int, default=100)
    args = parser.parse_args()

    with h5py.File(args.input, "r") as f_in:
        print("Loading dataset...")
        train = f_in["train"][:]
        test = f_in["test"][:]
    num_docs, num_queries = train.shape[0], test.shape[0]
    print(f"  train: {train.shape}, test: {test.shape}")

    suffixes = [pct_suffix(p) for p in args.percentages]
    print(f"Percentages: {args.percentages} -> fields filter{{{', filter'.join(suffixes)}}}")

    # Validate take_n against smallest percentage
    for pct in args.percentages:
        n_pass = int(round(pct / 100.0 * num_docs))
        if n_pass < args.take_n:
            print(f"ERROR: {pct}% passes only {n_pass} docs < take_n={args.take_n}. "
                  f"Reduce --take-n or raise the percentage.")
            sys.exit(1)

    # Assign attributes: for each percentage, mark exactly n_pass random docs 'true'
    rng = np.random.default_rng(args.seed)
    attributes = np.full((num_docs, len(args.percentages)), b"false", dtype="|S8")
    passing_ids = {}
    for col, pct in enumerate(args.percentages):
        n_pass = int(round(pct / 100.0 * num_docs))
        chosen = rng.permutation(num_docs)[:n_pass]
        attributes[chosen, col] = b"true"
        passing_ids[col] = np.sort(chosen)
        print(f"  filter{suffixes[col]}: exactly {n_pass} docs marked true")

    # Per-percentage output arrays
    out = {}
    for s in suffixes:
        out[s] = {
            "neighbors": np.empty((num_queries, args.take_n), dtype=np.int64),
            "distances": np.empty((num_queries, args.take_n), dtype=np.float32),
        }

    print(f"\nComputing ground truth for {num_queries} queries "
          f"x {len(args.percentages)} percentages (take_n={args.take_n})...")
    t0 = time.time()
    for batch_start in range(0, num_queries, args.query_batch_size):
        batch_end = min(batch_start + args.query_batch_size, num_queries)
        if batch_start % 1000 == 0:
            print(f"  {batch_start}/{num_queries} ({time.time()-t0:.0f}s)")
        # One distance computation per batch, reused for all percentages
        batch_dists = calculate_distances_batch(test[batch_start:batch_end], train, args.space_type)

        for col, s in enumerate(suffixes):
            ids = passing_ids[col]
            sub = batch_dists[:, ids]                      # distances to passing docs only
            for i in range(batch_end - batch_start):
                d = sub[i]
                if args.take_n < len(d):
                    idx = np.argpartition(d, args.take_n - 1)[:args.take_n]
                    idx = idx[np.argsort(d[idx])]
                else:
                    idx = np.argsort(d)
                out[s]["neighbors"][batch_start + i] = ids[idx]
                out[s]["distances"][batch_start + i] = d[idx]
    print(f"Done in {time.time()-t0:.0f}s")

    # Write output
    shutil.copy2(args.input, args.output)
    with h5py.File(args.output, "a") as f_out:
        def write(name, data):
            if name in f_out:
                del f_out[name]
            f_out.create_dataset(name, data=data)

        write("attributes", attributes)
        for s in suffixes:
            dist = out[s]["distances"]
            write(f"neighbors_{s}", out[s]["neighbors"])
            write(f"distances_{s}", dist)
            write(f"faiss_max_distance_{s}", dist.copy())
            lucene_dist = -dist if args.space_type == "innerproduct" else dist.copy()
            write(f"lucene_max_distance_{s}", lucene_dist)
            score = raw_distance_to_opensearch_score(dist, args.space_type).astype(np.float32)
            write(f"faiss_min_score_{s}", score)
            write(f"lucene_min_score_{s}", score)

        f_out.attrs["space_type"] = args.space_type
        f_out.attrs["percentages"] = args.percentages
        f_out.attrs["take_n"] = args.take_n
        f_out.attrs["seed"] = args.seed

    print(f"\nWritten to {args.output}")
    print(f"  attributes: {attributes.shape}")
    for s in suffixes:
        print(f"  neighbors_{s} / distances_{s} / thresholds_{s}: ({num_queries}, {args.take_n})")
    print(f"\nBenchmark filter body example: {{\"term\": {{\"filter{suffixes[-1]}\": \"true\"}}}}")


if __name__ == "__main__":
    main()
