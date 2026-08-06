"""
Generate a filtered vector search dataset with ground truth.

Assigns each document a shuffled ID (0 to N-1) stored as field "id".
Filtering on id <= filter_ratio * N passes exactly filter_ratio fraction
of documents, scattered randomly across the HNSW graph (no locality bias).

Then brute-force computes k nearest neighbors among passing docs and
computes engine-specific radial search thresholds.

Usage:
    python generate_filter_ground_truth.py \
        --input cohere-1m.hdf5 \
        --output cohere-1m-filter-10pct.hdf5 \
        --space-type innerproduct \
        --filter-ratio 0.1

    # To benchmark with this dataset, use filter:
    # {"range": {"id": {"lte": <filter_ratio * num_docs>}}}
    # e.g. for 10% of 1M docs: {"range": {"id": {"lte": 100000}}}

Output HDF5 contains:
    train:               (N, dim) float32  — all vectors
    test:                (Q, dim) float32  — query vectors
    id:                  (N,)     int64    — shuffled doc IDs
    neighbors:           (Q, k)   int64    — ground truth doc indices (sorted nearest-first)
    distances:           (Q, k)   float32  — raw distances
    faiss_max_distance:  (Q, k)   float32  — per-query Faiss radial threshold
    faiss_min_score:     (Q, k)   float32  — per-query Faiss min_score threshold
    lucene_max_distance: (Q, k)   float32  — per-query Lucene radial threshold
    lucene_min_score:    (Q, k)   float32  — per-query Lucene min_score threshold
"""

import argparse
import h5py
import numpy as np
import shutil
import sys


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
        description="Generate filtered vector search dataset with ground truth"
    )
    parser.add_argument("--input", required=True, help="Input HDF5 dataset path")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument("--space-type", required=True, choices=["l2", "innerproduct", "cosine"])
    parser.add_argument("--filter-ratio", type=float, required=True,
                        help="Fraction of docs passing filter (e.g. 0.1 = 10%%)")
    parser.add_argument("--k", type=int, default=1000,
                        help="Number of neighbors to compute (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    parser.add_argument("--query-batch-size", type=int, default=100,
                        help="Queries per batch (default: 100)")

    args = parser.parse_args()

    if not 0 < args.filter_ratio <= 1.0:
        print("ERROR: --filter-ratio must be between 0 and 1")
        sys.exit(1)

    print(f"Input:        {args.input}")
    print(f"Output:       {args.output}")
    print(f"Space type:   {args.space_type}")
    print(f"Filter ratio: {args.filter_ratio} ({args.filter_ratio * 100:.1f}%)")
    print(f"k:            {args.k}")
    print(f"Seed:         {args.seed}")
    print()

    with h5py.File(args.input, "r") as f_in:
        print("Loading dataset...")
        train = f_in["train"][:]
        test = f_in["test"][:]
        print(f"  train: {train.shape}")
        print(f"  test:  {test.shape}")

    num_docs = train.shape[0]
    filter_max = int(args.filter_ratio * num_docs)
    print(f"\nFilter max: {filter_max} / {num_docs} ({100 * filter_max / num_docs:.2f}%)")

    # Assign shuffled IDs
    rng = np.random.default_rng(args.seed)
    shuffled_ids = np.arange(num_docs, dtype=np.int64)
    rng.shuffle(shuffled_ids)

    # Docs with shuffled_id <= filter_max pass the filter
    mask = shuffled_ids <= filter_max
    candidate_doc_indices = np.where(mask)[0]
    print(f"Candidates passing filter: {len(candidate_doc_indices)}")

    if len(candidate_doc_indices) < args.k:
        print(f"ERROR: Only {len(candidate_doc_indices)} candidates pass filter, need at least {args.k}")
        sys.exit(1)

    candidate_vecs = train[candidate_doc_indices]
    num_queries = test.shape[0]

    neighbors = np.empty((num_queries, args.k), dtype=np.int64)
    distances = np.empty((num_queries, args.k), dtype=np.float32)

    print(f"\nComputing {args.k} nearest filtered neighbors for {num_queries} queries...")
    for batch_start in range(0, num_queries, args.query_batch_size):
        batch_end = min(batch_start + args.query_batch_size, num_queries)
        if batch_start % 1000 == 0:
            print(f"  {batch_start}/{num_queries}")

        batch_queries = test[batch_start:batch_end]
        batch_dists = calculate_distances_batch(batch_queries, candidate_vecs, args.space_type)

        for i in range(batch_end - batch_start):
            dists = batch_dists[i]
            top_k_idx = np.argpartition(dists, args.k - 1)[:args.k]
            top_k_idx = top_k_idx[np.argsort(dists[top_k_idx])]
            neighbors[batch_start + i] = candidate_doc_indices[top_k_idx]
            distances[batch_start + i] = dists[top_k_idx]

    print("Done computing ground truth.")

    # Compute engine-specific thresholds
    faiss_max_distance = distances.copy()
    lucene_max_distance = -distances if args.space_type == "innerproduct" else distances.copy()
    min_score = raw_distance_to_opensearch_score(distances, args.space_type)

    # Print stats
    k_values = [k for k in [100, 200, 500, 1000] if k <= args.k]
    print(f"\nThreshold stats by k (space_type={args.space_type}):")
    for k_val in k_values:
        col_f = faiss_max_distance[:, k_val - 1]
        print(f"  k={k_val}: faiss_max_dist median={np.median(col_f):.4f}")

    # Write output
    shutil.copy2(args.input, args.output)

    with h5py.File(args.output, "a") as f_out:
        def write_dataset(name, data):
            if name in f_out:
                del f_out[name]
            f_out.create_dataset(name, data=data)

        write_dataset("id", shuffled_ids)
        write_dataset("neighbors", neighbors)
        write_dataset("distances", distances.astype(np.float32))
        write_dataset("faiss_max_distance", faiss_max_distance.astype(np.float32))
        write_dataset("faiss_min_score", min_score.astype(np.float32))
        write_dataset("lucene_max_distance", lucene_max_distance.astype(np.float32))
        write_dataset("lucene_min_score", min_score.astype(np.float32))

        f_out.attrs["space_type"] = args.space_type
        f_out.attrs["enriched_k"] = args.k
        f_out.attrs["filter_ratio"] = args.filter_ratio
        f_out.attrs["filter_max"] = filter_max
        f_out.attrs["seed"] = args.seed

    print(f"\nWritten to {args.output}")
    print(f"  id:        {shuffled_ids.shape}")
    print(f"  neighbors: {neighbors.shape}")
    print(f"\nBenchmark filter body:")
    print(f'  {{"range": {{"id": {{"lte": {filter_max}}}}}}}')


if __name__ == "__main__":
    main()
