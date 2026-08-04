"""
Generate ground truth for filtered vector search benchmarks.

Given an HDF5 dataset with vectors and attributes, this script:
1. Applies a filter to identify candidate vectors
2. Brute-force computes the k nearest neighbors among filtered candidates
3. Computes engine-specific radial search thresholds

Usage:
    python generate_filtered_ground_truth.py \
        --input cohere-1m-restrictive-filters.hdf5 \
        --output cohere-1m-radial-filter-restrictive.hdf5 \
        --space-type innerproduct \
        --filter-spec '{"age_gte": 30, "age_lte": 60, "taste": "bitter", "colors": ["blue", "green"]}'
"""

import argparse
import h5py
import json
import numpy as np
import sys


def apply_filter(attributes, filter_spec):
    """Apply filter to attributes and return mask of passing doc indices.

    Args:
        attributes: (N, 3) array with columns [color, taste, age] as byte strings
        filter_spec: dict with filter conditions

    Returns:
        boolean mask of shape (N,)
    """
    mask = np.ones(len(attributes), dtype=bool)

    if "age_gte" in filter_spec or "age_lte" in filter_spec:
        age = attributes[:, 2].astype(int)
        if "age_gte" in filter_spec:
            mask &= age >= filter_spec["age_gte"]
        if "age_lte" in filter_spec:
            mask &= age <= filter_spec["age_lte"]

    if "taste" in filter_spec:
        mask &= attributes[:, 1] == filter_spec["taste"].encode()

    if "colors" in filter_spec:
        color_mask = np.zeros(len(attributes), dtype=bool)
        for c in filter_spec["colors"]:
            color_mask |= attributes[:, 0] == c.encode()
        mask &= color_mask

    return mask


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
        description="Generate filtered ground truth for vector search benchmarks"
    )
    parser.add_argument("--input", required=True, help="Input HDF5 dataset path")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument("--space-type", required=True, choices=["l2", "innerproduct", "cosine"])
    parser.add_argument("--filter-spec", required=True,
                        help="JSON filter specification")
    parser.add_argument("--k", type=int, default=1000,
                        help="Number of neighbors to compute (default: 1000)")
    parser.add_argument("--query-batch-size", type=int, default=100,
                        help="Queries to process per batch (default: 100)")

    args = parser.parse_args()
    filter_spec = json.loads(args.filter_spec)

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Space type: {args.space_type}")
    print(f"Filter: {filter_spec}")
    print(f"k: {args.k}")
    print()

    with h5py.File(args.input, "r") as f_in:
        print("Loading dataset...")
        train = f_in["train"][:]
        test = f_in["test"][:]
        attributes = f_in["attributes"][:]
        print(f"  train: {train.shape}")
        print(f"  test: {test.shape}")
        print(f"  attributes: {attributes.shape}")

    mask = apply_filter(attributes, filter_spec)
    candidate_ids = np.where(mask)[0]
    print(f"\nCandidates passing filter: {len(candidate_ids)} / {len(attributes)}")

    if len(candidate_ids) < args.k:
        print(f"ERROR: Only {len(candidate_ids)} candidates pass filter, need at least {args.k}")
        sys.exit(1)

    candidate_vecs = train[candidate_ids]
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
            neighbors[batch_start + i] = candidate_ids[top_k_idx]
            distances[batch_start + i] = dists[top_k_idx]

    print("Done computing ground truth.")

    # Compute engine-specific thresholds
    faiss_max_distance = distances.copy()
    if args.space_type == "innerproduct":
        lucene_max_distance = -distances
    else:
        lucene_max_distance = distances.copy()
    min_score = raw_distance_to_opensearch_score(distances, args.space_type)

    # Print stats
    k_values = [k for k in [100, 200, 500, 1000] if k <= args.k]
    print(f"\nThreshold stats by k (space_type={args.space_type}):")
    for k_val in k_values:
        col_f = faiss_max_distance[:, k_val - 1]
        col_l = lucene_max_distance[:, k_val - 1]
        print(f"  k={k_val}: faiss_max_dist median={np.median(col_f):.4f}, "
              f"lucene_max_dist median={np.median(col_l):.4f}")

    # Write output
    import shutil
    shutil.copy2(args.input, args.output)

    with h5py.File(args.output, "a") as f_out:
        def write_dataset(name, data):
            if name in f_out:
                del f_out[name]
            f_out.create_dataset(name, data=data)

        write_dataset("neighbors", neighbors)
        write_dataset("distances", distances.astype(np.float32))
        write_dataset("faiss_max_distance", faiss_max_distance.astype(np.float32))
        write_dataset("faiss_min_score", min_score.astype(np.float32))
        write_dataset("lucene_max_distance", lucene_max_distance.astype(np.float32))
        write_dataset("lucene_min_score", min_score.astype(np.float32))

        f_out.attrs["space_type"] = args.space_type
        f_out.attrs["enriched_k"] = args.k
        f_out.attrs["filter_spec"] = json.dumps(filter_spec)

    print(f"\nWritten to {args.output}")
    print(f"  neighbors: {neighbors.shape}")
    print(f"  distances: {distances.shape}")
    print(f"\nUsage:")
    print(f"  python generate_filtered_ground_truth.py \\")
    print(f"    --input {args.input} \\")
    print(f"    --output {args.output} \\")
    print(f"    --space-type {args.space_type} \\")
    print(f"    --filter-spec '{json.dumps(filter_spec)}'")


if __name__ == "__main__":
    main()
