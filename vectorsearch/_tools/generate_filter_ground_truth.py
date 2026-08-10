"""
Generate a filtered vector search dataset with universal ground truth.

Assigns each document a shuffled ID (0 to N-1). The universal neighbor list
per query contains all docs that ever appear in the top-k window as the filter
threshold decreases from 100% down to min_filter_ratio. This allows benchmarking
at any filter ratio >= min_filter_ratio without regenerating the dataset.

At benchmark time, filter the stored neighbors by id <= filter_ratio * N,
then take top-k by distance as ground truth for that ratio.

Usage:
    python generate_filter_ground_truth.py \
        --input cohere-1m.hdf5 \
        --output cohere-1m-filter.hdf5 \
        --space-type innerproduct \
        --k 100 \
        --min-filter-ratio 0.001

Output HDF5 contains:
    train:               (N, dim)   float32  — all vectors (copied from input)
    test:                (Q, dim)   float32  — query vectors (copied from input)
    id:                  (N,)       int64    — shuffled doc IDs (0 to N-1)
    neighbors:           (Q, M)     int64    — universal neighbor lists (-1 padded)
    distances:           (Q, M)     float32  — distances for each neighbor (-1 padded)
    faiss_max_distance:  (Q, k)     float32  — per-query Faiss radial thresholds
    faiss_min_score:     (Q, k)     float32  — per-query Faiss min_score thresholds
    lucene_max_distance: (Q, k)     float32  — per-query Lucene radial thresholds
    lucene_min_score:    (Q, k)     float32  — per-query Lucene min_score thresholds

Where M = max universal neighbor list size across all queries (padded with -1).

The threshold datasets (faiss_max_distance etc.) are computed at min_filter_ratio,
i.e. distance to the k-th neighbor when only min_filter_ratio fraction of docs pass.
"""

import argparse
import h5py
import heapq
import numpy as np
import shutil
import sys
import time


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


def compute_universal_neighbors(sorted_row, sorted_dists, shuffled_ids, k, min_filter_max, num_docs):
    """Compute universal neighbor list using sliding window approach.

    Returns:
        neighbors: list of doc indices in distance order (universal set)
        distances: corresponding distances
    """
    # Start with top-k (all pass at full filter)
    window_set = set(int(sorted_row[i]) for i in range(k))
    # Store (distance, doc) pairs for universal set, in order
    universal = [(float(sorted_dists[i]), int(sorted_row[i])) for i in range(k)]
    universal_set = set(int(sorted_row[i]) for i in range(k))

    # Max-heap of (-shuffled_id, doc) for eviction order
    eviction_heap = [(-int(shuffled_ids[sorted_row[i]]), int(sorted_row[i])) for i in range(k)]
    heapq.heapify(eviction_heap)

    pointer = k
    current_filter_max = num_docs - 1

    while True:
        # Clean stale heap entries
        while eviction_heap:
            neg_id, doc = eviction_heap[0]
            if doc in window_set:
                break
            heapq.heappop(eviction_heap)

        if not eviction_heap:
            break

        neg_id, doc = eviction_heap[0]
        next_evict_id = -neg_id

        if next_evict_id <= min_filter_max:
            break

        current_filter_max = next_evict_id - 1

        # Evict all docs with shuffled_id > current_filter_max
        evicted_count = 0
        while eviction_heap and -eviction_heap[0][0] > current_filter_max:
            neg_id, doc = heapq.heappop(eviction_heap)
            if doc in window_set:
                window_set.discard(doc)
                evicted_count += 1

        # Find replacements
        added = 0
        while added < evicted_count and pointer < num_docs:
            candidate = int(sorted_row[pointer])
            dist = float(sorted_dists[pointer])
            pointer += 1
            if shuffled_ids[candidate] <= current_filter_max and candidate not in window_set:
                window_set.add(candidate)
                if candidate not in universal_set:
                    universal_set.add(candidate)
                    universal.append((dist, candidate))
                heapq.heappush(eviction_heap, (-int(shuffled_ids[candidate]), candidate))
                added += 1

    # Sort by distance (nearest first)
    universal.sort(key=lambda x: x[0])
    neighbors = [doc for _, doc in universal]
    distances = [dist for dist, _ in universal]
    return neighbors, distances


def main():
    parser = argparse.ArgumentParser(
        description="Generate filtered vector search dataset with universal ground truth"
    )
    parser.add_argument("--input", required=True, help="Input HDF5 dataset path")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument("--space-type", required=True, choices=["l2", "innerproduct", "cosine"])
    parser.add_argument("--k", type=int, default=100,
                        help="Number of neighbors for ground truth (default: 100)")
    parser.add_argument("--min-filter-ratio", type=float, default=0.001,
                        help="Minimum filter ratio to support (default: 0.001 = 0.1%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    parser.add_argument("--query-batch-size", type=int, default=100,
                        help="Queries per batch (default: 100)")

    args = parser.parse_args()

    if not 0 < args.min_filter_ratio <= 1.0:
        print("ERROR: --min-filter-ratio must be between 0 and 1")
        sys.exit(1)

    print(f"Input:             {args.input}")
    print(f"Output:            {args.output}")
    print(f"Space type:        {args.space_type}")
    print(f"k:                 {args.k}")
    print(f"Min filter ratio:  {args.min_filter_ratio} ({args.min_filter_ratio * 100:.2f}%)")
    print(f"Seed:              {args.seed}")
    print()

    with h5py.File(args.input, "r") as f_in:
        print("Loading dataset...")
        train = f_in["train"][:]
        test = f_in["test"][:]
        print(f"  train: {train.shape}")
        print(f"  test:  {test.shape}")

    num_docs = train.shape[0]
    num_queries = test.shape[0]
    min_filter_max = int(args.min_filter_ratio * num_docs)

    print(f"\nMin filter max: {min_filter_max} / {num_docs} "
          f"({100 * min_filter_max / num_docs:.2f}%)")

    # Assign shuffled IDs
    rng = np.random.default_rng(args.seed)
    shuffled_ids = np.arange(num_docs, dtype=np.int64)
    rng.shuffle(shuffled_ids)

    if min_filter_max < args.k:
        print(f"ERROR: min_filter_max={min_filter_max} < k={args.k}. "
              f"Increase --min-filter-ratio or decrease --k")
        sys.exit(1)

    # Compute universal neighbor lists
    all_neighbors = []
    all_distances = []

    print(f"\nComputing universal neighbor lists for {num_queries} queries...")
    t0 = time.time()

    for batch_start in range(0, num_queries, args.query_batch_size):
        batch_end = min(batch_start + args.query_batch_size, num_queries)
        if batch_start % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  {batch_start}/{num_queries} ({elapsed:.0f}s)")

        batch_queries = test[batch_start:batch_end]
        batch_dists = calculate_distances_batch(batch_queries, train, args.space_type)
        sorted_indices = np.argsort(batch_dists, axis=1)

        for i in range(batch_end - batch_start):
            sorted_row = sorted_indices[i]
            sorted_dists_row = batch_dists[i][sorted_row]
            neighbors, distances = compute_universal_neighbors(
                sorted_row, sorted_dists_row, shuffled_ids,
                args.k, min_filter_max, num_docs
            )
            all_neighbors.append(neighbors)
            all_distances.append(distances)

    print(f"Done in {time.time()-t0:.1f}s")

    # Compute stats on universal list sizes
    sizes = np.array([len(n) for n in all_neighbors])
    max_size = sizes.max()
    print(f"\nUniversal list sizes:")
    print(f"  min={sizes.min()}, median={int(np.median(sizes))}, "
          f"p95={int(np.percentile(sizes, 95))}, p99={int(np.percentile(sizes, 99))}, max={max_size}")

    # Pad to max_size with -1
    neighbors_arr = np.full((num_queries, max_size), -1, dtype=np.int64)
    distances_arr = np.full((num_queries, max_size), -1.0, dtype=np.float32)
    for i, (n, d) in enumerate(zip(all_neighbors, all_distances)):
        neighbors_arr[i, :len(n)] = n
        distances_arr[i, :len(d)] = d

    # Store engine-specific threshold arrays at FULL width (same shape as
    # neighbors/distances). At query time OSB applies the same filter mask it
    # uses for the neighbor list, then reads index k-1 — giving the correct
    # per-ratio radial threshold for any filter_id_max. Without a filter,
    # index k-1 directly is the unfiltered top-k threshold (the first k
    # entries of the universal list are the unfiltered top-k).
    faiss_max_distance = distances_arr.copy()
    lucene_max_distance = -distances_arr if args.space_type == "innerproduct" else distances_arr.copy()
    min_score = raw_distance_to_opensearch_score(distances_arr, args.space_type)

    # Write output
    shutil.copy2(args.input, args.output)

    with h5py.File(args.output, "a") as f_out:
        def write_dataset(name, data):
            if name in f_out:
                del f_out[name]
            f_out.create_dataset(name, data=data)

        write_dataset("attributes", shuffled_ids.reshape(-1, 1).astype(np.int64))
        write_dataset("neighbors", neighbors_arr)
        write_dataset("distances", distances_arr)
        write_dataset("faiss_max_distance", faiss_max_distance)
        write_dataset("faiss_min_score", min_score.astype(np.float32))
        write_dataset("lucene_max_distance", lucene_max_distance)
        write_dataset("lucene_min_score", min_score.astype(np.float32))

        f_out.attrs["space_type"] = args.space_type
        f_out.attrs["k"] = args.k
        f_out.attrs["min_filter_ratio"] = args.min_filter_ratio
        f_out.attrs["min_filter_max"] = min_filter_max
        f_out.attrs["seed"] = args.seed

    print(f"\nWritten to {args.output}")
    print(f"  id:        {shuffled_ids.shape}")
    print(f"  neighbors: {neighbors_arr.shape} (padded with -1)")
    print(f"  distances: {distances_arr.shape}")
    print(f"\nSupports any filter ratio >= {args.min_filter_ratio * 100:.2f}%.")
    print(f"Set filter_id_max = filter_ratio * {num_docs} in the param file.")
    print(f"Example for 10%: filter_id_max={int(0.1 * num_docs)}")


if __name__ == "__main__":
    main()
