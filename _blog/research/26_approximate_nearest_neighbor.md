---
layout: notes
title: Approximate nearest neighbor search
category: blog
---
*July 7, 2026*

Most retrieval systems are built on top of a single operation: given a query vector, find the closest vectors in a large collection. Doing this exactly means comparing the query against every stored vector. That's fine for a few thousand items but hopelessly slow at a hundred million.

Approximate nearest neighbor (ANN) tackles this problem by trading off accuracy for speed. This post covers the key methods for handling this tradeoff, considering quantities like recall (how often the true neighbors are returned), query latency, index build time, and memory footprint:

<img src="../../notes/assets/ann_cheatsheet.svg" style="width: 100%; filter: none; margin: 0;" />

## Locality-sensitive hashing

Locality-sensitive hashing (LSH) is a [fuzzy hashing](https://en.wikipedia.org/wiki/Fuzzy_hashing) technique that maps similar inputs into the same buckets with high probability. The idea is to project each vector onto a set of random directions, use the projections to assign it a bucket, and at query time only search the vectors that land in the same bucket. Note that this reverses the usual goal of a hash function: here collisions are maximized rather than minimized.

The method is largely outdated now. The main problem is that the number of random projections needed for good recall grows quickly, which erodes the speedup it was supposed to provide. It is worth knowing mostly for context, and because finding embeddings with a neural network can be viewed as a learned special case of the same idea, sometimes called semantic hashing.

<div style="display: flex; gap: 2%; justify-content: center; align-items: flex-start; margin: 1.5rem 0;">
  <img src="../../notes/assets/lsh_index_time_dark.svg" style="width: 49%; filter: none; margin: 0;" />
  <img src="../../notes/assets/lsh_query_time_dark.svg" style="width: 49%; filter: none; margin: 0;" />
</div>

## k-d trees

A k-d (k-dimensional) tree is a balanced binary tree built over the data, where each node splits the points along a single dimension. At query time you walk down to a leaf, then back up the tree, checking the neighbors of the leaf as you go. This works well in low dimensions but degrades in the high-dimensional spaces that embeddings live in.

[Annoy](https://github.com/spotify/annoy) (from Spotify) improved on the basic tree in two ways. Instead of a single tree it builds a forest of random trees, and instead of splitting on the median of one dimension it splits in a data-driven way, picking two random points and cutting along the direction between them.

<div style="display: flex; gap: 2%; justify-content: center; align-items: flex-start; margin: 1.5rem 0;">
  <img src="../../notes/assets/kd_tree_index_time_dark.svg" style="width: 49%; filter: none; margin: 0;" />
  <img src="../../notes/assets/kd_tree_query_time_dark.svg" style="width: 49%; filter: none; margin: 0;" />
</div>

## Inverted file index (IVF)

The inverted file index runs k-means over the embeddings to group them into clusters. At query time you first compare the query against the cluster centers, then search only within the vectors of the top few clusters. This cuts the number of comparisons from the full dataset down to a handful of clusters.

Recall suffers near cluster boundaries, since a true neighbor can sit just across the line in a cluster you never open. A common fix is to assign each vector to several nearby clusters instead of just one, so boundary cases are covered from more than one side.

<div style="display: flex; gap: 2%; justify-content: center; align-items: flex-start; margin: 1.5rem 0;">
  <img src="../../notes/assets/ivf_index_time_dark.svg" style="width: 49%; filter: none; margin: 0;" />
  <img src="../../notes/assets/ivf_query_time_dark.svg" style="width: 49%; filter: none; margin: 0;" />
</div>

## Product quantization

Product quantization ([Jégou et al. 2011](https://ieeexplore.ieee.org/document/5432202)) attacks a different part of the problem. Rather than reducing how many vectors you compare, it speeds up each individual comparison and shrinks memory at the same time.

Each vector is split into subvectors, for example its first half and second half. You run k-means separately on each subvector position across the dataset, then replace every subvector with the index of its nearest center. A vector is now stored as a short list of small integer codes instead of full-precision floats.

At query time you split the query the same way and precompute the squared distance from each query subvector to every center for that position. Then, for any stored vector, you skip the dot product entirely and just look up the precomputed distance for each of its codes, summing them to approximate the squared distance to the whole vector.

<div style="display: flex; gap: 2%; justify-content: center; align-items: flex-start; margin: 1.5rem 0;">
  <img src="../../notes/assets/pq_index_time_wide_dark_v2.svg" style="width: 49%; filter: none; margin: 0;" />
  <img src="../../notes/assets/pq_query_time_wide_dark_v2.svg" style="width: 49%; filter: none; margin: 0;" />
</div>

Note: splitting vectors and clustering struggles if there are lots of correlations between different subvector parts, one fix for this is [Optimized PQ](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/pami13opq.pdf) (OPQ), which learns a rotation to apply before splitting so the variance is balanced and decorrelated across subspaces.

## HNSW

Hierarchical Navigable Small World graphs ([Malkov & Yashunin 2016](https://arxiv.org/abs/1603.09320)) are the most popular modern approach for medium-to-large collections, up to roughly ten million vectors. They sit under Lucene, pgvector, Qdrant, Weaviate, Milvus, and Pinecone's early stack.

The idea is to build a proximity graph over the vectors and answer queries with a greedy best-first walk: start somewhere, repeatedly hop to whichever neighbor is closest to the query, and search that neighbor's connections. The hierarchy adds several layers. Search begins on a sparse top layer that covers long distances, then drops into denser lower layers for finer refinement as it homes in.

The tricky part is choosing edges at insert time. A naive graph of nearest neighbors does not work, because if the walk starts in the wrong region it can never reach the right one. The underlying NSW construction avoids this by inserting points in random order and connecting each new point to its M nearest neighbors at the moment of insertion. Points added early end up with useful long-range edges that keep the graph navigable.

HNSW supports incremental inserts, though both building and inserting are expensive. Deletes are awkward, and the graph generally needs to live in RAM rather than on disk, which becomes hard at large scale.

<div style="display: flex; gap: 2%; justify-content: center; align-items: flex-start; margin: 1.5rem 0;">
  <img src="../../notes/assets/hnsw_index_time_insert_dark.svg" style="width: 49%; filter: none; margin: 0;" />
  <img src="../../notes/assets/hnsw_layered_greedy_walk_dark.svg" style="width: 49%; filter: none; margin: 0;" />
</div>

## Pushing the index to disk

Once a collection reaches a hundred million vectors or more, keeping everything in RAM stops being affordable, and the newer systems move most of the data to SSD.

[DiskANN](https://suhasjs.github.io/files/diskann_neurips19.pdf) (Subramanya et al. 2019, from CMU, UT Austin, and Microsoft Research) keeps the full-precision vectors and the graph on SSD and holds only the PQ-compressed vectors in RAM. Its Vamana graph construction is designed to produce fewer, longer-range hops, which matters because every hop that misses the RAM cache costs an SSD round-trip.

[SPANN](https://arxiv.org/abs/2111.08566) (Chen et al. 2021) applies the same disk-first strategy to the IVF cluster approach instead of the graph approach, keeping cluster routing information in memory and the cluster contents on disk.

<div style="display: flex; gap: 2%; justify-content: center; align-items: flex-start; margin: 1.5rem 0;">
  <span style="display: inline-block; width: 49%; text-align: center;">
    <b>DiskANN</b><br>
    <img src="../../notes/assets/diskann_index_time_dark.svg" style="width: 100%; filter: none; margin: 0;" />
  </span>
  <span style="display: inline-block; width: 49%; text-align: center;">
    <b>SPANN</b><br>
    <img src="../../notes/assets/spann_index_time_dark.svg" style="width: 100%; filter: none; margin: 0;" />
  </span>
</div>

## Other considerations

Two concerns cut across all of these methods in practice. Modern libraries put a lot of care into quantization and into running on GPUs, since both change the memory and throughput math. And real queries are rarely pure vector search: users want filtered lookups like "nearest neighbors where tenant = X and date > Y." Building indexes and graphs that stay correct and fast under those filters, rather than degrading to a full scan, is an active area of research.

## Libraries

*Notes: popularity and internals shift quickly, so verify against [ann-benchmarks.com](https://ann-benchmarks.com/) before depending on any of these. The engines section contains no new algorithms: each is HNSW-or-IVF plus quantization plus the operational layer (filtering, replication, segments, hybrid search).*


| Library                                                     | What it is                                                   | Methods it connects to                                       |
| :----------------------------------------------------------- | :------------------------------------------------------------ | :------------------------------------------------------------ |
| [Faiss](https://github.com/facebookresearch/faiss) (Meta)   | The field's reference library, CPU and GPU. Nearly every classical index type, composable (e.g. IVF coarse layer + PQ codes + rescoring). | Brute force, IVF, PQ/OPQ, IVF-PQ (asymmetric distance computation), HNSW, scalar/binary quantization |
| [hnswlib](https://github.com/nmslib/hnswlib) (Malkov)       | The canonical lightweight HNSW implementation by the paper's author; embedded inside many engines. Sibling research library: [nmslib](https://github.com/nmslib/nmslib) (original NSW). | HNSW exactly, with M, efConstruction, efSearch, diversity pruning |
| [ScaNN](https://github.com/google-research/scann) (Google)  | Partitioning plus anisotropic quantization, the PQ loss reshaped to penalize ranking-relevant error. | IVF + PQ with a smarter estimator ("train the index for the metric") |
| [DiskANN](https://github.com/microsoft/DiskANN) (Microsoft) | Vamana graphs served from SSD; Fresh (streaming) and Filtered variants. | DiskANN, with α-prune build, PQ-steered beam search, packed SSD sector reads |
| [SPTAG](https://github.com/microsoft/SPTAG) (Microsoft)     | Tree-plus-graph library; home of the SPANN design used in Bing-scale search. | SPANN, with in-RAM centroid routing, closure assignment for boundary vectors |
| [Annoy](https://github.com/spotify/annoy) (Spotify)         | Random-projection tree forest; mmap-friendly, immutable, simple. Largely legacy now but still deployed. | The tree lineage, the k-d tree's randomized, ensembled descendant |
| [FALCONN](https://github.com/FALCONN-LIB/FALCONN)           | Research-grade LSH (cross-polytope and hyperplane families) with multi-probe. | LSH, with the modern hash families and fewer tables          |
| [cuVS](https://github.com/rapidsai/cuvs) (NVIDIA)           | GPU-native ANN: CAGRA graph build and search, GPU IVF-PQ; build speedups that change reindexing economics. | Graph traversal + IVF-PQ, rebuilt for massive parallelism    |

| Engine                                                       | What it is                                                   | Methods it connects to                                       |
| :------------------------------------------------------------ | :------------------------------------------------------------ | :------------------------------------------------------------ |
| [Lucene](https://lucene.apache.org/) (→ Elasticsearch, OpenSearch, MongoDB Atlas) | Segment-based search engine whose vector support underlies Elasticsearch, OpenSearch, and Atlas Vector Search; immutable segments sidestep graph deletes. | HNSW per segment + int8/binary quantization with exact rescoring |
| [pgvector](https://github.com/pgvector/pgvector)             | Postgres extension; vectors as a column type with SQL filtering and joins. | HNSW, IVF (IVFFlat), and exact brute-force scans             |
| [Milvus](https://github.com/milvus-io/milvus)                | Distributed vector database; its Knowhere engine wraps multiple index families behind one API. | Nearly everything: Faiss indexes, HNSW, DiskANN, GPU variants |
| [Qdrant](https://github.com/qdrant/qdrant)                   | Rust vector database emphasizing payload filtering and compression. | HNSW + scalar/binary quantization; filter-aware traversal    |
| [Weaviate](https://github.com/weaviate/weaviate)             | Go vector database with hybrid (BM25 + vector) search built in. | Custom HNSW with PQ/binary compression and rescoring         |

If you want to know (a lot) more about this topic, check out this book: Foundations of vector retrieval book ([bruch, 2024](https://arxiv.org/abs/2401.09350.pdf)).