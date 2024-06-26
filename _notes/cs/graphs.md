---
layout: notes
title: Graphs
category: cs
subtitle: Some notes on computer science graphs.
---

{:toc}

- Edges are of the form (v1, v2)
    - Can be ordered pair or unordered pair
- Definitions
    - A *weight* or cost can be associated with each edge - this is determined by the application
    - w is adjacent to v iff (v, w) $\in$ E
    - path: sequence of vertices w1, w2, w3, ..., wn such that (wi, wi+1) ∈ E for 1 ≤ i < n
    - length of a path: number of edges in the path
    - simple path: all vertices are distinct
    - cycle:
        - directed graph: path of length $\geq$ 1 such that w1 = wn
        - undirected graph: same, except all edges are distinct
    - connected: there is a path from every vertex to every other vertex
    - loop: (v, v) $\in$ E
    - complete graph: there is an edge between every pair of vertices
- digraph
    - directed acyclic graph: no cycles; often called a "DAG"
    - strongly connected: there is a path from every vertex to every other vertex
    - weakly connected: the underlying undirected graph is connected
- For Google Maps, an adjacency matrix would be infeasible - almost all zeros (sparse)
    - an adjacency list would work much better
    - an adjacency matrix would work for airline routes
- detect cycle
  - dfs from every vertex and keep track of visited, if repeat then cycle
- Topological Sort
    - Given a directed acyclic graph, construct an ordering of the vertices such that if there is a path from vi to vj, then vj appears after vi in the ordering
    - The result is a linear list of vertices
    - indegree of v: number of edges (u, v) -- meaning the number of incoming edges
    - Algorithm
        - start with something of indegree 0
        - take it out, and take out the edges that start from it
        - keep doing this as we take out more and more edges
    - can have multiple possible topological sorts
- Shortest Path
    - single-source - start somewhere, get shortest path to everywhere
    - unweighted shortest path - breadth first search
    - Weighted Shortest Path
        - We assume no negative weight edges
        - Djikstra's algorithm: uses similar ideas as the unweighted case
        - Greedy algorithms: do what seems to be best at every decision point
        - Djikstra: v^2
            - Initialize each vertex's distance as infinity
            - Start at a given vertex s
                - Update s's distance to be 0
            - Repeat
                - Pick the next unknown vertex with the shortest distance to be the next v
                - If no more vertices are unknown, terminate loop
            - Mark v as known
            - For each edge from v to adjacent unknown vertices w
                - If the total distance to w is less than the current distance to w
                - Update w's distance and the path to w
            - It picks the unvisited vertex with the lowest-distance, calculates the distance through it to each unvisited neighbor, and updates the neighbor's distance if smaller. Mark visited (set to red) when done with neighbors.
    - Shortest path from a start node to a finish node
        - 1. We can just run Djikstra until we get to the finish node
        - 2. Have different kinds of nodes
            - Assume you are starting on a "side road"
            - Transition to a "main road"
            - Transition to a "highway"
            - Get as close as you can to your destination via the highway system
            - Transition to a "main road", and get as close as you can to your destination
            - Transition to a "side road", and go to your destination
- Traveling Salesman
    - Given a number of cities and the costs of traveling from any city to any other city, what is the least-cost round-trip route that visits each city exactly once and then returns to the starting city?
    - Hamiltonian path: a path in a connected graph that visits each vertex exactly once
    - Hamiltonian cycle: a Hamiltonian path that ends where it started
    - The traveling salesperson problem is thus to find the least weight Hamiltonian path (cycle) in a connected, weighted graph
- Minimum Spanning Tree
    - Want fully connected
    - Want to minimize number of links used
        - We won't have cycles
    - Any solution is a tree
    - Slow algorithm: Construct a spanning tree:
        - Start with the graph
        - Remove an edge from each cycle
        - What remains has the same set of vertices but is a tree
        - Spanning Trees
- Minimal-weight spanning tree: spanning tree with the minimal total weight
    - Generic Minimum Spanning Tree Algorithm
        - KnownVertices <- {}
        - while KnownVertices does not form a spanning tree, loop:
            - find edge (u,v) that is "safe" for KnownVertices
            - KnownVertices <- KnownVertices U {(u,v)}
        - end loop
    - Prim's algorithm
        - Idea: Grow a tree by adding an edge to the "known" vertices from the "unknown" vertices. Pick the edge with the smallest weight.
        - Pick one node as the root,
        - Incrementally add edges that connect a "new" vertex to the tree.
        - Pick the edge (u,v) where:
        - u is in the tree, v is not, AND
        - where the edge weight is the smallest of all edges (where u is in the tree and v is not)
        - Running time: Same as Dijkstra's: Θ(e log v)
    - Kruskal's algorithm
        - Idea: Grow a forest out of edges that do not create a cycle. Pick an edge with the smallest weight.
        - When optimized, it has the same running time as Prim's and Dijkstra's: Θ(e log v)
        - unoptomized: v^2