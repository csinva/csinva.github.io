[toc]

# graphs
- 3 basic ways to represent a graph in memory (objects and pointers, matrix, and adjacency list)
	1. objections & pointers
	2. matrix
	3. adjacency list
- Djikstra
	1.  initialize all vertex's distance to infinity
	2. start somewhere, set distance to 0, set it visited
	3. pick next unknown vertex v with shortest distance
		- set that visited
		- update its distance / path
		- for each edge from v to adajenct unknown
			- update v
	4. repeat 3 until nothing not visited
- Prim's
	1. start somewhere
	2. find least outgoing edge of mst so far
	3. add that vertex
		4. repeat 2,3
	O (E log E)
- Kruskal's
	- grow a bunch of trees
	O (E log V) - same asymptotic (log(V^2) = 2log(V))
- detect cycle
	- dfs from every vertex and keep track of visited, if repeat then cycle
- topological sort - list vertices in order (all edges point in one direction)
	- something with no edges can be put anywhere
	- doesn't work iff there are cycles
	- algorithm
		1. calculate all indegrees
		2. find node of indegree 0
			- subtract from indegrees, all outgoing edges of 1st node
		3. repeat 2 until no more nodes
		
# divide 2 nums
- cant just loop
- have to cast to longs
- keep shifting until greater
- repeatedly do this shifting

# recursion
- moving down/right on an NxN grid - each path has length (N-1)+(N-1)
	- we must move right N-1 times
	- ans = (N-1+N-1 choose N-1)
	- for recursion, if a list is declared outside static recursive method, it shouldn't be static
- *generate permutations* - recursive, add char at each spot
- think hard about the base case before starting 
	- look for lengths that you know
	- look for symmetry
- n-queens - one array of length n, go row by row

# searching/sorting
- in exceptional cases insertion-sort or radix-sort are much better than the generic QuickSort / MergeSort / HeapSort answers.
- binary search - use low<= val and high >=val so you get correct bounds
- insertion sort best when almost sorted
- radix sort best when small number of possible values
- quicksort usually fastest, but can be O(n^2)
	- pick pivot, move things less than to left and things greater than to right
	- returns void
	- don't actually have to put pivot anywhere
	- log n average extra space, sometimes n
	- On average, mergesort does fewer comparisons than quicksort, so it may be better when complicated comparison routines are used. Mergesort also takes advantage of pre-existing order, so it would be favored for using sort() to merge several sorted arrays. 
	- quicksort  often faster for small arrays, and on arrays of a few distinct values, repeated many times
- *mergesort* can be parallelized, but usually uses extra space
	- in place goes to n log^2(n)
	- to implement, create a class so each method doesn't have to create its own array
		- only extra memory is when merging (create temp array)
	- returns an int[]
	- have to write merge method
- heapsort
	- n log n
	- put all objects into a heap 
	- keep removing min and adding to array
- merge a and b sorted - start from the back
- binary sort can't do better than linear if there are duplicates
- if data is too large, we need to do external sort (sort parts of it and write them back to file)
- write binary search recursively
	- binary search with empty strings - make sure that there is an element at the end of it
- "a".compareTo("b") is -1 

# dp - knapsack ex.
```java
//returns max value for knapsack of capacity W, weights wt, vals val
int knapSack(int W, int wt[], int val[])
int n = wt.length;
int K[n+1][W+1];
//build table K[][] in bottom up manner
for (int i = 0; i <= n; i++)
   for (int w = 0; w <= W; w++)
	   if (i==0 || w==0) // base case
		   K[i][w] = 0;
	   else if (wt[i-1] <= w) //max of including weight, not including
		   K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]], K[i-1][w]);
	   else //weight too large
		   K[i][w] = K[i-1][w]; 
return K[n][W];
```

# hungarian
- assign N things to N targets, each with an associated cost

# max-flow
- A list of pipes is given, with different flow-capacities. These pipes are connected at their endpoints. What is the maximum amount of water that you can route from a given starting point to a given ending point?