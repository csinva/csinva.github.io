---
layout: notes
title: Data Structures
category: cs
subtitle: Some notes on advanced data structures, based on UVA's "Program and Data Representation" course.
---

{:toc}

# lists
## arrays and strings
- start by checking for null, length 0
- *ascii* is 128, extended is 256

## queue - linkedlist
- has insert at back (enqueue) and remove from front (dequeue)
```java
class Node {
	Node next;
	int val;
	public Node(int d) { val = d; } 
}
```
- finding a loop is tricky, use visited
- reverse a linked list
  - requires 3 ptrs (one temporary to store next)
  - return pointer to new end

## stack
```java
class Stack { 
	Node top;
	Node pop() {
		if (top != null) {
			Object item = top.data; 
			top = top.next;
			return item;
		}
		return null;
	}
	void push(Object item) { 
		Node t = new Node(item);
		t.next = top;
		top = t;
	} 
}
```
- sort a stack with 2 stacks
  - make a new stack called ans
  - pop from old
  - while old element is > ans.peek(), old.push(ans.pop())
  - then new.push(old element)
- stack with min - each el stores min of things below it
- queue with 2 stacks - keep popping everything off of one and putting them on the other
- sort with 2 stacks

# trees
- Balanced binary trees are generally logarithmic
    - Root: a node with no parent; there can only be one root
    - Leaf: a node with no children
    - Siblings: two nodes with the same parent
    - Height of a node: length of the longest path from that node to a leaf
       - Thus, all leaves have height of zero
        - Height of a tree: maximum depth of a node in that tree = height of the root
    - Depth of a node: length of the path from the root to that node
    - Path: sequence of nodes n1, n2, ..., nk such that ni is parent of ni+1 for 1 ≤ i ≤ k
    - Length: number of edges in the path
    - Internal path length: sum of the depths of all the nodes
- Binary Tree - every node has at most 2 children
- Binary Search Tree - Each node has a key value that can be compared
    - Every node in left subtree has a key whose value is less than the root's key value
    - Every node in right subtree has a key whose value is greater than the root's key value

```java
void BST::insert(int x, BinaryNode * & curNode){    //we pass in by reference because we want a change in the method to actually modify the parameter (the parameter is the curNode *)
    //left associative so this is a reference to a pointer
    if (curNode==NULL)
        curNode = new BinaryNode(x,NULL,NULL);
    else if(x<curNode->element)
        insert(x,curNode->left);
    else if(x>curNode->element)
        insert(x,curNode->right);
}
```
- BST Remove
  - if no children: remove node (reclaiming memory), set parent pointer to null
        - one child: Adjust pointer of parent to point at child, and reclaim memory
        - two children: successor is min of right subtree
            - replace node with successor, then remove successor from tree
    - worst-case depth = n-1 (this happens when the data is already sorted)
    - maximum number of nodes in tree of height h is 2^(h+1) - 1
    - minimum height h ≥ log(n+1)-1
    - Perfect Binary tree - impractical because you need the perfect amount of nodes
        - all leaves have the same depth
        - number of leaves 2^h
## AVL Tree
- For every node in the tree, the height of the left and right sub-trees differs at most by 1
    - guarantees log(n)
    - balance factor := The height of the right subtree minus the height of the left subtree
    - "Unbalanced" trees: A balance factor of -2 or 2
    - AVL Insert - needs to update balance factors
    - same sign -> single rotation
    - -2, -1 -> needs right rotation
    - -2, +1 -> needs left then right
    - Find: Θ(log n) time: height of tree is always Θ(log n)
    - Insert: Θ(log n) time: find() takes Θ(log n), then may have to visit every node on the path back up to root to perform up to 2 single rotations
    - Remove: Θ(log n): left as an exercise
    - Print: Θ(n): no matter the data structure, it will still take n steps to print n elements

## Red-Black Trees
- definition
    1. A node is either red or black
    2. The root is black
    3. All leaves are black
        The leaves may be the NULL children
    4. Both children of every red node are black
        Therefore, a black node is the only possible parent for a red node
    5. Every simple path from a node to any descendant leaf contains the same number of black nodes
- properties
    - The height of the right and left subtree can differ by a factor of n
    - insert (Assume node is red and try to insert)
        1. The new node is the root node
        2. The new node's parent is black
        3. Both the parent and uncle (aunt?) are red
        4. Parent is red, uncle is black, new node is the right child of parent
        5. Parent is red, uncle is black, new node is the left child of parent
    - Removal
        - Do a normal BST remove
            - Find next highest/lowest value, put it's value in the node to be deleted, remove that highest/lowest node
                - Note that that node won't have 2 children!
            - We replace the node to be deleted with it's left child
                - This child is N, it's sibling is S, it's parent is P
## Splay Trees
- A self-balancing tree that keeps "recently" used nodes close to the top
  - This improves performance in some cases
  - Great for caches
  - Not good for uniform access
- Anytime you find / insert / delete a node, you splay the tree around that node
- Perform tree rotations to make that node the new root node
- Splaying is Θ(h) where h is the height of the tree
    - At worst this is linear time - Θ(n)
    - We say it runs in Θ(log n) amortized time - individual operations might take linear time, but other operations take almost constant time - averages out to logarithmic time
        - m operations will take m*log(n) time

## other trees
- to go through *bst (without recursion) in order*, use stacks
  - push and go left
  - if can't go left, pop
    - add new left nodes
      - go right
- *breadth-first tree*
  - recursively print only at a particular level each time
  - create pointers to nodes on the right
- *balanced tree*  = any 2 nodes differ in height by more than 1
  - (maxDepth - minDepth) <=1
- *trie* is an infix of the word “retrieval” because the trie can find a single word in a dictionary with only a prefix of the word
  - root is empty string
  - each node stores a character in the word
  - if ends, full word
    - need a way to tell if prefix is a word -> each node stores a boolean isWord

# heaps
- used for *priority queue*
- peek(): just look at the root node
- add(val): put it at correct spot, percolate up
  - percolate - Repeatedly exchange node with its parent if needed
  - expected run time: ∑i=1..n 1/2^n∗n=2
- pop(): put last leaf at root, percolate down
  - Remove root (that is always the min!)
  - Put "last" leaf node at root
  - Repeatedly find smallest child and swap node with smallest child if needed.
- Priority Queue - Binary Heap is always used for Priority Queue
    1. insert
        - inserts with a priority
    2. findMin
        - finds the minimum element
    3. deleteMin
        - finds, returns, and removes minimum element
- perfect (or complete) binary tree - binary tree with all leaf nodes at the same depth; all internal nodes have 2 children.
    - height h, 2h+1-1 nodes, 2h-1 non-leaves, and 2h leaves
- Full Binary Tree
    - A binary tree in which each node has exactly zero or two children.
- Min-heap - parent is min
    1. Heap Structure Property: A binary heap is an almost complete binary tree, which is a binary tree that is completely filled, with the possible exception of the bottom level, which is filled left to right.
        - in an array - this is faster than pointers 
            - left child: 2*i
            - right child: (2*i)+1
            - parent: floor(i/2)
            - pointers need more space, are slower
            - multiplying, dividing by 2 are very fast
    2. Heap ordering property: For every non-root node X, the key in the parent of X is less than (or equal to) the key in X. Thus, the tree is partially ordered.
- Heap operations
    - findMin: just look at the root node
    - insert(val): put it at correct spot, percolate up
        - percolate - Repeatedly exchange node with its parent if needed
        - expecteed run time: ∑i=1..n 1/2^n∗n=2
    - deleteMin: put last leaf at root, percolate down
        - Remove root (that is always the min!)
        - Put "last" leaf node at root
        - Repeatedly find smallest child and swap node with smallest child if needed.
- Compression
    - Lossless compression: X = X'
    - Lossy compression: X != X'
        - Information is lost (irreversible)
    - Compression ratio: $\vert X\vert /\vert Y\vert $
        - Where $\vert X\vert $ is the number of bits (i.e., file size) of X
- Huffman coding
    - Compression
        1. Determine frequencies
        2. Build a tree of prefix codes
            - no code is a prefix of another code
            - start with minheap, then keep putting trees together
        3. Write the prefix codes to the output
        4. reread source file and write prefix code to the output file
    - Decompression
        1. read in prefix code - build tree
        2. read in one bit at a time and follow tree
- ASCII characters - 8 bits, 2^7 = 128 characters
    - cost - total number of bits
    - "straight cost" - bits / character = log2(numDistinctChars)
- Priority Queue Example
    - insert (x)
    - deleteMin()
    - findMin()
    - isEmpty()
    - makeEmpty()
    - size()

# Hash tables

- java: load factor = .75, default init capacity: 16, uses buckets
- string hash function: s[0]*31^(n-1) + s[1]*31^(n-2) + ... + s[n-1] where n is length mod (table_size)
    - Standard set of operations: find, insert, delete
    - No ordering property!
    - Thus, no findMin or findMax
    - Hash tables store key-value pairs
    - Each value has a specific key associated with it
- fixed size array of some size, usually a prime number
- A hash function takes in a "thing" )string, int, object, etc._
  
    - returns hash value - an unsigned integer value which is then mod'ed by the size of the hash table to yield a spot within the bounds of the hash table array
- Three required properties
    1. Must be deterministic
        - Meaning it must return the same value each time for the same "thing"
    2. Must be fast
    3. Must be evenly distributed
        - implies avoiding collisions
    - Technically, only the first is required for correctness, but the other two are required for fast running times
- A perfect hash function has:
    - No blanks (i.e., no empty cells)
    - No collisions
- Lookup table is at best logarithmic
- We can't just make a very large array - we assume the key space is too large
  
    - you can't just hash by social security number
- hash(s)=(∑k−1i=0si∗37^i) mod table_size
  
    - you would precompute the powers of 37
- collision - putting two things into same spot in hash table
    - Two primary ways to resolve collisions:
        1. Separate Chaining (make each spot in the table a 'bucket' or a collection)
        2. Open Addressing, of which there are 3 types:
            1. Linear probing
            2. Quadratic probing
            3. Double hashing
- Separate Chaining
    - each bucket contains a data structure (like a linked list)
    - analysis of find
        - The load factor, λ, of a hash table is the ratio of the number of elements divided by the table size
            - For separate chaining, λ is the average number of elements in a bucket
                - Average time on unsuccessful find: λ
                    - Average length of a list at hash(k)
                - Average time on successful find: 1 + (λ/2)
                    - One node, plus half the average length of a list (not including the item)
            - typical case will be constant time, but worst case is linear because everything hashes to same spot
            - λ = 1
                - Make hash table be the number of elements expected
                - So average bucket size is 1
                - Also make it a prime number
            - λ = 0.75
                - Java's Hashtable but can be set to another value
                - Table will always be bigger than the number of elements
                - This reduces the chance of a collision!
                - Good trade-off between memory use and running time
            - λ = 0.5
                - Uses more memory, but fewer collisions
- Open Addressing: The general idea with all of them is that, if a spot is occupied, to 'probe', or try, other spots in the table to use
    - 3 Types:
        - General: pi(k) = (hash(k) + f(i)) mod table_size
          1.Linear Probing: f(i) = i
            - Check spots in this order :
                - hash(k)
                - hash(k)+1
                - hash(k)+2
                - hash(k)+3
                - These are all mod table_size
            - find - keep going until you find an empty cell (or get back)
            - problems
                - cannot have a load factor > 1, as you get close to 1, you get a lot of collisons
                - clustering - large blocks of occupied cells
                - "holes" when an element is removed
                  2.Quadratic:  f(i) = i^2
            - hash(k)
            - hash(k)+1
            - hash(k)+4
            - hash(k)+9
            - you move out of clusters much quicker
              3.Double hashing: i * hash2(k)
            - hash2 is another hash function - typically the fastest
            - problem where you loop over spots that are filled - hash2 yields a factor of the table size
                - solve by making table size prime
            - hash(k) + 1 * hash2(k)
            - hash(k) + 2 * hash2(k)
            - hash(k) + 3 * hash2(k)
    - a prime table size helps hash function be more evenly distributed
    - problem: when the table gets too full, running time for operations increases
    - solution: create a bigger table and hash all the items from the original table into the new table
        - position is dependent on table size, which means we have to rehash each value
        - this means we have to re-compute the hash value for each element, and insert it into the new table!
        - When to rehash?
            - When half full (λ = 0.5)
            - When mostly full (λ = 0.75)
                - Java's hashtable does this by default
            - When an insertion fails
            - Some other threshold
        - Cost of rehashing
            - Let's assume that the hash function computation is constant
            - We have to do n inserts, and if each key hashes to the same spot, then it will be a Θ(n2) operation!
            - Although it is not likely to ever run that slow
    - Removing
        - You could rehash on delete
        - You could put in a 'placeholder' or 'sentinel' value
            - gets filled with these quickly
            - perhaps rehash after a certain number of deletes
- has functions    
    - MD5 is a good hash function (given a string or file contents)
        - generates 128 bit hash
        - when you download something, you download the MD5, your computer computes the MD5 and they are compared to make sure it downloaded correctly
        - not reversible because when a file has more than 128 bits, won't be 1-1 mapping
        - you can lookup a MD5 hash in a rainbow table - gives you what the password probably is based on the MD5 hash
    - SHA (secure Hash algorithm) is much more secure
        - generates hash up to 512 bits