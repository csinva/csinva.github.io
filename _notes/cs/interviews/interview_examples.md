# find common ancestor
- DFS to find first node
	- loop upwards and DFS to find common ancestor

# find next node in bst
```
public Node next_val(Node n){
	if(n==null)
		return null;
	// need to look at right
		// return leftmost node in right subtree
	// otherwise look at parent
	while(parent!=null)
		// if was a left child, parent.val
		// otherwise look at parent again
	}		
	// otherwise return null
}
public class Node{
	int val;
	Node parent;
	Node left;
	Node right;
}
```

# check if binary tree is binary search tree
- binary search tree - all the nodes in left subtree are less than all the nodes in right subtree

# invert tree
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Solution {
    public TreeNode invertTree(TreeNode root) {
        if(root==null)
            return null;
        TreeNode temp = root.left;
        root.left = invertTree(root.right);
        root.right = invertTree(temp);
        
        return root;
    }
}
```

# gray code

# top k-frequent
	- Integer can be null, int can't
	- use bucket-sort

# permutations
- generate permutations - keep adding new numbers at all possible locations

```java
public class Permutation {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> l = new LinkedList<List<Integer>>();
        
        if(nums.length==0)
            return l;
            
        LinkedList<Integer> init = new LinkedList<Integer>();
        init.add(nums[0]);
        l.add(init);
        for(int i=1;i<nums.length;i++){
            l = permute(l,nums[i]);
            System.out.println(i+" "+l);
        }
        return l;
    }
    
    public List<List<Integer>> permute(List<List<Integer>> l, int target){
        List<List<Integer>> l2 = new LinkedList<List<Integer>>();
        for(List<Integer> el:l){
            // l2.add(el);
            for(int i=0;i<=el.size();i++){
                List<Integer> el_new = new LinkedList<Integer>(el);
                el_new.add(i,target);
                l2.add(el_new);
            }
        }
        return l2;
    }
}
```

# dp no adjacents
it will automatically contact the police if two adjacent houses were broken into on the same night.
Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.

```java
public int rob(int[] nums) {
        int N = nums.length;
        if(N<=0)
            return 0;
        if(N==1)
            return nums[0];
        int[][] helper = new int[3][N];
        helper[0][0] = nums[0];
        helper[1][0] = 0;
        for(int i=1;i<N;i++){
            helper[0][i] = helper[1][i-1] + nums[i];
            helper[1][i] = Math.max(helper[0][i-1],helper[1][i-1]);
            System.out.println(helper[0][i]+" "+helper[1][i]);
        }
        return Math.max(helper[0][N-1],helper[1][N-1]);
}
```

# shuffle
Fisher–Yates shuffle Algorithm works in O(n) time complexity. The assumption here is, we are given a function rand() that generates random number in O(1) time.
The idea is to start from the last element, swap it with a randomly selected element from the whole array (including last)

# combinations - generate n choose k?
- from icpc binder - think about removing one element from front and moving it around

# unique bsts
*dynamic programming - think about what could be the root*
public int numTrees(int n) {
        if(n==0)
            return 0;
        if(n==1)
            return 1;
        int[] counts = new int[n+1];
        counts[0] = 1; //tree of size 0
        counts[1] = 1;
        
        for(int i=2;i<n+1;i++){
            int distr = i-1;
            for(int j=0;j<=distr;j++){ //inc
                counts[i]+=counts[j]*counts[distr-j];
            }
        }
        
        return counts[n];
    }
    
# longest increasing subsequence - DP with each bucket the longest sequence ending at index i that must include nums[i]
public int lengthOfLIS(int[] nums) {
        if(nums.length==0)
            return 0;
        
        int[] max_ends = new int[nums.length];
        max_ends[0] = 1;
        
        for(int i=1;i<max_ends.length;i++){
            int max_len_before = 0;
            for(int j=0;j<i;j++){
                if(nums[j]<nums[i]){
                    if(max_ends[j]>max_len_before)
                        max_len_before = max_ends[j];
                }
                max_ends[i] = max_len_before + 1;
            }
        }
        
        int max=max_ends[0];
        for(int i:max_ends){
            System.out.print(i+" ");
            if(i>max)
                max = i;
        }
        return max;
    }

# towers of hanoi
	- Base Case - When n = 1
		- Move the disc from start pole to end pole 
	- Recursive Case - When n > 1
		Step 1: Move (n-1) discs from start pole to auxiliary pole.
		Step 2: Move the last disc from start pole to end pole.
		Step 3: Move the (n-1) discs from auxiliary pole to end pole.
		Steps 1 and 3 are recursive invocations of the same procedure. 
	- public void solve(int n, String start, String auxiliary, String end) {
       if (n == 1) {
           System.out.println(start + " -> " + end);
       } else {
           solve(n - 1, start, end, auxiliary);
           System.out.println(start + " -> " + end);
           solve(n - 1, auxiliary, start, end);
       }
      }