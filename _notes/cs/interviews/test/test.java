import java.util.*;
import java.math.BigInteger;
public class test{
	public static void main(String[] args){
		BigInteger b = new BigInteger(new byte[100]);
		System.out.println("a".compareTo("A"));
		ArrayList<Integer> a = new ArrayList<Integer>();
		a.add(1);a.add(2);a.add(3);
		a.add(1);a.add(2);a.add(3);
		a.remove(3);
		System.out.println(a);
		StringBuilder sb = new StringBuilder("abc");
		sb.append('a');
		System.out.println(sb);
		
		// String stuff
		String s = "chandan";
		String s2= "canhdna";
		System.out.println("1.1: "+allUnique(s));
		System.out.println("1.2: "+reverseString(s));
		System.out.println("1.3: "+removeDuplicates(s));
		System.out.println("1.4: "+anagrams(s,s2));
		
		// LinkedList stuff
		Node z = new Node(4,null);
		Node y = new Node(3,z);
		Node x = new Node(3,y);
		Node head = new Node(1,x);
		System.out.print("2.1: "); printList(removeDuplicatesL(head));
		System.out.print("2.1: "); printList(removeDuplicatesL2(head));
		
		// Recursion
		System.out.println("8.3");
		ArrayList<Integer> set = new ArrayList<Integer>();
		set.add(1);
		set.add(2);
		set.add(3);
		ArrayList<ArrayList<Integer>> sets = new ArrayList<ArrayList<Integer>>();
		sets.add(new ArrayList<Integer>(1));
		for(int i=0;i<set.size();i++){
			int len=sets.size();
			int val = set.get(i);
			for(int j=0;j<len;j++){
				ArrayList<Integer> newSet = (ArrayList<Integer>) (sets.get(j).clone());
				newSet.add(val);
				sets.add(newSet);
			}
		}
		for(ArrayList<Integer> sett:sets){
			System.out.println(sett);
		}
		
		System.out.println("8.5");
		char[] str = new char[8];
		printPars(4,4,str,0);
		System.out.println();
		
		System.out.print("8.7: ");
		System.out.println(numWays(25));
				
		String[] anagrams = {"abc","def","cba","fed","xxx","efd","axy","xya"};
		System.out.print("9.2: ");
		Arrays.sort(anagrams,new AnaComparator());
		for(String ana:anagrams)
			System.out.print(ana+", ");
		System.out.println();
		
		System.out.println("9.6");
		int[][] m = {{1,2,3,4},{5,6,7,9},{11,14,25,36},{181,190,201,202}};
		System.out.println(mSearch(m,190));
		
	}
	
	public static boolean allUnique(String s){
		boolean[] seen = new boolean[256];
		for(int i=0;i<256;i++){
			seen[i]=false;
		}
		for(char c:s.toCharArray()){
			if(seen[c])
				return false;
			seen[c]=true;
		}
		return true;
	}
	
	public static String reverseString(String s){
		if(s==null||s.length()==0)
			return "";
		char[] cs = s.toCharArray();
		int end = cs.length-1;
		int start=0;
		while(start<end){
			char temp = cs[start];
			cs[start] = cs[end];
			cs[end] = temp;
			end--;
			start++;
		}
		return new String(cs);
	}
	
	public static String removeDuplicates(String s){
		StringBuffer ans = new StringBuffer();
		char[] cs = s.toCharArray();
		for(int i=0;i<cs.length;i++){
			char c = cs[i];
			boolean dup = false;
			for(int j=0;j<i;j++){
				if(c==cs[j])
					dup=true;
			}
			if(!dup)
				ans.append(c);
		}
		return ans.toString();
	}
	
	public static boolean anagrams(String s1, String s2){
		if(s1==null || s2==null)
			return false;
		if(s1.length()!=s2.length())
			return false;
		char[] cs1 = s1.toCharArray();
		char[] cs2 = s2.toCharArray();
		Arrays.sort(cs1);
		Arrays.sort(cs2);
		for(int i=0;i<cs1.length;i++){
			if(cs1[i]!=cs2[i])
				return false;
		}
		
		return true;
	}
	
	public static Node removeDuplicatesL(Node head){
		if(head==null)
			return null;
		HashSet<Integer> h = new HashSet<Integer>();
		Node n = head;
		h.add(n.data);
		while(n.next!=null){
			if(h.contains(n.next.data)){
				//remove n.next
				n.next=n.next.next;
			}
			else{
				h.add(n.next.data);
				n=n.next;
			}
		}
		return head;
	}
	
	public static Node removeDuplicatesL2(Node head){
		if(head==null)
			return null;
		Node n = head;
		while(n.next!=null){
			// check all elements before n.next to see if they have same val
			int val = n.next.data;
			boolean dup = false;
			for(Node t=head;t.next!=n.next;t=t.next){
				if(t.data==val){
					dup = true;
					break;
				}
			}
			if(dup){
				//remove n.next
				n.next=n.next.next;
			}
			else{
				n=n.next;
			}
		}
		return head;
	}
	
	public static void printList(Node head){
		while(head!=null){
			System.out.print(head.data+" ");
			head=head.next;
		}
		System.out.println();
	}
	
	public static void printPars(int l,int r,char[] str,int count){
		if(l<0 | r<l)
			return;
		if(l==0&&r==0)
			System.out.println(new String(str));
		if(l>0){
			str[count]='(';
			printPars(l-1,r,str,count+1);
		}
		if(r>l){
			str[count]=')';
			printPars(l,r-1,str,count+1);
		}
			
	}
	
	public static int numWays(int cents){
		int count=0;
		for(int q=cents/25;q>=0;q--){
			int remQ = cents-q*25;
			for(int d=remQ/10;d>=0;d--){
				int remD = remQ-d*10;
				if(remD%5==0)
					count++;
			}
		}
		return count;		
	}
	
	public static Point mSearch(int[][] m,int val){
		int R=m.length;
		int C=m[0].length;
		int lr=0;int ur=R-1;
		int lc=0;int uc=C-1;
		boolean foundR=false;
		while(!foundR){
			int mid = (lr+ur)/2;
			if(m[mid][0]==val)
				return new Point(mid,0);
			else if(m[mid][0]<val){
				lr = mid;
			}
			else if(m[mid][0]>val){
				ur = mid;
			}
			if(ur-lr<=1)
				foundR=true;
		}
		
		return null;
	}
	
}

class Node{
	Node next;
	int data;
	public Node(int data, Node next){
		this.data=data;
		this.next=next;
	}
}

class AnaComparator implements Comparator<String>{
	public String charSort(String x){
		char[] cs = x.toCharArray();
		Arrays.sort(cs);
		return new String(cs);
	}
	public int compare(String one, String two){
		return charSort(one).compareTo(charSort(two));
	}
}

class Point{
	int r;
	int c;
	public Point(int ra,int ca){
		r=ra;
		c=ca;
	}
	public String toString(){
		return r+","+c;
	}
}
