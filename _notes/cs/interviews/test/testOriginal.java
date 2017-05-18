import java.util.*;
public class test{
	public static void main(String[] args){
		LinkedList<String> l = new LinkedList<String>();
		l.add("a");
		l.add("c");
		l.add("b");
		Collections.sort(l);
		System.out.println(l.get(0));
		Node n = new Node(3);
		n.next = new Node(5);
		Node test = n;
		System.out.println(l.removeFirst());
		System.out.println(l.removeFirst());
		System.out.println(l.removeFirst());
		while(n!=null){
			System.out.println(n.val);
			n = n.next;
		}
		HashSet<String> h = new HashSet<String>();
		h.add("Chandan");
		for(String s:h)
			System.out.println(s);
		HashMap<String,Integer> m = new HashMap<String,Integer>();
		m.put("c",3);
		System.out.println(m.get("c"));
		System.out.println(m.get("b"));
		for(String key:m.keySet())
			System.out.println(key);
		Stack<Integer> st = new Stack<Integer>();
		PriorityQueue<String> pq = new PriorityQueue<String>();
		pq.add("c");
		pq.add("b");
		System.out.println(pq);
		System.out.println(pq.poll());
		System.out.println(pq.poll());
		st.push(3);
		st.push(2);
		st.push(1);
		System.out.println(st);
		System.out.println(st.pop());
		System.out.println(st.peek());
		System.out.println(st.pop());
		
		
	}
}
class Node{
	Node next;
	int val;
	public Node(int val){
		this.val=val;
	}
}