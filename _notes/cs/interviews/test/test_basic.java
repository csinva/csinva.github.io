import java.util.*;
public class test_basic{
	public static void main(String[] args){
		String s = "hello";
		char[] c = s.toCharArray();	
		HashMap<Integer, String> m = new HashMap<Integer,String>();
		m.put(3,"test");
		System.out.println(m.get(4));
		PriorityQueue<Integer> pq = new PriorityQueue<Integer>(10);
	}
}
