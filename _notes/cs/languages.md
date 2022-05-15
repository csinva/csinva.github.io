---
layout: notes
title: languages
category: cs
---

* TOC
{:toc}

# python

## basic

- from math import log, log2, floor, ceil
- % is modulus
- from random import random
  - `random.random() ## [0, 1.0)`
- copying: x.copy() for list, dict, numpy
  - this is still shallow (won't recursively copy things like a list in a list)
    - this does work for basic things though (the copy is not exactly the same as the original) 
  - for generic objects, need to do from copy import deepcopy

## data structures

```python
- list l (use this for stack as well) ## implemented like an arraylist
	- l.append(x)
    - l.insert(index, element)
  - l.pop()
  - ['x'] + ['y'] = ['x', 'y']
  - [True] * 5
  - l.extend(l2) # add all elements of l2 to l1
  - l.index('dog') # index of element, throws err if not found
- queue: from collections import deque ## implemented as doubly linked list
      - q = deque()
      - q.append(x)
      - q.pop()
      - q.popleft()
      - q.appendleft(x)
      - index like normal
      - len(q)
- class Node: ## this implements a linkedlist
		def __init__(self, val, next):
			self.val = val
			self.next = next
- set()
	- add(x)
  - remove(x)
  - intersection(set2)
- dict {'key': 3}
	- keys()
	- values() - returns values as a list
    - del m['key']
- PriorityQueue
	- from queue import PriorityQueue
	- q = PriorityQueue()
    - q.put(x)
    - q.get()
- from collections import Counter
	- Counter(Y_train) ## this counts unique values and makes it into a dict of counts
```
## useful

*strings*

```python
- s = 'test', 
- s.upper() ## convert to all upper case
- s[::-1] ## reverse the str
- "_".join([s, s]) ## fastest way to join lots of strings (with _ between them)
- s.split("e") ## split into a list wherever there is an e
- s.replace("e", "new_str") ## replaces all instances
- s.index("t") ## list index function
- s.find("t") ## returns first index, otherwise -1
- formatting
	- f"{x:05d}" # f-string
	- "05d"	# pad to fill 5 spaces
	- "8.3f" # max number of digits
	- "-d"	# left justify
	- ",d" # print commas ex. "1,000,000"
  - d (int), f (float), s (str)
- int("3") = 3
- bin(10) = '0b1010'
- hex(100) = '0x64'
- ord('a') = 97
- 'x' * 3 = 'xxx'
- 'B' = chr(ord('A') + 1)
```

*sorting*

```python
l = ['abc', 'ccc', 'd', 'bb']
- sorted(l, reverse=False, key=len) ## decreasing order
	- example keys: values for key
  	- key=str.lower
    - key=func_name
    - key=lambda x: x[1]
    - def func_name(s):
     	 return s[-1]
- l.sort(reverse=False, key=len) ## sorts in place
```

*exceptions*
```python
try:
    something...
except ValueError as e:
    print('error!', e)

raise Exception('spam', 'eggs')
assert(x == 3)
```

## higher level

- *primitives*: `int, float, bool, str`
- only primitive and reference *types*
  - when you assign primitives to each other, it's fine
  - when you pass in a primitive, its value is copied
  - when you pass in an object, its reference is copied
    - you can modify the object through the reference, but can't change the object's address

## object-oriented

```python
## example of a class
class Document:
    def __init__(self, name):    
        self.name = name
 
    def show(self):             
        raise NotImplementedError("Subclass must implement abstract method")

## example of inheritance
class Pdf(Document):
    def show(self):
        return 'Show pdf contents!'

## example of different types of methods
class MyClass:
    def method(self): ## can modify self (and class)
        return 'instance method called', self

    @classmethod
    def classmethod(cls): ## can only modify class
        return 'class method called', cls

    @staticmethod
    def staticmethod(): #can't modify anything
        return 'static method called'
```

## numpy/pandas

```python
df.loc indexes by val
df.iloc indexes by index position
df.groupby returns a dict

merging
pd.merge(df1, df2, how='left', on='x1')
```

## pytorch

- `model = nn.DataParallel(model)`
  
  - automatically runs multiple batches from dataset at same time
  
  - `nn.DistributedDataParallel` is often faster - replicates model on each gpu and gives some data to each one (less data transferes)
- dataset has `__init__, __getitem__, & __len__`
  
  - rather than storing images, can load image from filename in `__getitem__`
- there's a `torch.nn.Flatten` module



## parallelization

- cores = cpus = processors - inividual processing units available on a single node
- node - different computers with distinc memory
- processes - instances of a program executing on a machine
  - shouldn't have more user processes than cores on a node
- more...
  - threads - multiple paths of execution within a single process
    - like a process, but more lightweight
- passing messages between nodes (e.g. for distributed memory) often use protocol known as MPI
  - packages such as Dask do this for you, without MPI
- python packages
  - dask: parallelizing tasks, distributed datasets, one or more machines
  - ray: parallelizing tasks, building distributed applications, one or more machines
- dask
  - separate code from code to parallelize things
  - some limitations on pure python code, but np/pandas etc. parallelize better
  - configuration: specify sequential ('synchronous'), threads within current session ('threaded') paralell ('processes') multiprocessing - this creates copies of objects, multi-node or not ('distributed')
  - need to wrap functions with `delayed()` or `map()`
    - alternatively can tag functions
  - map() operation does caching
  - dask dataframes
  - ssh setup
    - can connecto to other workers locally (ssh-dask or SSHCluster)
    - when using slurm, must run dask-scheduler on cluster node
      - then run dask-worker many times (as many tasks as there are)
    - can also directly submit slurm jobs from dask

# c/c++

- The C memory model: global, local, and heap variables. Where they are stored, their properties, etc.
- Variable scope
- Using pointers, pointer arithmetic, etc.
- Managing the heap: malloc/free, new/delete

## C basic
- #include <stdio.h>
- printf("the variable 'i' is: %d", i);
- can only use /* */ for comments
- for constants: ```#define MAX_LEN 1024```

## malloc
- malloc ex. 
- There is no bool keyword in C

```c
  /* We're going to allocate enough space for an integer 
     and assign 5 to that memory location */
  int *foo;
  /* malloc call: note the cast of the return value
     from a void * to the appropriate pointer type */
  foo = (int *) malloc(sizeof(int));  
  *foo = 5;
   free(foo);
```
- `char *some_memory = "Hello World";`
- this creates a pointer to a read-only part of memory
- it's disastrous to free a piece of memory that's already been freed
- variables must be declared at the beginning of a function and must be declared before any other code

## memory
- heap variables stay - they are allocated with malloc
- local variables are stored on the stack
- global variables are stored in an initialized data segment

## structs
```c
  struct point {
    int x;
    int y;
  };
 struct point *p;
 p = (struct point *) malloc(sizeof(struct point));
 p->x = 0;
 p->y = 0;
```

## strings
- array with extra null character at end '\0'
- strLen doesn't include null character

## pointers
```c
int fake = NULL;
int val = 20;
int * x; // declare a pointer
x = &val; //take address of a variable
- can use pointer ++ and pointer -- to get next values
```

//Hello World
#include <iostream>
using namespace std; //always comes after the includes, like a weaker version of packages
//everything needs to be in a namespace otherwise you have to writed std::cout to look in iostream - you would use this for very long programs
int main(){ //main function, not part of a class, must return an int
    cout << "Hello World" << endl;
    return 0; //always return this, means it didn't crash
}

Preprocessor
    #include <iostream> //System file - angle brackets
    #include "ListNode.h" //user file - inserts the contents of the file in this place
    #ifndef - "if not defined"
    #define - defines a macro (direct text replacement)
    #define TRUE 0 //like a final int, we usually put it in all caps
    if(TRUE ==0)
    #define MY_OBJECT_H //doesn't give it a value - all it does is make #ifdef true and #ifndef false
    #if/#ifdef needs to be closed with #endif
    if 2 files include each other, we get into an include loop
        we can solve this with the header of the .h files - everything is only defined once
        odd.h: 
            #ifndef ODD_H
            #define ODD_H
            #include "even.h"
            bool odd (int x);
            #endif
        even.h:
            #ifndef EVEN_H
            #define EVEN_H
            #include "odd.h"
            bool even (int x);
            #endif

I/O
    #include <iostream>
    using namespace std; 
    int main(){
        int x;
        cout << "Enter a value for x: "; //the arrows show you which way the data is flowing
        cin >> x;
        return 0;
    }

C++ Primitive Types
    int
        can be 16,32,64 bits depending on the platform
    double better than char

If statement can take an int, if (0) then false.  Otherwise true.
    //don't do single equals instead of double equals, will return false

Compiler: clang++
    
Functions - you can only call methods that are above you in the file
    function prototype - to compile mutually recursive functions, you need to declare the function with a semicolon instead of brackets and no body.
    bool even(int x); //called forward declaration / function prototype
    bool odd(int x){
        if(x==0)
            return false;
        return even(x-1);
    }
    bool even(int x){
        if(x==0)
            return true;
        return odd(x-1);
    }
Classes
    Need 3 Separate files:
    1. Header file that contains class definition - like an interface - IntCell.h
        #ifndef INTCELL_H //all .h files start w/ these
        #define INTCELL_H
        class IntCell{
            public: //visibility blocks, everything in this block is public
                IntCell(int initialValue=0); //if you don't provide a parameter, it assumes it is 0.  You can call it with 1 or no parameters.
                ~IntCell(); //destructor, takes no parameters        
        int getValue() const; //the const keyword when placed here means the method doesn't modify the object
                void setValue(int val);
            private:
                int storedvalue;
                int max(int m);
        };
        #endif //all .h files end w/ these
    2. C++ file that contains class implementation -IntCell.cpp
        #include "IntCell.h" 
        using namespace std; // (not really necessary, but...)
        IntCell::IntCell( int initialValue ) :  //default value only listed in .h file
                  storedValue( initialValue ) { //put in all the fieldname(value), this is shorthand
        }
        int IntCell::getValue( ) const { 
            return storedValue; 
        }
        void IntCell::setValue( int val ) { //this is how you define the body of a method
            storedValue = val; 
        } 
        int IntCell::max(int m){
            return 1;
        }
    3. C++ file that contains a main() - TestIntCell.cpp
        #include <iostream>
        #include "IntCell.h"
        using namespace std;
        int main(){
            IntCell m1; //calls default constructor - we don't use parentheses!
            IntCell m2(37);
            cout << m1.getValue() << " " << m2.getValue() << endl;
            m1 = m2; //there are no references - copies the bits in m2 into m1
            m2.setValue(40);
            cout << m1.getValue() << " " << m2.getValue() << endl;
            return 0;
        }
               
Pointers
    Stores a memory address of another object //we will assume everyhing is 32 bit
        Can be a primitive type or a class type
    int * x;
        pointer to int
    char *y;
        pointer to char
    Rational * rPointer;
        pointer to Rational
    all pointers are 32 bits in size because they are just addresses
    in a definition, * defines pointer type: int * x;
    in an expression, * dereferences: *x=2; (this sets a value for what the pointer points to)
    in a definition, & defines a reference type
        &x means get the address of x
    int x = 1;              //Address 1000, value 1 - don't forget to make the pointee
    int y = 5;              //Address 1004, value 5
    int * x_pointer = &x;   //Address 1008, value 1000
    cout << x_pointer;      //prints the address 1000
    cout << *x_pointer;     //prints the value at the address
    *x_pointer = 2;         //this changes the value of x to 2
    x_pointer = &y;         //this means x_pointer now stores the address of y
    *x_pointer = 3;         //this changes the value of y to 3
               
    int n = 30;
    int * p;                //variables are not initialized to any value
    *p = n;                 //this throws an error because you have not requested enough memory, unless it happens to be pointing to memory that you have allocated
    int *p = NULL;          //this will still crash, but it is a better way to initialize
               
    void swap(int * x, int * y) {
        int temp = *x;      //temp takes the value x is pointing to
        *x = *y;            //x points to the value that y was pointing to
        *y = temp;          //y points to the value 3
    }                       //at the end, x and y still are the same addresses
              
    int main() {
        int a=0;
        int b=3;
        cout << "Before swap(): a: " << a << "b: " 
             << b << endl;
        swap(&b,&a);
        cout << "After swap(): a: " << a << "b: " 
             << b << endl;
        return 0;
    }
Dynamic Memory Allocation   //not very efficient
    Static Memory Allocation - the compiler knows at compile time how much memory is needed
        int someArray[10];  //declare array of 10 elements
        int *value1_address = &someArray[3]; // declare a pointer to int
    new keyword
        returns a pointer to newly created "thing"
        int main() {
            int n;
            cout << "Please enter an integer value: " ;         // read in a value from the user
            cin >> n;
            int * ages = new int [n];// use the user's input to create an array of int using new
            for (int i=0; i < n; i++) {                // use a loop to prompt the user to initialize array
                cout << "Enter a value for ages[ " << i << " ]: ";
                cin >> ages[i];
            }
            for(int i=0; i<n; i++) {            // print out the contents of the array
                cout << "ages[ " << i << " ]: " << ages[i];
            delete [] ages;  //finished with the array - clean up the memory used by calling delete
            return 0;        //everything you allocate with new needs to be deleted, this is faster than java
        }
    Generally, SomeTypePtr = new SomeType;
        int * intPointer = new int;
        delete intPointer; //for array, delete [] ages; -this only deals with the pointee, not the pointer
    Accessing parts of an object
        regular object:
            Rational r;
            r.num = 3;
        for a pointer, dereference it:
            Rational *r = new Rational();
            (*r).num=4; //or r->num = 4; (shorthand)
    char* x,y; //y is not a pointer!  Write like char *x,y;
Linked Lists
    List object keeps track of size, pointers to head, tail
        head and tail are dummy nodes
    ListNode holds a value, previous, and next
    ListItr has pointer to current ListNode
Friend
    class ListNode {
    public:
        ListNode();                //Constructor
    private:                       //only this class can modify these fields
        int value;
        ListNode *next, *previous; //for doubly linked lists
        friend class List;         //these classes can bypass private visibility
        friend class ListItr;
    };
Constructor - just has to initialize fields
    Foo() {
        ListNode* head = new ListNode(); //because we put the class type ListNode*, then we are creating a new local variable and not modifying the field
                                        //head = new Listnode() - this works
    }
    Foo() {
      ListNode temp;
      head = &temp;                 //this ListNode is deallocated after the constructor ends, doesn't work
    }
    Assume int *x has been declared
    And int y is from user input
    Consider these separate C++ lines of code:
    x = new int[10]; // 40 bytes
    x = new int;     // 4 bytes
    x = new int[y];  // y*4 bytes
sizeof(int) -> tells you how big an integer is (4 bytes)
References - like a pointer holds an address, with 3 main differences
    1. Its address cannot change (its address is constant)
    2. It MUST be initialized upon declaration
        Cannot (easily) be initialized to NULL
    3. Has implicit dereferencing
        If you try to change the value of the reference, it automatically assumes you mean the value that the reference is pointing to
    //can't use it when you need to change ex. ListItr has current pointer that changes a lot
    Declaration
        List sampleList
        List & theList = sampleList;//references has to be initialized to the object, not the address
    void swap(int & x, int & y) {   //this passes in references
        int temp = x;
        x = y;
        y = temp;
    }
    int main() {                    //easier to call, references are nice when dealing with parameters
        int a=0;
        int b=3;
        cout << "Before swap(): a: " << a << "b: "
             << b << endl;
        swap(b,a);
        cout << "After swap(): a: " << a << "b: " 
             << b << endl;
        return 0;
    }
    You can access its value with just a period
    Location	       *	           &
    Definition	"pointer to"	"reference to"
    Statement	"dereference"	"address of"
subroutines 
    methods are in a class
    functions are outside a class
Parameter passing
    Call by value - actual parameter is copied into formal parameter
        This is what Java always does - can be slow if it has to copy a lot
            -actual object can't be modified
    Call by reference - pass references as parameters
        Use when formal parameter should be able to change the value of the actual argument
        void swap (int &x, int &y);
    Call by constant reference - parameters are constant and are passed by reference
        Both efficient and safe
        bool compare(const Rational & left, const Rational & right);
    Can also return by different ways
C++ default class
    1. Destructor                                           //this will do nothing
        Frees up any resources allocated during the use of an object
    2. Copy Constructor                                     //copies something over
        Creates a new object based on an old object
        IntCell copy = original;                         //or Intcell copy(original)
        automatically called when object is passed by value into a subroutine
        automatically called when object is returned by value from a subroutine
    3. operator=()
        also known as the copy assignment operator
        intended to copy the state of original into copy
        called when = is applied to two objects after both have been previously constructed
            IntCell original;   //constructor called
            IntCell copy;
            copy = original;    //operator called
        overrides the = operator    //operator overrides only work on objects, not pointers
    (and a default constructor, if you don't supply one) //this will do nothing        
            
C++ has visibility on the inheritance

class Name {
public:
    Name(void) : myName("") { }
    ~Name(void) {  }
    void SetName(string theName) {
        myName = theName;
    }
    void print(void) {
        cout << myName << endl;
    }

private:
    string myName;
};
    
class Contact: public Name { //this is like contact extends name
public:
    Contact(void) {
        myAddress = "";
    }
    ~Contact(void) { }
    void SetAddress(string theAddress) {
        myAddress = theAddress;
    }
    void print(void) {
        Name::print();  //this can't access private fields in Name, needs to call print from super class
        cout << myAddress << endl;
    }
private:
    string myAddress;
};

C++ has multiple inheritance - you can have as many parent classes as you want
    class Sphere : public Shape, public Comparable, public Serializable {
    };

Dispatch
    Static - Decision on which member function to invoke made using compile-time type of an object
        when you have a pointer
        Person *p;
        p = new Student();
        p.print();  //will alway call the Person print method - uses type of the pointer
    Dynamic - Decision on which member function to invoke made using run-time type of an object
        Incurs runtime overhead
            Program must maintain extra information
            Compiler must generate code to determine which member function to invoke
        Syntax in C++: virtual keyword 
            (Java does this by default, i.e. everything is virtual)
        Example
            class A 
                virtual void foo()            
            class B : public A
                virtual void foo()
            void main () 
                int which = rand() % 2;
                A *bar;
                if ( which )
                    bar = new A();
                else
                    bar = new B();
                bar->foo();
                return 0;
            
        Virtual method tables - stores the virtual methods in an array
            Each object contains a pointer to the virtual method table
                In addition to any other fields
            That table has the addresses of the methods
                Any virtual method must follow the pointer to the object... (one pointer dereference)
                Then follow the virtual method table pointer... (second pointer dereference)
                Then lookup the method pointer
                    In Java default is Dynamic
                    In C++, default is Static - this is faster
            When creating a subclass object, the constructor of each subclass overwrites the appropriate pointers in the virtual method table with the overridden method pointers

Abstract Classes
    class foo {
        public:
          virtual void bar() = 0;
    };

Types of multiple inheritance
    1. Shared
        What Person is in the diagram on the previous slide
    2. Replicated (or repeated)
        What gp_list_node is in the diagram on the previous slide
    3. Non-replicated (or non-repeated)
        A language that does not allow shared or replicated (i.e. no common ancestors allowed)
    4. Mix-in
        What Java (and others) use to fake multiple inheritance through the use of interfaces
    
    In C++, replicated is the default
        Shared can be done by specifying that a base class is virtual:
            class student: public virtual person, public gp_list_node {
            class professor: public virtual person, public gp_list_node {
                
    Java has ArrayStoreException - makes sure the thing you are adding to the array is of the correct type
        String[] a = new String[1];
        Object[] b = a;
        b[0] = new Integer (1);


# java

## data structures

```java
- LinkedList, ArrayList
	- add(Element e), add(int idx, Element e), get(int idx)
	- remove(int index)
	- remove(Object o)
- Stack
	- push(E item)
	- peek()
	- pop()
- PriorityQueue
	- peek()
	- poll()
	- default is min-heap
	- PriorityQueue(int initialCapacity, Comparator<? super E> comparator)
	- PriorityQueue(Collection<? extends E> c)
- HashSet, TreeSet
	- add, remove
- HashMap
	- put(K key, V value)
	- get(Object key)
	- keySet()
	- if you try to get something that's not there, will return null
```
- default init capacities all 10-20
- clone() has to be cast from Object

## useful
*iterator*

```java
- it.next() - returns value
- it.hasNext() - returns boolean
- it.remove() - removes last returned value
```

*strings*

```java
- String.split(" |\\.|\\?") //split on space, ., and ?
- StringBuilder
	- much faster at concatenating strings
	- thread safe, but slower
	- StringBuilder s = new StringBuilder(CharSequence seq)();
	- s.append("cs3v");
	- s.charAt(int x), s.deleteCharAt(int x), substring
	- s.reverse()
	- Since String is immutable it can safely be shared between many threads
- formatting
	String s = String.format("%d", 3);
	"%05d"	//pad to fill 5 spaces
	"%8.3f" //max number of digits
	"%-d"	//left justify
	"%,d" 	//print commas ex. 1,000,000
	| int | double | string |
	|-----|--------|--------|
	| d   | f      | s      |
	new StringBuilder(s).reverse().toString()
	int count = StringUtils.countMatches(s, something);
- integer
	- String toString(int i, int base)
	- int parseInt(String s, int base)
- array
	char[] data = {'a', 'b', 'c'};
	String str = new String(data);
```

*sorting*

```java
- Arrays.sort(Array a)
- Collections.sort(Collection c), Collections.sort(Collection l, Comparator c)
	- use mergeSort (with insertion sort if very small)
- Collections.reverseOrder() returns comparator opposite of default
class ComparatorTest implements Comparator<String>
	public int compare(String one, String two) //if negative, one comes first
class Test implements Comparable<Object>
	public int compareTo(Object two)
```

*exceptions*
- ArrayIndexOutOfBoundsException
- `throw new Exception("Chandan type")`

## higher level
- *primitives* - `byte, short, char, int, long, float, double`
- java only has primitive and reference *types*
  - when you assign primitives to each other, it's fine
  - when you pass in a primitive, its value is copied
  - when you pass in an object, its reference is copied
    - you can modify the object through the reference, but can't change the object's address
- *garbage collection*
  - once an object no longer referenced, gc removes it and reclaims memory
  - jvm intermittently runs a mark-and-sweep algorithm
    - runs when short-term stuff gets full
    - older stuff moves to different part
    - eventually older stuff is cleared

## object-oriented
| declare | instantiate | initialize |
| ------- | ----------- | ---------- |
| Robot k | new         | Robot()    |
- *class method* = *static*
  - called with Foo.DoIt()
  - initialized before constructor
  - class shares one copy, can't refer to non-static
- *instance method* - invoked on specific instance of the class
  -  called with f.DoIt()
- *protected* member is accessible within its class and subclasses

# R

```r
x %%2 # modulus
x <- 3 # assignment
class(x) checks the class of x
rm(list=ls())
```
- *vectors*
	- numeric_vector <- c(1, 2, 3)
	- poker_vector <- c(140, -50, 20, -120, 240)
	- names(poker_vector) <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
		- assigns names to elements of poker vector
- *matrices*
	- matrix(1:9, byrow = TRUE, nrow = 3)
	- can name the rows / cols
	- 1-indexed
	- has slicing
	- dim(m) prints dimensions
- *factor* - data type for storing categorical variable
- *data frame* - when you want different types of data
	- columns are variables, rows are observations
- *lists* - ordered, can hold any data type
	- length(list)