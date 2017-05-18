---
layout: notes
section-type: notes
title: C/C++ Reference
category: cs
---

* TOC
{:toc}

- The C memory model: global, local, and heap variables. Where they are stored, their properties, etc.
- Variable scope
- Using pointers, pointer arithmetic, etc.
- Managing the heap: malloc/free, new/delete

# C basic
- #include <stdio.h>
- printf("the variable 'i' is: %d", i);
- can only use /* */ for comments
- for constants: ```#define MAX_LEN 1024```

# malloc
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

# memory
- heap variables stay - they are allocated with malloc
- local variables are stored on the stack
- global variables are stored in an initialized data segment

# structs
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

# strings
- array with extra null character at end '\0'
- strLen doesn't include null character

# pointers
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
        
    