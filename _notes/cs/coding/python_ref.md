---
layout: notes
section-type: notes
title: python ref
category: cs
---

* TOC
{:toc}
# basic

- from math import log2, floor, ceil
- % is modulus
- from random import random
  - `random.random() # [0, 1.0)`
- copying: x.copy() for list, dict; np.copy(x) for numpy

# data structures

```Python
- list [] l (arraylist)
	- l.append(x)
    - l.insert(index, element)
  - queue: from collections import deque # implemented as doubly linked list
      - q = deque()
      - q.append(x)
      - q.pop()
      - q.popleft()
      - q.appendleft(x)
      - index like normal
      - len(q)
  - stack - use normal list
      - pop()
- linked list
	class Node:
		def __init__(self, val, next):
			self.val = val
			self.next = next
- set()
	- add(x)
    - remove(x)
- map {'key': 3}
	- keys()
	- values()
    - del m['key']
- PriorityQueue
	- from queue import PriorityQueue
	- q = PriorityQueue()
    - q.put(x)
    - q.get()
```
# useful

*strings*

```python
- s = "test"
- s.upper()
- reversed(s)
- "".join(s)
- "_".join([s, s])
  	- fastest way to join lots of strings
- s.split("e")
- s.replace("e", "new_str") # replaces all
- s.find("t") # returns first index
- formatting
	"%05d"	//pad to fill 5 spaces
	"%8.3f" //max number of digits
	"%-d"	//left justify
	"%,d" 	//print commas ex. "1,000,000"
	| int | double | string |
	|-----|--------|--------|
	| d   | f      | s      |
	- print("%05d" % x)
- int("3") = 3
- bin(10) = '0b1010'
- hex(100) = '0x64'
- ord('a') = 97
```

*sorting*

```python
l = ['abc', 'ccc', 'd', 'bb']
- sorted(l, reverse=False, key=len) # decreasing order
	- key examples: str.lower, func_name
    - key = lambda x: x[1]
    - slightly faster: key=operator.itemgetter(1)
    - def func_name(s):
     	 return s[-1]
- l.sort(reverse=False, key=len) # sorts in place
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

# higher level

- *primitives* - `byte, short, char, int, long, float, double`, bool
- only primitive and reference *types*
  - when you assign primitives to each other, it's fine
  - when you pass in a primitive, its value is copied
  - when you pass in an object, its reference is copied
    - you can modify the object through the reference, but can't change the object's address

# object-oriented

```python
class Document:
    def __init__(self, name):    
        self.name = name
 
    def show(self):             
        raise NotImplementedError("Subclass must implement abstract method")
 
class Pdf(Document):
    def show(self):
        return 'Show pdf contents!'
 
class Word(Document):
    def show(self):
        return 'Show word contents!'
 
documents = [Pdf('Document1'),
             Pdf('Document2'),
             Word('Document3')]
 
for document in documents:
    print document.name + ': ' + document.show()
```

- *class method* = *static*
  - @classmethod
  - called with Foo.DoIt()
  - initialized before constructor
  - class shares one copy, can't refer to non-static
- *instance method* - invoked on specific instance of the class
  - @staticmethod
  - called with f.DoIt()
- *protected* member is accessible within its class and subclasses