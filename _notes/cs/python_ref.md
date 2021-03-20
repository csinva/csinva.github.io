---
layout: notes
title: python ref
category: cs
---

* TOC
{:toc}

# basic

- from math import log, log2, floor, ceil
- % is modulus
- from random import random
  - `random.random() # [0, 1.0)`
- copying: x.copy() for list, dict, numpy
  - this is still shallow (won't recursively copy things like a list in a list)
    - this does work for basic things though (the copy is not exactly the same as the original) 
  - for generic objects, need to do from copy import deepcopy

# data structures

```python
- list l (use this for stack as well) # implemented like an arraylist
	- l.append(x)
    - l.insert(index, element)
  - l.pop()
  - ['x'] + ['y'] = ['x', 'y']
  - [True] * 5
- queue: from collections import deque # implemented as doubly linked list
      - q = deque()
      - q.append(x)
      - q.pop()
      - q.popleft()
      - q.appendleft(x)
      - index like normal
      - len(q)
- class Node: # this implements a linkedlist
		def __init__(self, val, next):
			self.val = val
			self.next = next
- set()
	- add(x)
  - remove(x)
  - intersection(set2)
- dict {'key': 3}
	- keys()
	- values()
    - del m['key']
- PriorityQueue
	- from queue import PriorityQueue
	- q = PriorityQueue()
    - q.put(x)
    - q.get()
- from collections import Counter
	- Counter(Y_train) # this counts unique values and makes it into a dict of counts
```
# useful

*strings*

```python
- s = 'test', 
- s.upper() # convert to all upper case
- s[::-1] # reverse the str
- "_".join([s, s]) # fastest way to join lots of strings (with _ between them)
- s.split("e") # split into a list wherever there is an e
- s.replace("e", "new_str") # replaces all instances
- s.find("t") # returns first index, otherwise -1
- formatting
	- "%05d"	//pad to fill 5 spaces
	- "%8.3f" //max number of digits
	- "%-d"	//left justify
	- "%,d" 	//print commas ex. "1,000,000"
  - d (int), f (float), s (str)
	- print(f"{x:05d}") # new in 3.6
- int("3") = 3
- bin(10) = '0b1010'
- hex(100) = '0x64'
- ord('a') = 97
- 'x' * 3 = 'xxx'
```

*sorting*

```python
l = ['abc', 'ccc', 'd', 'bb']
- sorted(l, reverse=False, key=len) # decreasing order
	- key examples: str.lower, func_name
    - key = lambda x: x[1]
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

- *primitives*: `int, float, bool, str`
- only primitive and reference *types*
  - when you assign primitives to each other, it's fine
  - when you pass in a primitive, its value is copied
  - when you pass in an object, its reference is copied
    - you can modify the object through the reference, but can't change the object's address

# object-oriented

```python
# example of a class
class Document:
    def __init__(self, name):    
        self.name = name
 
    def show(self):             
        raise NotImplementedError("Subclass must implement abstract method")

# example of inheritance
class Pdf(Document):
    def show(self):
        return 'Show pdf contents!'

# example of different types of methods
class MyClass:
    def method(self): # can modify self (and class)
        return 'instance method called', self

    @classmethod
    def classmethod(cls): # can only modify class
        return 'class method called', cls

    @staticmethod
    def staticmethod(): #can't modify anything
        return 'static method called'
```

# numpy/pandas

- loc indexes by val
- iloc indexes by index position
- .groupby returns a dict
- merging
  - pd.merge(df1, df2, how='left', on='x1')

# pytorch

- `model = nn.DataParallel(model)`
  
  - automatically runs multiple batches from dataset at same time
  
  - `nn.DistributedDataParallel` is often faster - replicates model on each gpu and gives some data to each one (less data transferes)
- dataset has `__init__, __getitem__, & __len__`
  
  - rather than storing images, can load image from filename in `__getitem__`
- there's a `torch.nn.Flatten` module



# parallelization

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