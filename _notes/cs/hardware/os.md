---
layout: notes
title: os
category: cs
---
* TOC
{:toc}

# 1 - introduction
## 1.1 what operating systems do
- *computer system* - hierarchical approach = layered approach
	1. hardware
	2. operating system
	3. application programs
	4. users
- views
	1. user view - OS maximizes work user is performing
	2. system view
		- os allocates resources - CPU time, memory, file-storage, I/O
		- os is a *control program* - manages other programs to prevent errors
- program types
	1. os is the *kernel* - one program always running on the computer
		- only kernel can access resources provided by hardware
	2. system programs - associated with OS but not in kernel
	3. application programs
- *middleware* - set of software frameworks that provide additional services to application developers

## 1.2 computer-system organization
- when computer is booted, needs *bootstrap program*
	- initializes things then loads OS
	- also launches *system processes*
		- ex. Unix launches "init"
- events
	- hardware signals with *interrupt*
	- software signals with *system call*
	- *interrupt vector* holds addresses for all types of interrupts
	- have to save address of interrupted instruction
- memory
	- *von Neumman architecture* - uses instruction register
	- main memory is *RAM*
		- *volatile* - lost when power off	
	- *secondary storage* is non-volatile (ex. hard disk)
	- *ROM* is unwriteable so static programs like bootstrap are ROM
	- access
		1. uniform memory access (*UMA*)
		2. non-uniform memory access (*NUMA*)
- I/O
	- *device driver* for I/O devices
	- *direct memory access (DMA)* - transfers entire blocks of data w/out CPU intervention
		- otherwise device controller must move data to its local buffer and return pointer to that
- multiprocessor systems
	1. increased throughput
	2. economies of scale (costwise)
	3. increased reliability (fault tolerant)

## 1.3 computer-system architecture 
1. single-processor system - one main cpu
	- usually have special-purpose processors (e.g. keyboard controller)
2. *multi-processor system* / *multicore system*
	- *multicore* means multi-processor on same chip
		- multicore is generally faster
	- multiple processors in close communication
	- advantages
		- increased throughput
		- economy of scale
		- increased reliability = *graceful degradation* = *fault tolerant*
	- types
		1. *asymmetric multiprocessing* - boss processor controls the system
		2. *symmetric multiproccesing (SMP)* - each processor performs all tasks
			- more common
		3. *blade server* - multiple independence multiprocessor systems in same chassis
3. *clustered system* - multiple loosely coupled cpus
	- types
		1. *asymmetric clustering* - one machine runs while other monitors it (*hot-standby mode*)
		2. *symmetric clustering* - both run something
	- parallel clusters
		- require *disributed lock manager* to stop conflicting parallel operations
		- can share same data via *storage-area-networks*
	- *beowulf cluster* - use ordinary PCs to make cluster

## 1.4 operating-system structure
- *multiprogramming* - increases CPU utilization so CPU is always doing something
	- keeps *job pool* ready on disk
	- *time sharing / multitasking* - multiple jobs switch so fast that both can be interacted with
		- requires an *interactive* computer system
	- *process* - program loaded into memory
	- scheduling
		- *job scheduling* - picking jobs from job pool (disk -> memory)
		- *CPU scheduling* - what to run first (memory -> cpu)
	- memory
		- processes are *swapped* from main memory to disk
		- *virtual memory* allows for execution of process not in memory

## 1.5 operating-system operations
- *trap / exception* - software-generated *interrupt*
- *user-mode* and *kernel mode* (also called system mode)
	- when in kernel mode, *mode bit* is 0
	- separate mode for virtual machine manager (VMM)
	- this is built into hardware
- kernel can use a timer to getting stuck in user mode

## 1.6 process management
- program is passive, process is active
- process needs resources
	- process is unit of work
- single-threaded process has one *program counter*

## 1.7 memory management
- cpu can only directly read from main memory
- computers must keep several programs in memory
	- hardware design is impmortant

## 1.8 storage management
- defines *file* as logical storage unit
- most programs stored on disk until loaded
- in addition to secondary storage, there is *tertiary storage* (like DVDs)
- *caching* - save frequent items on faster things
	- *cache coherency* - make sure cache coherency is properly updated with parallel processes

## 1.9 protection & security
- process can execute only within its address space
- *protection* - controlling access to resources
- *security* - defends a system from attacks
- maintain list of *user IDs* and *group IDs*
	- can temporarily *escalate priveleges* to an *effective UID* - *setuid* command

## 1.10 basic data structures
- *bitmap* - string of n binary digits

## 1.11 computing environments
- *network computers* - are essentially terminals that understand web-based computing
- *distributed system* - shares resources among separate computer systems
	- *network* - communication path between two or more computers
	- *TCP/IP* is most common network protocol
- networks
	- *PAN* - personal-area network (like bluetooth)
	- *LAN* - local-area network connects computers within a room, building, or campus
	- *WAN* - wide-area network
	- *MAN* - metropolitan-area network
	- *network OS* provides features like file sharing across the network
	- *distributed OS* provides less autonomy - makes it feel like one OS controls entire network
- *client-server* computing
	1. *compute-server* - performs actions for user
	2. *file-server* - stores files
- *peer-to-peer* computing
	1. all clients w/ central lookup service, ex. Napster
	2. no centralized lookup service
		- uses *discovery protocol* - puts out request and other peer must respond
- *virtualization* - allows OS to run within another OS
	- *interpretation* - run programs as non-native code (ex. java runs on JVM)
	- BASIC can be compiled or interpreted
- *cloud-computing* - computing, storage, and applications as a service accross a network
	- public cloud
	- private cloud
	- hybrid cloud
	- software as a service (SAAS)
	- platform as a service (PAAS)
	- infrastructure as a service (IAAS)
	- cloud is behind a firewall, can only make requests to it
- *embedded systems* - like microwaves / robots 
	- specific tasks
	 - have *real-time OS* - fixed time constraints

# 2 - OS Structures
## 2.1 os services
- for the user
	- *user interface* - command-line interface and graphical user interface
	- *program execution* - load a program and run it
	- *I/O operations* - file or device
	- *File-system manipulation*
	- *communications* - between processes / computer systems
	- *error detection* 
- for system operation
	- *resource allocation*
	- *accounting* - keeping stats on users / processes
	- *protection / security* 

## 2.2 user and os interface
1. *command interpreter* = *shell* - gets and executes next user-specified command
	- could contain the code to execute the command
	- command interpreter could have code to execute commands
	- more often, executes *system programs*, such as "rm", that are executed
2. *GUI*

## 2.3 system calls
- *system calls* - provide an interface to os services
- API usually wraps system calls (ex. java)
	- *libc* - provided by Linux/Mac OS for C
	- *system-call interface* links API calls to system calls
- passing parameters
	1. pass parameters in registers
	2. parameters stored in block of memory and address passed in register
	3. parameters pushed onto stack

## 2.4 system call types
1. process control - halting, ending
	- *lock* shared data - no other process can access until released
2. file manipulation
3. device manipulation
	- similar to file manipulation
4. information maintenance - time, date, dump()
	- *single step* is CPU mode which throws trap for CPU after every instruction for a debugger
5. communications
	1. *message-passing model*
		- each computer has *host name* and *network identifier* (IP address)
		- each process has *process name*
		- *daemons* - system programs for receiving connections (like servers waiting for a client)
	2. *shared-memory model*
6. protection

## 2.5 system programs
- *system programs* = *system utilities* 
- some provide interfaces for system calls
- other uses
	1. file management
	2. status info
	3. file modification
	4. programming-language support
	5. program loading and execution
	6. communications
	7. background services

## 2.6 os design and implementation
- *mechanism* - how to do something
	- want this to be general so only certain parameters change
- *policy* - what will be done
- os mostly in C, low-level kernel in assembly
	- high-level is easier to port but slower

## 2.7 os structure
- want modules but current models aren't very modularized
	- *monolithic* system has performance advantages - very little overhead
	- in practice everything is a hybrid
- system can be modularized with a *layered approach*
	- layers: hardware, ..., user interface
	- easy to construct and debug
	- hard to define layers, less efficient
- *microkernel approach* - used in os *Mach* 
	- move nonessential kernel components to system / user-level
	- smaller kernel, everything communicates with *message passing*
	- makes extending os easier, but slower functions due to system overhead
- *loadable kernel modules*
	- more flexible - kernel modules can change
- examples (see pics)

## 2.8 os debugging
- errors are written to *log file* and *core dump* (memory snapshot) is written to file
- if kernel crashes, must save its dump to s special area
- *performance tuning* - removing *bottlenecks*
	- monitor *trace listings* - log if interesting events with times / parameters
- SolarisDTrace is a tool to debug and tune the os
- *profiling* - periodically samples instruction pointer to determine which code is being executed

## 2.9 generation
- *system generation* - configuring os on a computer
	- usually on a CD-ROM
	- lots of things must be determined (like what CPU to use)

## 2.10 system boot
- bootstrap program

# 3 - processes
## 3.1 process concept
- *process* - program in execution
	- batch system executes *jobs* = *processes*
	- time-shared system has user programs or *tasks*
	- program is passive while process is active
- parts
	- program code - *text section*
	- program counter
	- registers
	- stack
	- data section
	- heap
- same program can have many processes
- process can be execution environment for other code (ex. JVM)
- *process state*
	- new
	- running
	- waiting
	- ready
	- terminated
- *process control block (PCB)* = *task control block* - repository for any info that varies process to process
	- process state
	- program counter
	- CPU registers
	- CPU-scheduling information
	- memory-management information
	- accounting information
	- I/O status information
	- could include information for each thread
- *parent* - process that created another process

## 3.2 process scheduling
- *process scheduler* - selects available process for multi-tasking
	- processes begin in *job queue*
	- processes that are ready and waiting are in the *ready queue* until they are *dispatched* - usually stored as a linked list
	- lots of things can happen here (fig 3_6)
		- ex. make I/O request and go to I/O queue
	- *I/O-bound process* - spends more time doing I/O
	- *CPU-bound process* - spends more time doing computations
	- each device has a list of process waiting in its *device queue*
- *scheduler* - selects processes from queues
	- *long-term scheduler* - selects from processes on disk to load into memory
		- controls the *degree of multiprogramming* = number of processes in memory
		- has much more time than short-term scheduler
		- want good mix of *I/O-bound* and *CPU-bound* processes
		- sometimes this doesn't exist
	- *short-term / CPU scheduler* - selects from processes ready to execute and allocates CPU to one of them
	- sometimes *medium-term scheduler*
		- does *swapping* - remove a process from memory and later reintroduce it
- *context switch* - occurs when switching processes
	- when interrupt occurs, kernel saves *context* of old process and loads saved context of new process
	- context is in the PCB
	- might be more or less work depending on hardware

## 3.3 operations on processes
- usually each process has unique process identifier (*pid*)
- linux everything starts with init process (pid=1)
- restricting a child process to a subset of the parent's resources prevents system overload
	- parent may pass along initialization data
- after creating new process
	1. parent continues to execute concurrently with children
	2. parent waits until some or all of its children terminate
- two address-space possibilities for the new process:
	1. child is duplicate of parent (it has the same program and data as the parent).
	2. child loads new program
- forking
	- when call *fork()* continue operation but returns 0 for parent process and nonzero for child
		- child is a copy of the parent
	- after fork, usually one process calls *exec()* to load binary file into memory
		- overrides program, doesn't return unless error occurs
	- parent can call *wait()* until child finishes (moves itself off ready queue until child finishes)
- on Windows, uses *CreateProcess()* which requires loading a new program rather than sharing address space
	- STARTUPINFO - 
	- PROCESSINFORMATION - 
- process termination
	- *exit()* kills process (return in main calls exit)
	- process can return status value
	- parent can terminate child if it knows its pid
	- *cascading termination* - if parent dies, its children die
	- *zombie process* - terminated but parent hasn't called wait() yet
		- remains because parent wants to know what exit status was
		- if parent terminates without wait(), *orphan* child is assigned *init* as new parent (init periodically invokes wait())
	
## 3.4 interprocess communication
- process *cooperation*
	- information sharing
	- computation speedup
	- modularity
	- convenience
- *interprocess communication (IPC)* - allows exchange of data and info
	1. *shared memory* - shared region of memory is established
		- one process establishes region
		- other process must attach to it (OS must allow this)
		- less overhead (no system calls)
		- suffers from cache coherency
		- ex. producer consumer
			- producer fills buffer and consumer empties it
			- *unbounded buffer* - producer can keep producing indefinitely
			- *bounded buffer* - consumer waits if empty, producer waits if full
			- in points to next free position
			- out points to first full position
	2. *message passing* - messages between coordinating processes
		- useful for smaller data
		- easier in a distributed system
		1. *direct or indirect communication*
			- direct requires knowing the id of process to send / receive
				- can be *asymmetrical* - need to know id of process to send to, but not receive from
				- results in hard-coding
			- indirect - messages are sent / received from mailboxes
				- more flexible, can send message to whoever shares mailbox
				- mailbox owned by process - owner receives those messages
				- mailbox owned by os - unclear
		2. *synchronous or asynchronous* communication
			- synchronous = blocking
			- when both send and recieve are blocking = *rendezvous*
		3. *automatic or explicit* buffering
			- queue for messages can have 3 implementations
				1. *zero capacity (must be blocking)*
				2. *bounded capacity*
				3. *unbounded capacity*

## 3.5 examples of IPC systems
1. POSIX - shared memory
2. Mach - message passing
3. Windows - shared memory for message passing

## 3.6 communication in client-server systems
1. *sockets* - endpoint for communication
	- IP address + port number
	- connecting
		1. server listens on a port
		2. client creates socket and requests connection to server's port
		3. server accepts connection (then usually writes data to socket)
	- all ports below 1024 are well known
	- *connection-oriented*=*TCP*
	- *connectionless* = *UDP*
	- special IP address 127.0.0.1 - *loopback* - refers to itself
	- sockets are *low-level* - can only send unstructured bytes
2. *remote procedure calls (RPCs)* - remote message-based communication
	- like IPC, but between different computers
	- message addressed to an RPC daemon listening to a port
	- messages are well-structured
	- specifies a *port* - a number included at the start of a message packet
		- system has many ports to differentiate different services
	- uses *stubs* to hide details
		- they *marshal* the parameters
		- might have to convert data into *external data representation (XDR)* (to avoid issues like big-endian vs. little-endian)
	- must make sure each message is acted on *exactly once*
	- client must know port
		1. binding info (port numbers) may be predetermined and unchangeable
		2. binding can be dynamic with rendezvous deaemon (*matchmaker*) on a fixed RPC port
3. *pipes* - conduit allowing 2 processes to communicate
	- four issues
		1. bidirectional?
		2. full duplex (data can travel in both directions at same time?) or half duplex (only one way)?
		3. parent-child relationship?
		4. communicate over a network?
	- *ordinary pipe* - write at one end, read at the other
		- unix function `pipe(int fd[])`
			- fd[0] is read-end and fd[1] is write-end
		- only exists while a child and parent process are communicating
			- therefore only on same machine
		- parent and child should both close unused ends of the pipe
		- on windows, called *anonymous pipes*
			- requires security attributes
	- *named pipe* - can be bidirectional
		- called *FIFOs* in Unix
			- only half-duplex, requires same machine
		- Windows - fulll-duplex and can be different machines
		- many processes can use them
		
# 4 - threads
- *thread* - basic unit of CPU utilization
	1. program counter
	2. register set
	3. stack
- making a thread is quicker and less resource-intensive than making a process
- used in RPC and kernels
- benefits
	1. responsiveness
	2. resource sharing
	3. economy
	4. scalability

## 4.2 - multicore programming (skipped)
- *amdahl's law*: $speedup \leq \frac{1}{S+(1-S)/N_{cores}}$
	- S is serial portion
- parallelism
	- *data parallelism* - distributing subsets of data across cores and performing same operation on each core
	- *task parallelism* - distribution tasks across cores

## 4.3 - multithreading models
- need relationship between *user threads* and *kernel threads*
	1. *many-to-one model* - maps user-level threads to one kernel thread
		- can't be parallel on multicore systems
		- ex. used by *Green threads*
	2. *one-to-one model*
		- small overhead for creating each thread
		- used by Linux and Windows
	3. *many-to-many model*
		- less than or equal number of kernel threads
		- *two-level model* mixes a one-to-one model and a many-to-many model
		
## 4.4 - thread libraries
- *thread library* - provides programmer with an API for creating/managing threads
- *asynchronous* v. *synchronous* threading

1 - POSIX Pthreads

```
/* get the default attributes */
pthread attr init(&attr);
/* create the thread */
pthread create(&tid,&attr,runner,argv[1]);  // runner is a func to call
/* wait for the thread to exit */
pthread join(tid,NULL);
```
- shared data is declared globally

2 - Windows

3 - Java
	- uses Runnable interface

## 4.5 - implicit threading (skipped)
- *implicit threading* - handle threading in run-time libraries and compilers
	1. *thread pool* - number of threads at startup that sit in a pool and wait for work
	2. *OpenMP* - set of compiler directives / API for parallel programming
		- identifies *parallel regions*
		- uses #pragma
	3. *Grand central dispatch* - extends C
		- uses *dispatch queue*

## 4.6 - threading issues
- fork/exec need to know if should fork all threads / when to replace program
- *signal* notifies a process that a particular event has occurred
	1. has a default signal handler
	2. user-defined signal handler
	- delivering a signal to a process: `kill(pid_t pid, int signal)`
	- delivering a signal to a thread: `pthread_kill(pthread_t tid, int signal)`
- *thread cancellation* - terminating *target thread* before it has completed
	1. *asynchronous cancellation* - one thread immediately terminates target thread
	2. *deferred cancellation* - target thread periodically checks whether it should terminate
	- pthread_cancel(tid)
		- uses deferred cancellation
		- cancellation occurs only when thread reaches *cancellation point*
- *thread-local storage* - when threads need separate copies of data
- *lightweight process* = *LWP* - between user thread and kernel thread
- *scheduler activation* - kernel provides application with LWPs
	- *upcall* - kernel informs application about certain events
	
## 4.7 - linux (skipped)
- linux process / thread are same = task
- uses clone() system call

# 5 - process synchronization
- *cooperating process* can effect or be affected by other executing processes
- ex. consumer/producer
	- if counter++ and counter-- execute concurrently, don't know what will happen
	- this is a *race condition*
	
## 5.2 - critical-section problem
- each process has *critical section* where it updates common variables
	- <img src="pics/5_1.png"/ width=40%>
- 3 requirements
	1. *mutual exclusion* -	2 processes can't concurrently do critical section
	2. *progress* - things should be in critical selection
	3. *bounded waiting* - every process should eventually get to critical selection
- kernels
	1. *preemptive kernels*
		- more responsive
	2. *nonpreemptive kernels*
		- no race conditions

## 5.3 - peterson's solution
- *peterson's solution*
	- <img src="pics/5_2.png" width=40%/>
		- here i is one task and j is the other
	- not guaranteed to work

## 5.4  - synchronization hardware
- *locking* - protecting critical regions using locks
- single-processor solution
	- prevent interrupts while shared variable is being modified
	- ex. `test_and_set()`
- instructions do things like swapping *atomically* - as one uninterruptable unit
	- these are basically locked instructions
	- ex. `compare_and_swap()`
	
## 5.5 - mutex locks
- *mutex*: <img src="pics/5_8.png" width=40%/>
	- simplest synchronization tool
	- this type of mutex lock is called *spinlock* because requires *busy waiting* - processes not in critical section are continuously looping
	- good when locks are short
	
### 5.6 - semaphores
- *semaphore* S - integer variable accessed through *wait()* (like trying to execute) and *signal()* (like releasing)
	- *counting semaphore* - unrestricted domain
	- *binary sempahore* - 0 and 1
	
```c
wait(S) {
	while(S<=0)
		// busy wait
	S--;
}
signal(S) {
	S++;
}
```

- to improve performace, replace busy wait by process blocking itself
	- places itself into a waiting queue
	- restarted when other process executes a signal() operation
	
```c
typedef struct{ 
	int value;
	struct process *list;
} semaphore;
wait(semaphore *S) { 
	S->value--;
	if (S->value < 0)
		add this process to S->list;
}
signal(semaphore *S) { 
	S->value++;
	if (S->value <= 0){
		remove a process P from S->list; 
		wakeup(P); // resumes execution
	}
}
```

- *deadlocked* - 2 processes are in waiting queues, can't wakeup unless other process signals them
- *indefinite blocking=starving* - could happen if we remove processes from waiting queue in LIFO order
	- bottom never gets out
- *priority inversion*
	- only occurs when processes have > 2 priorities
	- usually solved with a *priority-inheritance protocol*
		- when a process accesses resources needed by a higher-priority process, it inherits the higher priority until they are finished with the resources in question
		
## 5.7 - classic synchronization problems
1. bounded-buffer problem
2. readers-writers problem
	- writers must have exclusive access
	- readers can read concurrently
3. dining-philosophers problem

### 5.8 - monitors
- *monitor* - highl-level synchronization construct
	- only 1 process can run at a time
	- *abstract data type* which includes a set of programmer-defined operations with mutual exlusion
	- has *condition* variables
		- these can only call wait() or signal()
		- when a signal is encountered, 2 options
			1. signal and wait
			2. signal and continue
- can implement with a semaphore
	- 1st semaphore: `mutex` - process must wait before entering and signal after leaving the monitor
	- 2nd semaphore: `next` - signaling processes use next to suspend themselves
	- 3rd semaphore: `next_count` = number of suspended processes
	
```
 wait(mutex);
// body of F

if (next count > 0) 
	signal(next);
else
	signal(mutex);
```

- *conditional-wait* construct can help with resuming
	- `x.wait(c);`
	- *priority number* c stored with name of process that is suspended
	- when `x.signal()` is executed, process with smallest priority number is resumed next
	
## 5.9.4 - pthreads synchronization
```
#include <pthread.h> 
pthread mutex t mutex;

/* create the mutex lock */ 
pthread mutex init(&mutex,NULL) // null specifies default attributes

pthread mutex lock(&mutex); // acquire the mutex lock
/* critical section */
pthread mutex unlock(&mutex); // release the mutex lock
```
- these functions return 0 w/ correct operation otherwise error code
- POSIX specifies *named* and *unnamed* semaphores
	- name has name and can be shared by different processes

```
#include <semaphore.h> sem t sem;
/* Create the semaphore and initialize it to 1 */ sem init(&sem, 0, 1);

/* acquire the semaphore */ 
sem wait(&sem);

/* critical section */

/* release the semaphore */ 
sem post(&sem);
```

## 5.10 - alternative approaches (skip)

## 5.11 - deadlocks
- resource utilization
	1. request
	2. use
	3. release
- deadlock requires 4 simultaneous conditions
	1. mutual exclusion
	2. hold and wait
	3. no preemption
	4. circular wait
- deadlocks can be described by *system resource-allocation graph*
	- *request edge* - directed edge from process P to resource R means P has requested instance of resource type R
	- *assignment edge* - R-> P
	- if the graph has no cycles, not deadlocked
	- if cycle, possible deadlock
- three ways to handle
	1. use protocol to never enter deadlock
	2. enter, detect, recover
	3. ignore the problem
		- developers must write code that avoids deadlocks

	
# 7 - main memory
## 7.1 - background
- CPU can only directly access main memory and registers
- accessing memory is slower than registers
	
	- processor must *stall* or use *cache*
- processes need separate memory spaces
	1. *base register* - holds smallest usable address
	2. *limit register* - specifies size of range
	- os / hardware check these, throw a trap if there was error
- *input queue* holds processes waiting to be be brought into memory
- compiler *binds* symbolic addresses to relocatable addresses
	
	- linkage editor binds relocatable addresses to absolute addresses
- CPU uses *virtual address*=logical address
	- *memory-management unit (MMU)* maps from virtual to *physical address*
		- simple ex. add virtual address to a process's base register = *relocation register*
- *dynamic loading* - don't load whole process, only load things when called
- *dynamically linked libraries* - system libraries linked to user programs when the programs are run
	
	- *stub* - tells how to load / locate library routine
- *shared libraries* - all use same library

## 7.2 (skipped)

## 7.3 - contiguous memory allocation
- *contiguous memory allocation* - each process has a section
	- put OS in low memory and process memory in higher
- *transient OS code* - not often used
	- ex. drivers
	- can remove this and change OS memory usage by decreasing val in OS limit register
- split mem into *partitions*
	- each partition can only have 1 process
	- *multiple-partition method* - free partitions take a new process
	- *variable-partition scheme* - OS keeps table of free mem
		- all available mem = *hole*
		- holes are divided between processes
			1. *first-fit* - allocate first hole big enough
			2. *best-fit* - allocate smallest hole that is big enough
			3. *worst-fit* - allocate largest hole (largest leftover hole)
				- worst
- *external fragmentation* - there is enough free mem, but it isn't contiguous
	- *50-percent rule* - 1/3 of mem is unusable
	- solved with *compaction* - shuffle mem to put free mem together
		- can be expensive to move mem around
- *internal fragmentation* - extra mem a proc is allocated but not using (because given in block sizes)
- 2 types of non-contiguous solutions
	1. segmentation
	2. paging

## 7.4 - segmentation (skip)
- *segments* make up logical address space
	- name (or number)
	- length
- logical address is a tuple
	- (segment-number, offset)
- *segment table*
	- each entry has *segment base* and *segment limit*
- doesn't avoid external fragmentation

## 7.5 - paging (skip)
- break physical mem into fixed-size *frames* and logical mem into corresponding *pages*
- CPU address = [*page number*|*page offset*]
	- *page table* contains base address of each page in physical mem
	- usually, each process gets a page table
	- <img src="pics/7_10.png" width=40%/>
- *frame table* keeps track of which frames are available / who owns them
- paging is prevalent
- avoids external fragmentation, but has internal fragmentation
- small page tables can be stored in registers
	- usually *page-table base register* points to page table in mem
	- has *translation look-aside buffer* - stores some page-table entries
		- some entries are *wired down* - cannot be removed from TLB
		- some TLBS store *address-space identifiers* (ADIDs)
			- identify a process
			- otherwise hard to contain entries for several processes
		- want high *hit ratio*
- page-table often stores a bit for read-write or read-only
	- *valid-invalid* bit sets whether page is in a process's logical address space
	- OR *page-table length register* - says how long page table is
- can share *reentrant code* = *pure code*
	- non-self-modifying code
		
## 7.6 - page table structure (skip)
- page tables can get quite large (total mem / page size)
1. *hierarchical paging* - ex. two-level page table
	- <img src="pics/7_18.png" width=40%/>
	- also called *forward-mapped page table*
	- unused things aren't filled in
	- for 64-bit, would generally require too many levels
2. *hashed page tables*
	- virtual page number is hash key -> physical page number
	- *clustered page tables* - each entry stores everal pages, can be faster
3. *inverted page tables*
	- only one page table in system
	- one entry for each page/frame of memory
	- <img src="pics/7_20.png" width=40%/>
	- takes more time to lookup
		- hash table can speed this up
	- difficulty with shared memory
	
## 7.7-9 (skipped)

# 6 - cpu scheduling
- *preemptive* - can stop and switch a process that is currently running

### 6.3 - algorithms
1. first-come, first-served
2. shortest-job-first
	- can be preemptive or non preemptive
3. priority-scheduling
	- indefinite blocking / starvation
4. round-robin
	- every process gets some time
5. multilevel queue scheduling
	- ex. foreground and background
6. multilevel feedback queues
	- allows processes to move between queues
	
### 6.4 - thread scheduling
- *process contention scope* - competition for CPU takes place among threads belonging to same process
	- PTHREAD_SCOPE_PROCESS - user-level threads onto available LWPs
	- PTHREAD_SCOPE_SYSTEM - binds LWP for each user-level thread
	
### 6.5 - multiple-processor scheduling
- asymmetric vs. symmetric
	- almost everything is symmetric (SMP)
	- *processor affinity* - try to not switch too much
	- *load balancing* - try to make sure all processes share work
	- *multithreading*
		1. coarse-grained - thread executes until long-latency event, such as memory stall
		2. fine-grained - switches between instruction cycle
		
### 6.6 - real-time systems
- *event latency* - amount of time that elapses from when an event occurs to when it is serviced
1. *interrupt latency* - period of time from the arrival of an interrupt at the CPU to the start of the routine that services the interrupt
2. *dispatch latency*
	1. Preemption of any process running in the kernel
	2. Release by low-priority processes of resources needed by a high-priority process
- *rate-monotonic* scheduling - schedules periodic tasks using a static priority policy with preemption

## 6.7 - SKIP

# 8 - virtual memory
## 8.1 - background
- lots of code is seldom used
- virtual mem allows the execution of processes that are not completely in 
- benefits
	- programs can be larger than physical mem
	- more processes in mem at same time
	- less swapping programs into mem
- *sparse* address space - virtual address spaces with hole (betwen heap and stack)
	- <img src="pics/8_3.png" width=40%/>

## 8.2 - demand paging
- *demand paging* - load pages only when they are needed
	- *lazy pager* - only swaps a page into memory when it is needed
	- can use valid-indvalid bit in page table to signal whether a page is in memory
- *memory resident* - residing in memory
- accessing page marked invalid causes *page fault*
	- <img src="pics/8_6.png" width=40%/>
	- must restart after fetching page
		1. don't let anything change while fetching
		2. use registers to store state before fetching
- *pure demand paging* - never bring a page into memory until it is required
	
	- programs tend to have *locality of reference*, so we bring in chunks at a time
- extra time when there is a page fault
	1. service the page-fault interrupt
	2. read in the page
	3. restart the process
	- effective access time is directly proportional to *page-fault rate*
- *anonymous memory* - pages not associated with a file

## 8.3 - copy-on-write
- *copy-on-write* - allows parent and child processes intially to share the same pages
	- if either process writes, copy of shared page is created
	- new pages can come from a set *pool*
- *zero-fill-on-demand* - zeroed out before being allocated
- *virtual memory fork* - not copy-on-write
	- child uses adress space of parent
	- parent suspended
	- meant for when child calls exec() immediately

## 8.4 - page replacement - select which frames to replace
- multiprogramming might *over-allocate* memory 
	
	- all programs might need all their mem at once
- buffers for I/O also use a bunch of mem
- when over-allocated, 3 options
	1. terminate user process
	2. swap out a process
	3. page replacement
- want lowest page-fault rate
- test with *reference string*, which is just a list of memory references
- if no frame is free, find one not being used and free it
- write its contents to swap space
- <img src="pics/8_10.png" width=40%/>
- *modify bit*=*dirty bit* reduces overhead
	
	- if hasn't been modified then don't have to rewrite it to disk
- page replacement examples
	1. FIFO
		
		- *Belady's anomaly* - for some algorithms, page-fault rate may increase as number of allocate frames increases
	2. optimal (OPT / MIN)
		
		- replace the page that will not be used for the longest period of time
	3. LRU - least recently used (last used)
		1. implement with counters since each use
		2. stack of page numbers (whenever something is used, put it on top)
		- *stack algorithms* - set of pages in memory for n frames is always a subset of the set of pages that would be in memory with n + 1 frames 
			- don't suffer from Belady's anomaly
	4. LRU-approximation
		- *reference bit* - set whenever a page is used
		- can keep *additional reference bits* by recording reference bits at regular intervals1
		- *second-chance* algorithm - FIFO, but if ref bit is 1, set ref bit to 0 and move on to next FIFO page
		- can have clock algorithm
		- <img src="pics/8_17.png" width=40%/>
		- *enhanced second-chance* - uses reference bit and modify bit
			- give preference to pages that have been modified
	5. counting-based - count and implement LFU (least frequently used) or MFU (most frequently used)
- page-buffering algorithms
	- pool of free frames - makes things faster
	- list of modified pages - written to disk whenever paging device is idle
	- som algorithms, like databases perform better when they get their own memory capability called *raw disk* instead of being managed by OS
	
## 8.5 *frame-allocation algorithms* - how many frames to allocate to teach process in memory (skipped)

## 8.6 - thrashing
- if low-priority process gets too few frames, swap it out
	- *thrashing* - process spends more time paging than executing
		- CPU utilization stops increasing
- *local replacement algorithm* = *priority replacement algorithm* - if one process starts thrashing, cannot steal frames from another
	- *locality model* - each locality is a set of pages actively used together
	- give process enough for its current locality
- *working-set model* - still based on locality
	- defines *working-set window* $\delta$
	- defines *working set* as pages in most recent $\delta$ refs
	- OS adds / suspends processes according to working set sizes
	- approximate with fixed-interval timer
- *page-fault frequency* - add / decrease pages based on targe page-fault rate

## 8.7 - (skipped)

## 8.8.1 - buddy system
- memory allocated with *power-of-2 allocator* - requests are given powers of 2
	- each page is split into 2 *buddies* and each of those splits again recursively
	- *coalescing* - buddies can be combined quickly
	
# 9 - mass-storage structure
## 9.1
## 9.2
## 9.4 - disk scheduling
- *bandwidth* - total number of bytes transferred, divided by time
- first-come first-served
- shortest-seek-time-frist
- *SCAN* algorithm - disk swings side to side servicing requests on the way
	- also called elevator algorithm
	- also has circular-scan

## 9.5 - disk management
- *low-level formatting* - dividing disk into sectors that *controller* can read/write
	- blocks have header / trailer with error-correcting codes
- *bad blocks* are corrupted - need to replace them with others = *sector sparing* = *forwarding*
	- *sector slipping* - just renumbers to not index bad blocks

# 10 - file-system interface
## 10.1
- os maintains *open-file table*
- might require file locking
- must support different file types

## 10.2 - access methods
- simplest - *sequential*
- *direct access* = *relative access*
	- uses relative block numbers
	
## 10.3
- disk can be partitioned
- two-level directory
	- users are first level
	- directory is 2nd level
- extend this into a tree
	- acyclic makes it faster to search
	- cycles require very slow *garbage collection*
- *link* - pointer to another thing

## 10.4 - file system mounting

# 11 - file-system implementation
## 11.1
- *file-control block (FCB)* contains info about file ownership, etc.

## 11.2
## 11.3 (SKIP)

## 11.4
- contiguous allocation
- linked allocation
	
	- FAT
- indexed allocation - all the pointers in 1 block

	# 11.5	
- keep track of *free-space list*
	
	- implemented as bit map
- keep track of linked list of free space
- *grouping* - block stores n-1 free blocks and 1 pointer to next block
- *counting* - keep track of ptr to next block and the number of free blocks after that

# 12 - i/o systems
- *bus* - shared set of wires	
- registers
	- data-in - read by the host
	- data-out
	- status
	- control
- *interrupt chaining* - each element in the interrupt vector points to the had of a list of interrupt handlers
- system calls use software interrupt
- *direct memory access* - read large chunks instead of one byte at a time
- *device-status table*
- *spool* - buffer for device (ex. printer) that can't hold interleaved data