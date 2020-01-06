---
layout: notes
section-type: notes
title: Architecture
category: cs
---

* TOC
{:toc}

---

# units
- we will use only the i versions (don't have to write i):
	- K - 10^3: Ki - 1024
	- M - 10^6: Mi - 1024^2
	- G - 10^9: Gi - 1024^3
- convert to these: 2^27 = 128M
- log(8K)=13
- hardware is parallel by default
- amdahl's law: tells you how much of a speedup you get
    - S = 1 / (1-a+a/k)
    - a-portion optimized, k-level of parallelization, S-total speedup
		if you really want performance increase in java, allocate a very large array, then keep track of it on your own	
	
# numbers
0x means hexadecimal
0 means octal
bit - stores 0 or 1
byte - 8 bits - 2 hex digits
integer - almost always 32 bits
"Big-endian": most significant first (lowest address) - how we thnk
    1000 0000 0000 0000 = 2^5 = 32768
"Little-endian": most significant last (highest address) - this is what computers do
    1000 0000 0000 0000 = 2^0 = 1
Note that although all the bits are reversed, usually it is displayed with just the bytes reversed
    Consider 0xdeadbeef
        On a big-endian machine, that's 0xdeadbeef
        On a little-endian machine, that's 0xefbeadde
        0xdeadbeef is used as a memory allocation pattern by some OSes
Representing integers
    Sign and magnitude - first digit specifies sign
    One's complement - encode using n-1 bits, flip if negative
    Two's complement - encode using n-1 bits, flip if negative, add 1
        only one representation for 0
        maximum: 2^(n-1) - 1
        minimum: - 2^(n-1)
        flip everything to the left of the rightmost 1
Floating point - like scientific notation
    3.24 * 10 ^ -6
    Mantissa - 3.24 - between 1 and the base (10)
        For binary, the mantissa must be between 1 and 2 
    we assume the base is 2
    32 bits are split as follows:
        bit 1: sign bit, 1 means negative (1 bit)
        bits 2-9: exponent (8 bits)
            Exponent values:
                0: zeros
                1-254: exponent-127
                255: infinities, overflow, underflow, NaN
        bits 10-32: mantissa (23 bits)
            mantissa=1.0+∑(i=1:23)(b^i/2^i) //we don't encode the 1. because it has to be there
    value=(1−2∗sign)∗(1+mantissa)∗2^(exponent−127)
    The largest float has:
        0 as the sign bit (it's positive)
        254 as the exponent (1111 1110)
            255 is reserved for infinities and overflows
        That exponent is 254-127 = 127
        All 1's for the mantissa
            Which yields almost 2
        2 * 2^127 = 2^128 = 3.402823 * 10^38    //actually a little bit lower
    Minimum positive:
        1 * 2^-126 = 2^-126 = 1.175494 x 10^-38 //this is exact
    Floating point numbers are not spatially uniform
        Depending on the exponent, the difference between two successive numbers is not the same
union class - converts from one data type to another //when you write one field, it overrides the other
    union foo {  //this converts a float to hex
        float f;
        int *x;
    } bar;

    int main() {
        bar.f = 42.125;
        cout << bar.x << endl; // this outputs as 0x42288000 (it is now converted to hex)
    }                          // if you were to dereference it, bad things would happen
Never compare floating point numbers - even if you print them out, they might be stored internally
Any fraction that doesn't have a power of 2 as the denominator will be repeating
                       // C++ (need to #include <math.h> and compile with -lm)
define EPSILON 0.000001
bool foo = fabs (a-b) < EPSILON;
You could use a rational or use more digits
64-bit: 11 exponent bits
    offset=1023
Cowlishaw encoding: use 3 bits to store a binary digit - inefficient
	
​	
# x86
Assembly Language - assembler translates text into machine code
x86 is the type of chip
8 registers, although you can't use 2 (stack pointer and base pointer)
    they are all 32 bit registers
1 byte = 8 bits
Declare variables with 3 things:
    identifier, how big it is, and value
    doesn't give you a type
    ? means uninitialized
    x	DD	 	1, 2, 3  //declares arr with 3, 4-byte integers
    y	TIMES 8 DB	0    //declares 8 bytes all with value 0
nasm assumes you are using 32 bits
mov <dest>, <src>
    more like copy
    Where dest and src can be:
        A register
        A constant
        Variable name
        Pointer: [ebx]
    always put square brackets around variable
    you can ADD up to two registers, add one constant, and premultipy ONE register by 2,4,or 8
The destination cannot be a constant (would overwrite the constant)
You cannot access memory twice in one instruction
    not enough time to do that at clock speed
Stack starts at the end of memory and goes backwards
    when you push onto it, it's actually at a lower index
    ESP points to most recently pushed item
    push
        First decrements ESP (stack pointer) by 4 (stack grows down)
        push (mov) operand onto stack (4 bytes - we make this assumption, not always true)
    pop
        Pop top element of stack to memory or register, then increment stack pointer (ESP) by 4
        Value is written to the parameter
Commands
    0fH - H at end specifies hex number
    lea is like & (get the address of)
        Load effective address
        Place address of second parameter into the first parameter 
        this is faster than arithmetic because you can do things as a a single command
    add, sub
        a += b
    inc, dec
        a++ 
    imul
        a *= b
    idiv - use shift if possible
        have to load a,b into one 64-bit integer
    and, xor - bitwise
    cmp - compare two things
        je - jump when equal - specify where you are going to jump to
        Others: jne, jz, jg, jge, jl, jle, js
    call <label> - subroutine call
        pushes address of next instruction onto stack
        then jumps to label
    ret - returns from subroutine
Calling conventions
    A set of rules/expectations between functions
    Using a stack for calling convention is implemented on most processors. Why? - Recursion
        Parameters: pushed on the stack
        Registers: saved on the stack
            eax,ecx,edx can be modified
            ebx,edi,esi shouldn't be 
        call - places return address on stack
        Local variables: placed in memory on the stack
        Return value: eax
    Callee: the function which is called by another function
Register Usage
    Three registers may be modified by the callee: eax, ecx, edx
    If the caller wants to keep those values, they need to be saved by pushing them onto the stack
        Return value: eax register
Varying number of parameters
    Method overloading
        Foo::bar(int) and Foo::bar(int, float) - this just creates two methods
    Default parameters
        Foo::bar (int x = 3) - value defaults to three, method always gets one parameter
    Variable number of parameters
        #include <cstdarg>
        #include <iostream>
        using namespace std;
        double average (int num, ...) { //num is the number of arguments following
          va_list arguments;
          double sum = 0;
          va_start (arguments, num);
          for ( int x = 0; x < num; x++ )
            sum += va_arg (arguments, double);
          va_end (arguments);
          return sum / num;
        }
        int main() {
          cout << average(3, 12.2, 22.3, 4.5) << endl;
          cout << average(5, 3.3, 2.2, 1.1, 5.5, 3.3) << endl;
        }
Caller: the function which calls another function
    Prologue
        Tasks to take care of BEFORE calling a subroutine
        Call the subroutine with the call opcode
    Epilogue
        Tasks to complete AFTER subroutine call returns
        (It is not really called this, but I use this to parallel the equivalent components in the callee convention)
    Before calling the function (the prologue)
        Save registers that might be needed after the call (eax, ecx, edx)
        Push parameters on the stack
    Call the function
        call instruction places return address in stack
    After the called function returns (the epilogue)
        Remove parameters from stack
        Restore saved registers
pop saves the value and the increments esp by 4
if we don't need to save the values (like with arguments we passed in), we can just increment esp    
Callee
    parameters are above ebp
    local variables are under ebp
    sub esp,4 - doing this at the beginning makes space for local variables
    ...
    return value is placed into eax //we won't focus on things that don't return 4 bytes, like returning a double
    mov esp,ebp - this undoes the above command.  We could do add esp,4 but if it were more complicated this would still work.

    subroutine may not know how many parameters are passed to it - thus, 1st arg must be at ebp+8 and the rest are pushed above it.
    Every subroutine call puts return address and ebp backup on the stack

Activation Records
    Every time a sub-routine is called, a number of things are pushed onto the stack:
        Registers
        Parameters
        Old base/stack pointers
        Local variables
        Return address
    The callee also pushes caller-saved registers
    Typically stack stops around 100-200 Megabytes, although this can be changed
        
Memory - There are two types of memory that need to be handled:
    Dynamic memory (via new, malloc(), etc.)
        This is stored on the heap
    Static memory (on the stack)
        This is where the activation records are kept

    The binary program starts at the beginning of the 2^32 = 4 Gb of memory
    The heap starts right after this
    The stack starts at the end of this 4 Gb of memory, and grows backward
    If they meet, you run out of memory

Buffer Overflow
    void security_hole() {
        char buffer[12];
        scanf ("%s", buffer); // how C handles input
    }
    The stack looks like (with sizes in parenthesis):

     esi (4) 	 edi (4) 	 buffer (12) 	 ebp (4) 	 ret addr (4) 
    
    Addresses increase to the right (the stack grows to the left)
    What happens if the value stored into buffer is 13 bytes long?
         We overwrite one byte of ebp
    What happens if the value stored into buffer is 16 bytes long?
         We completely overwrite ebp
    What if it is exactly 20 bytes long?
         We overwrite the return address!
    Buffer Overflow Attack
         When you read in a string (etc.) that goes beyond the size of the buffer
         You can then overwrite the return address
         And set it to your own code
         For example, code that is included later on in the string - overwrite ebp, overwrite ret addr with beginning of malicious code

We are using nasm as our assembler for the x86 labs
    looks different when you use the compiler
in C, you can only have one method with the same name
    C translates more cleanly into assembly
optimization rearranges stuff to lessen memory access
_Z3maxii:
    ii is the parameter list (two integers)
         
In little-Endian, the entire 32-bit word and the 8-bit least significant byte have the same address
    this makes casting very easy
RISC
    Reduced instruction set computer
    Fewer and simpler instructions (maybe 50 or so)
    Less chip complexity means they can run fast
CISC
    Complex instruction set computer
    More and more complex instructions (300-400 or so)
    More chip complexity means harder to make run fast

Caller
  Parameters: pushed on the stack
        Registers: saved on the stack
            eax,ecx,edx can be modified
            ebx,edi,esi shouldn't be 
        call - places return address on stack
        Local variables: placed in memory on the stack
        Return value: eax
Callee: the function which is called by another function
    push ebp
    mov ebp, esp
    sub esp, 4 //allocate local variables
    push ebx   //you don't have to back these up
    mov ebp-4, 1 //load 1 into local variable

    add esp, 4  //deallocate local var
    pop ebx
    pop ebp
    ret

# intro to C - we use ANSI standard
```java
all C is valid C++
// doesn't work
always use +=1 not ++
compile with gcc -ansi -pedantic -Wall -Werror program.c
	-Werror will stop the program from compiling
all variables have to be declared at the top
	int main(int argc, char*argv[]){
		int x = 1;
		int y = 34;
		int z = y*y/x;
		x = 13;
		int w = 1; <- this will not work
	label_name:
		printf("omg!");
		goto label_name; /*this goes to the label_name line - don't do this, but assembly only has this*/
		return 0;
	}
printf(const char *format, ...)
	printf("%d %f %g %s\n",); /* int, double (these must be explicitly doubles), double (as small as possible), string */
```
# compile steps
source (text) -> pre-processor -> modified source (text) -> compiler -> assembly (text) -> assembler -> binary program -> linker -> executable

1. pre-processing - deals with hashtags - sets up line numbers for errors, includes external definitions, normal defines (.i)
2. compile - turns it into assembly (.s)
	- this assembly has commands with destination, src
3. assemble - turns assembly into binary with a lot of wrappers (.o)
4. link - makes the file executable, gets the code from includes together (a.out)

# strings
- char - number that fits in 1 byte
- string is an array of chars: char*
- all strings end with null character \0, bad security
- length of string doesn't include null character

h|e|l|l|o|\0
-|
10|.|.|.|.|15

# memory in C
- byte is smallest accessible memory unit - 2 hex digits (ex: 0x3a)

Bits | Name | Bits | Name
- | 
1 | bit | 16 | word
4 | nyble | 32 | double word, long word
8 | byte | 64 | quad word

theoretically would work:

```java
Void * p = 3 (address 3)
*p = 0x3a (value 3a)
p[0] = 0x3a (value 3a)
p[3] is same as *(p+3) - can even use negative addresses, (at end wraps around - overflows)
```
in practice:

```java
int* p
sizeof(int) == 4, but all pointers are just one byte - points to location of four consecutive bytes that are int
indexing this pointer will tell you how much to offset memory by
address must be 4-bytes aligned (means it will be a multiple of 4)
```
- little-endian - least significant byte first
- big-endian - most significant byte first (normal) - networks use this
- we will use little-endian, this is what most computers use
- low addresses are unused - will throw error if accessed
- top has the os - will throw error if accessed
- contains heap, stack, code globals, shared stuff

# call stack
1. return address
2. local variables
3. backups
4. top pointer
5. base pointer
6. next pointer
7. parameters (sometimes)
8. return values (sometimes)

one frame - between the lines is everything the current method has - largest addresses at top, grows downwards

- parameters (backwards)
- ret address

---
- base pointer
- previous stack base
- saved stuff
- locals
- top pointer (actually at the bottom)
- return value

---
- in practice, most parameters and return values are put in registers

# types
- 2's complement: positive normal, negative subtract 1 more than biggest number you can do
	- flip everything to the left of the rightmost one
	- math is exactly the same, discard extra bits
	
type | signed? | bits
- | 
char | ? | 8
short | signed | 16 (usually)
int | . | 16 or 32
long | . | ≤16 or ≥ int
long long | signed | 64
- everything can have an unsigned / signed in front of the type
- C will cast things without throwing error
	
# boolean operators
- 9 = 1001 = 1.001 x 2^3
- x && y -> {0,1}
- x & y -> {0,1,...,2^32} (bit operations)
- ^ is xor
- !x - 0 -> 1, anything else -> 0
- and is often called masking because only the things with 1s go through
- shifting will pad with 0s
	 (1 << 3 )-1 			gives us 3 1s
	 ~((1 << 3 )-1)		gives us 111...111000
	 x & ~((1 << 3 )-1)	gives us x1x2.....000
- >> copies the msb
- then we can or it in order to change those last 3 bits
- trinary operator - a≥b:c means if(a) b; else c 
- a&b | ~a&c
	- only works for 2-bit numbers if a=00 or a=11
	-!x 1 if x=0
	-!!x 1 if x!=0 so we want a=-!!x
                                 
# ATT x86 assembly
- there are 2 hardwares
	- x86-64 (desktop market)
	- Arm7 (mobile market) - all instructions are like cmov
- think -> not = ex. mov $3, %rax is 3 into rax
- prefixes
	- $34 - $ is an immediate (literal) value
	- %rax - the contents of register rax
	- main - label (no prefix) - assembler turns this into an immediate
	- 3330(%rax,%rdi,8) - memory at (3330+rax+8*rdi) - in x86 but not y86
		- you could do 23(%rax)
- gcc -S will give you .S file w/ assembly
	- what would actually be compiled
- gcc -c will give you object file
- then, objdump -d will dissassemble object file and print assembly
	- leaves out suffixes that tell you sizes
	- can't be compiled, but is informative
- gcc -O3 will be fast, default is -O0, -OS optimizes for size
- we call the part of x86 we use y86
- registers
	- general purpose registers (program registers)
	- PC - program counter - cannot directly change this - what line to run next
	- CC - condition codes - sign of last math op result
		- remembers whether answer was 0,-,+
- cmp - basically subtraction (works backwards, 2nd-1st), but only sets the condition codes
- in assembly, arithmetic works as +=
	- ex. imul %rdi, %rax multiplies and stores result in rdi
- doesn't really matter: eax, rax are same register but eax is bottom half of rax
	- on some hardwares eax is faster than rax
- call example
		- PC=0x78 callq 0x56
	- PC=0x7d next command (because callq is 5 bytes long, it could be different)
		- puts 7d on stack to return to at address 0x100
		- this address (0x100) is subtracted by number of bytes in address (8)
		- this value (0x0f8) is put into rsp(in C this is always on the stack)
			- rsp stores address of the top of the stack
		- PC becomes 56
- call
	- movq (next PC), (%rsp) ~PC can't actually be changed
	- addq $-8, %rsp
	- jmp $0x56
- ret
	- addq $8, %rsp
	- movq (%rsp), (PC)  ==same as== jmp (%rsp)
- push
	- mov _, (%rsp)
	- sub $8, %rsp
- pop does the opposite
	- add $8, %rsp
		 movq (%rsp), _	
cmp
	- cmovle %5, (%rax) - move only if we are in this state
	
# y86 - all we use
1. halt - stops the chip
2. nop - do nothing
3. op_q
	- addq, subq, andq, xorq
	- takes 2 registers, stores values in second
		- sub is 2nd-1st
4. jxx
	- jl, jle, jg, je, jne, jmp
	- takes immediate
5. movq longer PC increment because it stores offset, register always comes first (is rA)
	- rrmovq (register-register, same as cmov where condition is always)
	- irmovq (immediate-register)
	- rmmovq
	- mrmovq (memory)
6. cmovxx (register-register)
7. call
	- takes immediate
	- pushes the return address on the stack and jumps to the destination address.
8. ret
	- pops a value and jumps to it
9. pushq
	- one register
	- pushes then decrements
10. popq
	- one register

- programmer-visible state
	- registers
		- program
			- rax-r14 (8 registers x86 has 15)
			- rsp is the special one
			- 64 bit integer (or pointer) - there is no floating point, in x86 floating point is stored in other registers
		- other
			- program counter (PC), instruction pointer
				- 64-bit pointer
			- condition codes (CC) - not important, tell us <, =, > on last operation
				- only set by the op_q commands
	- memory - all one byte array
		- instruction
		- data
- encoding (assembly -> bytes)
	- 1st byte -> high-order nybble | low-order nybble
		- higher order is opcode (add, sub, ...)
		- lower-order is either instruction function or flag(le, g, ...) - usually 0
	- remaining bytes 
		- argument in little-endian order
	- examples
		- call $0x123			-> 80 23 01 00 00 00 00 00 00
			 ret 					-> 90
			 subq %rcx, %r11		-> 81 1b (there are 15 registers, specify register with one nybble)
			 irmov $0x3330, %rdi	-> 30 f7 30 33 00 00 00 00 00 00 (register-first always, f means no source, but destination of register 7)
	- compact binary (variable-length encoding) vs. simple binary (fixed-length encoding)
		- x86 vs. ARM
		- people can't decide
		- compact binary - complex instruction set computer (cisc) - emphasizes programmer
		- simple binary - reduced instruction set computer (risc) - emphasizes hardware
			- have more complex compilers
			- fixed width instructions
			- lots of registers
			- few memory addressing modes - no memory operands (only mrmov, rmmov)
			- few opcodes
			- passes parameters in registers, not the stack (usually)
			- no condition codes (uses condition operands)
		- in general, computers use compact and tablets/phones use simple
		- if we can get away from x86 backwards compatibility, we will probably meet in the middle
		- study RISC vs. CISC
		
# hardware
- flows when control is high
- power - everything loses power by creating heat (every gate consumes power)
	- changing a transistor takes more power than leaving it
- voltage - threshold above which transistor is open
- register - on rising clock edge store input
- overclock computer - could work, or logic takes too long to get back - things break
	- could be fixed with colder, more power
		chips are small because of how fast they are	
- mux - selectors pick which input goes through
- out = [
	guard:value;
		...
];
- this is a language called HCL written by the book's authors
- out = g?input: g2:input2: ...:0
	- if first is true return that, otherwise keep going otherwise return 0

# executing instructions
- we have wires
1. register file
2. data memory
3. instruction memory
4. status output - 3-bits
- ex. popq, %rbx
  - todo: get icode, check if it was pop, read mem at rsp, put value in rbx, inc rsp by 8
  - getting icode
  	- instruction in instruction memory: B0 3F
  	- register pP (p is inputs), (P is outputs)
  		- pP { pc:64 = 0;} - stores the next pc
  	- pc <- P_pc - the fixed functionality will create i10 bytes
  	- textbook: icode:ifun = M_1[PC] - gets one byte from PC
  	- HCL (HCL uses =): 
  		wire icode:4;
  		icode = i10bytes[4..8] - little endian values, one byte at a time - this grabs B from B0 3F
  - assume icode was b (in reality, this must be picked with a mux)
  ```java
  	valA 	<- R[4] 		// gets %rsp - rsp is 4th register
  	rA:rB	<- M_1[PC+1]	// book's notation - splits up a byte into 2 halves, 1 byte in rA, 1 byte in rB, PC+1 because we want second byte
  							// 3 is loaded into rA, F is loaded into rB
  	valE 	<- valE+8		// inc rsp by 8
  	valM 	<- M_8[valA] 	// send %rsp to 
  	R[rA] 	<- valM			// writes to %rbx 
  	R[4]	<- valE			// writes to %rsp
  	p_pc	=  P_pc+2		// increment PC	by 2 because popq is 2-byte instruction	
  ```
```
- steps
​```java
	1. fetch - what is wanted
	2. decode - find what to do it to - read prog registers
	3. execute and/or memory - do it
	4. write back - tell you result

020 10		
	nop						fetch		change pc to 021
021 6303	
	xorq 	%rax,%rbx		fetch 		pc <- 0x023
							decode 		read reg. file at 0,3 to get 17 and 11
							execute		17^11 = 10001^01011 = 11010 = 26, also sets CC to >0
							write back	write 26 to regFile[3]
023 50 23 1000000000000000 	the immediate 16 is in little-endian but is 8 bytes - 1st in memory last in little-endian
	mrmovq  16(%rbx),%rcx	fetch 		read bytes, understand, PC <- 0x02d
							decode		read regFile to get (26),13
							execute		16+26=42 (to be new address)
							memory		ask RAM for address 42, it says 0x0000000071000000 - little-endian
							write back	put 0x71000000 into regFile[2]
02d 71 0000000032651131				
    jle 0x3111653200000000 	fetch 		valP <- 0x036
							decode cc > 0, not jump, PC <- valP
036	00 
	halt - set STAT to HALT and the computer shuts off, STAT is always going on in the background
push
```
	- reads source register
	- reads rsp
	- dec rsp by 8
	- writes read value to new rsp address


# hardware wires - opq example 
```java
# fetch
# 1. set pc
# 1.a. make a register to store the next PC
register qS {
	pc : 64 = 0;
	lt : 1 = 0;
	eq : 1 = 0;
	gt : 1 = 0;
}

# 2. read i10bytes
pc = S_pc;

# 3. parse out pieces of i10bytes
wire icode:4, ifun:4, rA:4, rB:4;
icode = i10bytes[4..8]; # 1st byte: 0..8  high-order nibble 4..8
ifun = i10bytes[0..4]; 

const OPQ = 6;
const NO_REGISTER = 0xf;

rA = [
	icode == OPQ : i10bytes[12..16];
	1: NO_REGISTER;
];

rB = [
	icode == OPQ : i10bytes[8..12];
	1: NO_REGISTER;
];

wire valP : 64;

valP = [
	icode == OPQ : S_pc + 2;
	1 : S_pc + 1; # picked at random
];


Stat = STAT_HLT; # fix this

# decode

# 1. set srcA and srcB

srcA = rA;
srcB = rB;

dstE = [
	icode == OPQ : rB;
	1 : NO_REGISTER;
];

# execute

wire valE : 64;

valE = [
	icode == OPQ && ifun == 0 : rvalA + rvalB;
	1 : 0xdeadbeef;
];

q_lt = [
	icode == OPQ : valE < 0;
	# ...
];

# memory

# writeback

wvalE = [
	icode == OPQ : valE;
	1: 0x1234567890abcdef;
];


# PC update
q_pc = valP; 
```

# pipelining 
- nonuniform partitioning - stages don't all take same amount of time
- register - changes on the clock
	- normal - output your input
	- bubble - puts "nothing" in registers - register outputs nop
	- stall - put output into input (same output)
- see notes on transitions
- stalling a stage (usually because we are waiting for some earlier instruction to complete)
	- stall every stage before you
	- bubble stage after you so nothing is done with incomplete work - this will propagate
	- stages after that are normal
- bubbling a stage - the work being performed should be thrown away
	- bubble stage after it
	- basically send a nop - use values NO_REGISTER and 0s
		- often there are some fields that don't matter
- stalling a pipeline = stalling a stage
- bubbling a pipeline - bubble all stages

```java
irmovq	 	$31, %rax
addq 		$rax,%rax
jle
```

- the stages are offset - everything always working
- when you get a jmp, you have to wait for the thing before you to writeback.  2 possible solns
	- stall decode 
	- forward value from the stage it is currently in
- look at online notes - save everything in register that needs to be used later

### problems
1. dependencies - outcome of one instruction depends on the outcome of another - in the software
	1. data - data needed before advancing - destination of one thing is used as source of another
		- load/use
	2. control - which instruction to run depends on what was done before
2. hazards - potential for dependency to mess up the **pipeline** - in the hardware design
	- hardware may or may not have a hazard for a dependency
	- can detect them by comparing the wire that reads / writes to regfile (rA,rB / dstE) - they shouldn't be the same because you shouldn't be reading/writing to the same register (except when all NO_REGISTER)

### solutions
- P is PC, then FDEMW
1. stall until it finishes if there's a problem
	stall_P = 1;	//stall the fetch/decode stage
	bubble_E = 1;   //completes and then starts a nop, gives it time to write values
	. forward values (find what will be written somewhere)	
	- in a 2-stage system, we have dstE and we use it to check if there's a problem
	- usually if we can check that there's a problem, we have the right answer
	- if we have the answer, put value where it should be
	- this is difficult, but doesn't slow down hardware
	- we decide whether we can forward based on a pipeline diagram - boxes with stages (time is y axis, but also things are staggered)
		- we need to look at when we get the value and when we need it
		- we can pipeline if we need the value after we get it 
		- if we don't, we need to stall
```java
subq %rax,%rbx
jge bazzle
```
- we could stall until CC is set in execute of subq to fetch for jge - this is slow
1. speculation execution - we make a guess and sometimes we'll be right (modern processors are right ~90%)
	- branch prediction - process of picking branch
	- jumps to a smaller address are taken more often than not - this algorithm and more things make it complicated
	- if we were wrong, we need to correct our work
- Example: 
```java
1	subq
2	jge 4
3	ir
4	rm
```
- the stage performing ir is wrong - this stage needs to be bubbled
- in a 5-stage pipleine like this, we only need to bubble decode
- ret - doesn't know the next address to fetch until after memory stage, also can't be predicted well
	- returns are slow, we just wait
	- some large processors will guess, can still make mistakes and will have to correct
	
### real processors
1. memory is slow and unpredictably slow (10-100 cycles is reasonable)
2. pipelines are generally 10-15 stages (Pentium 4 had 30 stages)
3. multiple functional units
	- alu could take different number of cycles for addition/division
	- multiple functional units lets one thing wait while another continues sending information down the pipeline
4. out-of-order execution
	- compilers look at whether instructions can be done out of order
	- it might start them out of order so one can compute in functional unit while another goes through
	- can swap operations if 2nd operation doesn't depend on 1sts
5. (x86) turn the assembly into another language
	- micro ops
	- makes things specific to chips
- profiler - software that times software - should be used to see what is taking time in code
- hardware vs. software
	- software: .c -> compiler -> assembler -> linker -> executes
		- compiler - lexes, parses, O_1, O_2, ...
	- hardware: register transfer languages (.hcl) -> ... -> circuit-level descriptions -> layout (veryify this) -> mask -> add silicon in oven -> processor
	
# memory
- we want highest speed at lowest cost (higher speeds correspond to higher costs)
- fastest to slowest
	- register - processor - about 1K
	- SRAM - memory (RAM) - about 1M
	- DRAM - memory (RAM) - 4-16 GB
	- SSD (solid-state drives) - mobile devices
	- Disk/Harddrive - filesystem - 500GB
- the first three are volatile - if you turn off power you lose them
- the last three are nonvolatile
- things have gotten a lot faster
- where we put things affects speed 
	- registers near ALU
	- SRAM very close to CPU
	- CPU also covered with heat sinks
	- DRAM is large - pretty far away (some other things between them)
- locality
	- temporal - close in time - the characteristic of code that repeatedly uses the same address 
	- spatial - close in space - the characteristic of code that uses addresses that are numerically close
- real-time performance is not big-O - a tree can be faster than hash because of locality
- caching - to keep a copy for the purpose of speed
	- if we need to read a value, we want to read from SRAM (we call SRAM the cache)
	- if we must read from DRAM (we call DRAM the main memory), we usually want to store the value in SRAM
	- cache - what they wanted most recently (good guess because of temporal locality)
	- slower caches are still bigger
		- cache nearby bytes to recent acesses (good guess because of spatial locality)
- simplified cache
	- 4 GB RAM -> 32-bit address
	- 1MB cache
	- 64-bit words
	- ex. addr: 0x12345678
		- simple - use entire cache as one block
		- chop off the bottom log(1MB)=20 bits
		- send addr = 0x12300000
		- fill entire cache with 1MB starting at that address
		- tag cache with 0x123
		- send value from cache at address offset=0x45678
		- if the tag of the address is the same as the tag of the cache, then return from cache
			- otherwise redo everything
- slightly better cache
	- beginning of address is tag
	- middle of address might give you index of block in table = log(num_blocks_in_cache)
	- end of address is block offset (offset size=log(block_size))
	- a (tag,block) pair is called a line
	1. Fully-associative cache set of (tag,block) pairs
		- table with first column being tags, second column being blocks
		- to read, check tag against every tag in cache
			- if found, read from that block
			- else, pick a line to evict (this is discussed in operating systems)
				- read DRAM into that line, read data from that line's block
		- long sets are slow! - typicaly 2 or 3 lines would be common
	2. Direct-mapped cache - array of (tag,block) pairs
		- table with 1st column tags, 2nd column blocks
		- to read, check the block at the index given in the address
			- if found, read from block
			- else, load into that line
		- good spatial locality would make the tags adjacent (you read one full block, then the next full block)
		- this is faster (like a mux) instead of many = comparisons, typically big (1K-1M lines)
	3. Set-associative cache - hybrid - array of sets
		- indices each link to a set, each set with multiple elements
		- we search through the tags of this set
	- look at examples in book, know how to tell if we have hit or miss

### writing
- assume write-back, write-allocate cache
1. load block into cache (if not already) - write-allocate cache
2. change it in cache
3. two optionsm
	1. write back - wait until remove the line to update RAM
		- line needs to have tag, block, and dirty bit
		- dirty = wrote but did not update RAM
	2. write through - update RAM now
- no-write-allocate bypasses cache and writes straight to memory if block not already in cache (typically goes with write-through cache)
- valid bits - whether or not the line contains meaningful information
- kinds of misses
	1. cold miss - never looked at that line before
		- valid bit is 0 or we've only loaded other lines into the cache
		- associated with tasks that need a lot of memory only once, not much you can do
	2. capacity miss - we have n lines in the cache and have read ≥ n other lines since we last read this one
		- typically associated with a fully-associative cache
		- code has bad temporal locality
	3. conflict miss - recently read line with same index but different tag
		- typically asssociated with a direct-mapped cache
		- characteristic of the cache more than the code

### cache anatomy
- i-cache - holds instructions
	- typically read only
- d-cache - holds program data
- unified cache - holds both
- associativity - number of cache lines per set

# optimization
- only need to worry about locality for accessing memory
	- things that are in registers / immediates don't matter
- compilers are often cautious
- some things can't be optimized because you could pass in the same pointer twice as an argument
- loop unrolling - often dependencies accross iterations
```java
for(int i=0;i<n;i++){ //this line is bookkeeping overhead - want to reduce this
	a[i]+=1; //has temporal locality because you add and then store back to same address, obviously spatial locality
}
// unrolled
for(int i=0;i<n-2;i+=3){ //reduced number of comparisons, can also get benefits from vectorization (complicated)
	a[i]+=1;
	a[i+1]+=1;
	a[i+2]+=1;
}
if(n%3>=1) a[n-1]+=1;
if(n%2>=2) a[n-2]+=1;
```

- n%4 = n&3
- n%8 = n&7s
```java //less error prone
	for(int i=0;i<n;i+=1){ //this can be slower, data dependency between lines might not allow full parallelism (more stalling)
		a[i]+=1;
		i+=1;
		a[i]+=1;
		i+=1;
		a[i]+=1;
		i+=1;
	}
```
- loop order
- the order of loops can make this way faster
```java
for(i...)
	for(j...)
		a[i][j]+=1; //two memory accesses
```
- flatten arrays //this is faster, especially if width is a factor of 2, end of one row and beginning of next row are spatially local
	- row0 then row1 then row2 ....
	- float[height*width], access with array[row*width+column]
- problem - loop can't be picked
```java
	for(i..)
		for(j...)
			a[i][j]=a[j][i]; 
```
- solution - blocking
	- pick two chunks 
	- one we read by rows and is happy - only needs one cache line
	- one we read by columns - needs blocksize cache lines (by the time we get to the second column, we want all rows to be present in cache)
```java
	int bs=8;
	for (bx=0;bx<N;bx+=bs) // for each block
		for(by=0;by<N;by+=bs)
			for(x=bx;x<bx+bs;x+=1) // for each element of block
				for(y=by;y<by+bs;y+=1)
					swap(a[x][y],a[y][x]); // do stuff
```
- conditions for blocking
	1. the whole thing doesn't fit in cache
	2. there is no spatially local loop order
	- block size must be able to fit in cach
- reassociation optimization - compiler will do this for integers, but not floats (because of round-off errors)
	- a+b+c+d+e -> this is slower (addition is sequential by default, the sum of the first two is then added to the third number)
	- ((a+b)+(c+d))+e -> this can do things in parallel, we have multiple adders (we can see logarithmic performace if the chip can be fully parallell)
- using methods can increase your instruction cache hit rate


# exceptions
- processor is connected to I/O Bridge which is connected to Memory Bus and I/O Bus
	- these things are called the mother board
	- we don't want to wait for I/O Bus
		1. Polling - CPU periodically checks if ready
		2. Interrupts - CPU asks device to tell the CPU when the device is ready
			- this is what is usually done
- CPU has an interrupt pin (interrupt sent by I/O Bridge)
- steps for an interrupt
	1. pause my work - save where I can get back to it
	2. decide what to do next
	3. do the right thing
	4. resume my suspended work
- jump table - array of code addresses
	- each address points to handler code
	- CPU must pause, get index from bus, jump to exception_table[index]
	- need the exception table, exception code in memory, need register that tells where the exception table is
	- the user's code should not be able to use exception memory
- memory
	- mode register (1-bit): 2 modes
		- kernel mode (operating system) - allows all instructions
		- user mode - most code, blocks some instructions, blocks some memory
			- cant set mode register
			- can't talk to the I/O bus
	- largest addresses are kernel only
		- some of this holds exception table, exception handlers
	- between is user thhings
	- smallest addresses are unused - they are null (people often try to dereference them - we want this to throw an error)
- exceptions
	1. interrupts, 	index: bus, 			who causes it: I/O Bridge
		- i1,i2,i3,i4 -> interrupt during i3 instruction
		- let i3 finish (maybe)
		- handle interrupt
		- resume i4 (or rest of i3)
			 trap, 		%al (user)				int assembly instruction (user code)
		- trap is between instructions, simple
			 fault,		based on what failed	failing user-mode instruction
		- fault during i3
		- suspend i3
		- handle fault
		- rerun i3 (assuming we corrected fault - ex. allocating new memory) otherwise abort
	- abort - reaction to an exception (usually to a fault) - quits instead of resuming
- suspending
	
	- save PC, program register, condition codes (put them in a struct in kernel memory)
- on an exception
	1. switch to kernel mode
	2. suspend program
	3. jump to exception handler 
	4. execute exception handler
	5. resume in user mode

# processes
- (user) read file -> (kernel) send request to disk, wait, clean up -> (user) resume
	
	- this has lots of waiting so we run another program while we wait (see pic)
- process - code with an address space
	- CPU has a register that maps user addresses to physical addresses (memory pointers to each process)
	- general we don't call the kernel a process
	- also had pc, prog. registers, cc, etc.
	- each process has a pointer to the kernel memory
	- also has more (will learn in OS)...
- context switch - changing from one process to another
	- generally each core of a computer is running one process
	1. freeze one process
	2. let OS do some bookkeeping
	3. resume another process
	- takes time because of bookkeeping and cache misses on the resume
	- you can time context switches 
		while(true) 
			getCurrentTime()
				if(increased a lot) contextSwitches++

# threads
- threads are like processes that user code manages, not the kernel
	- within one address space, I have 2 stacks
	- save/restores registers and stack
	- hardware usually has some thread support
		- save/restore instructions
		- a way to run concurrent threads in parallel
			python threads don't run in parallel	

# system calls
- how user code asks the kernel to do stuff
	- exception table - there are 30ish, some free spots for OS to use
- system call - Linux uses exception 128 for almost all user -> kernel requests
	- uses rax to decide what you are asking //used in a jump table inside the 128 exception handler
	- most return 0 on success
	- non-zero on error where the # is errno
- you can write assembly in C code

# software exceptions
- you can throw an exception and then you want to return to something several method calls before you
- nonlocal jump
	- change PC
	- and registers (all of them)
- try{} - freezes what we need for catch
- catch{} - what is frozen
- throw{} - resume
- hardware exception can freeze state

# signals, setjmps
- exceptions - caused by hardware (mostly), handled by kernel
- signal - caused by kernel, handled by user code (or kernel)
	- mimics exception (usually a fault)
	- user-defined signal handler know to the OS
	- various signals (identified by number)
		- implemented with a jump table
	- we can mask (ignore) some signals
	- turns hardware fault into software exception (ex. divide by 0, access memory that doesn't exist), this way the user can handle it
	- SIGINT (ctrl-c) - usually cancels, can be blocked
		- ctrl-c -> interrupt -> handler -> terminal (user code) -> trap (SIGINT is action) -> handler -> sends signal
	- SIGTER - cancels, can't be blocked
	- SIGSEG - seg fault
- setjmp/longjmp - caused by user code, handled by user code
	- functions in standard C library in setjmp.h
	- jumps somewhere where pointer is something that stores current state
	- setjmp - succeeds first time (returns 0)
	- longjmp - never returns - calls setjmp with a different return value
	- you usually use if(setjmp) else {handle error} - basically try-catch

# virtual memory
- 2 address spaces 
	1. virtual address space (addressable space)
		- used by code
		- fixed by ISA designer 
	2. memory management unit (MMU) - takes in a virtual address and spits out physical address
		- page fault - MMU says this virtual address does not have a physical address
			- when there's a page fault, go to exception handler in kernel
			- usually we go to disk
	3. physical address space (cannot be discovered by code)
		- used by memory chips
		- constrained by size of RAM
- assume all virtual addresses have a physical address in RAM (this is not true, will come back to this)
	- each process has code, globals, heap, shared functions, stack
	- lots of unused at bottom, top because few programs use 2^64 bytes
	- RAM - we'll say this includes all caches
	- virtual memory is usually mostly empty
		
		- allocated in a few blocks / regions
	- MMU
		1. bad idea 1: could be a mapping from every virtual address to every physical address, but this wastes a lot
		- instead, we split memory into pages (page is continuous block of addresses ~ usually 4k)
			- bigger = fewer things to map, more likely to include unused addresses
			- address = low-order bits: page offset, high-order bits: page number
				- page offset takes log_2(page_size)
		2. bad idea 2: page table - map from virtual page number -> physical page number
			- we put the map in RAM, we have a register (called the PTBR) that tells us where it is
			- we change the PTBR for each process
			- CPU sends MMU a virtual address
			- MMU splits it into a virtual page number and page offset
			- takes 2 separate accesses to memory
				1. uses register to read out page number from page table
					- page table - array of physical page numbers, 2^numbits(virtual page numbers)
						- page table actually stores page table entries (PTEs)
						- PTE = PPN, read-only?, code or data?, user allowed to see it? 
						- MMU will check this and fault on error
				2. then it sends page number and page offset and gets back data
					- lookup address PTBR + VPN*numbytes(PPN)
			- consider 32-bit VA, 16k page (too large)
				- page offset is 14 bits
				- 2^18 PTEs = 256k PTEs
				- each PTE could be 4 bytes so the page table takes about 1 Megabyte
			- 64-bit VA, 4k page
				- 2^52 PTE -> the page table is too big to store
		3. good idea: multi-level page table
			- virtual address: page offset, multiple virtual page numbers $VPN_0,VPN_1,VPN_2,VPN_3$ (could have different number of	these)
			1. start by reading highest VPN: PTBR[VPN_3] -> PTE_3
			2. read PPN[VPN_2] -> PTE_2
			3. read PPN_2[VPN_1] -> PTE_1
			4. read PPN_1[VPN_0] -> PTE_0
			5. read PPN_0[VPN] -> PTE_ans
			- check at each level if valid, if unallocated/kernel memory/not usable then fault and stop looking
			- highest level VP_n is highest bits of address, likely that it is unused
				- therefore we don't have to check the other addresses
				- they don't exist so we save space, only create page tables when we need them - OS does this
			- look at these in the textbook
			- virtual memory ends up looking like a tree
				- top table points to several tables which each point to more tables
- TLB - maps from virtual page numbers to physical page numbers
- TLB vs L1, L2, etc:
- Similarities
	
	- They are all caches- i.e., they have an index and a tag and a valid bit (and sets)
- Differences
	- TLB has a 0-bit BO (i.e., 1 entry per block; lg(1) = 0)
	- TLB is not writable (hence not write-back or write-through, no dirty bit)
	- TLB entries are PPN, L* entries are bytes
	- TLB does VPN â†’ PPN; the L* do PA â†’ data

# overview
- CPU -> creates virtual address
- virtual address: 36 bits (VPN), 12 bits (PO) //other bits are disregarded
- VPN broken into 32 bits (Tag), 4 bits (set index)
	- set index tells us which set in the Translation Lookaside Buffer to look at
		- there are 2^4 sets in the TLB
		- currently there are 4 entries per set ~ this could be different
			- each entry has a valid bit
			- a tag - same length as VP Tag
			- value - normally called block - but here only contains one Physical page number - PPN = 40 bits ~ this could be different
	- when you go into kernel mode, you reload the TLB
		
# segments
- memory block
	- kernel at top
	- stack (grows down)
	- shared code
	- heap (grows up)
	- empty at bottom



base: 0x60

read: 0xb6
val at 0x6b -> 0x3d
val at 0xd6 -> ans

read: 0xa4
val at 0x6a -> 0x53 
val at 0x34 -> ans

0xb3a6

read: 0xb3
val at 0x6b -> 0x3d
val at 0xd3 -> 0x0f
val at 0xfa -> 0x6b
val at 0xb6


[TOC]

# quiz rvw
- commands
- floats
- labs
- In method main you declare an int variable named x. The compiler might place that variable in a register, or it could be in which region of memory? - Stack
- round to even is default
- Which Y86-64 command moves the program counter to a runtime-computed address? - ret
- [] mux defaults to 0
- caller-save register - caller must save them to preserve them
- callee-saved registers - callee must save them to edit them
- in the sequential y86 architecture valA<-eax
- valM is read out of memory - used in ret, mrmovl
- labels are turned into addresses when we assemble files
- accessing memory is slow
- most negative binary number: 100000
- floats can represent less numbers than unsigned ints
	- 0s and -0s are same
	- NaN doesn't count
- push/pop - sub/add to %rsp, put value into (%rsp)
- opl is 32-bit, opq is 64-bit
- fetch determines what the next PC will be
- fetch reads rA,rB,icode,ifun - decode reads values from these

# labs
### strlen
```java
unsigned int strlen( const char * s ){
	unsigned int i = 0; 
	while(s[i]) 
		i++; 
	return i;
}
```
### strsep
```java
char *strsep( char **stringp, char delim ){
	char *ans = *stringp;
	if (*stringp == 0) 
		return 0;
	while (**stringp != delim && **stringp != 0) /* don't need this 0 check, 0 is same as '\0' */
		*stringp += 1;
	if (**stringp == delim){ 
		**stringp = 0;
		*stringp += 1; 
	}
	else 
		*stringp = 0; 
	return ans;
}
```
###lists
- always test after malloc
- singly-linked list: node* { TYPE payload, struct node *next }
	- length: while(list) list = (*list).next
	- allocate: malloc(sizeof(node)*length)
				head[i].next=(i >= length) ? 0 : (head+i+1) 
	- access: (*list).payload or list[i].payload (for accessing)
- array: TYPE*
	- length: while(list[i] != sentinel)
	- allocate: malloc(sizeof(TYPE) * (length+1));
	- access: list[i]
- range: { unsigned int length, TYPE *ptr }
	- length: list.length
	- allocate: list.ptr = malloc(sizeof(TYPE) * length);
				ans.length = length;
	- access: list.ptr[i]
	
### bit puzzles
```java
// leastBitPos - return a mask that marks the position of the least significant 1 bit
int leastBitPos(int x) {
    return x & (~x+1);
}
int bitMask(int highbit, int lowbit) {
    int zeros = ~1 << highbit; /* 1100 0000 */
    int ones = ~0 << lowbit;   /* 1111 1000 */
    return ~zeros & ones;      /* 0011 1000 */
}
/* satAdd - adds two numbers but when positive overflow occurs, returns maximum possible value, and when negative overflow occurs, it returns minimum positive value. */
// soln - overflow when operands have same sign and sum and operands have different sign
int satAdd(int x, int y) {
	int x_is_neg = x >> 31;
	int y_is_neg = y >> 31;
	int sum = x + y;
	int same_sign = (x_is_neg & y_is_neg  |  ~x_is_neg & ~y_is_neg);
	int overflow = same_sign & (x_is_neg ^ (sum >> 31));
	int pos_overflow = overflow & ~x_is_neg;
	int neg = 0x1 << 31;
	int ans = ~overflow&sum | overflow & (pos_overflow&~neg | ~pos_overflow&neg);
	return ans;
}
```

# reading
### ch 1 (1.7, 1.9)
- files are stored as bytes, most in ascii
- all files are either text files or binary files
- i/o devices are connected to the bus by a controller or adapter
- processor holds PC, main memory holds program
- os-layer between hardware and applications - protects hardware and unites different types of hardware
- concurrent - instructions of one process are interleaved with another
    - does a context switch to switch between processes
    - concurrency - general concept of multiple simultaneous activities
    - parallelism - use of concurrency to make a system faster
- virtual memory-abstraction that provides each process with illusion of full main memory
	- memory - code-data-heap-shared libraries-stack
- threads allow us to have multiple control flows at the same time - switching
- multicore processor: either has multicore or is hyperthreaded (one CPU, repeated parts)
- processors can do several instructions per clock cycle
- Single-Instruction, Multiple-Data (SIMD) - ex. add four floats

### ch 2 (2.1, 2.4.2, 2.4.4)
- floating points (float, double)
	- sign bit (1)
	- exponent-field (8, 11)
		- bias = 2^(k-1)-1 ex. 127
		- normalized
			exponent = exp-Bias, mantissa = 1.mantissa
		- denormalized: exp - all 0s
			- exponent = 1-Bias, mantissa without 1 - 0 and very small values
		- exp: all 1s
			- infinity (if mantissa 0) 
			- NaN otherwise
	- mantissa 
- rounding
	1. round-to-even - if halfway go to closest even number - avoides statistical bias
	2. round-toward-zero
	3. round-down
	4. round-up
- leading 0 specifies octal
- leading 0x specifies hex
- leading 0b specifies binary

### ch 3 (3.6, 3.7)
- computers execute machine code
- intel processors are all back-compatible
- ISA - instruction set architecture
- control - condition codes are set after every instruction (1-bit registers)
	1. Zero Flag - recent operation yielded 0
	2. Carry Flag - yielded carry
	3. Sign Flag - yielded negative
	4. Overflow Flag - had overflow (pos or neg)
- guarded do can check if a loop is infinite
- instruction src, destination
- parentheses dereference a point
- there is a different add command for 16-bit operands than for 64-bit operands
- all instructions change the program counter
- call instruction only changes the stack pointer

### 4.1,4.2
- eight registers
	- esp is stack pointer
- CC and PC
- 4-byte values are little-endian
- status code State
	- 1 AOK
	- 2 HLT
	- 3 ADR - seg fault
	- 4 INS - invalid instruction code
- lines starting with "." are assembler directives
- assembly code is assembled resulting in just addresses and instruction codes
- pushl %esp - this doesn't change esp
- pop %esp - pops the value in esp
- high voltage = 1
- digital system components
	1. logic
	2. memory elements
	3. clock signals
- mux - picks a value and lets it through
- int Out = [
		s: A;
		1: B; 
    ];
	- B is the default
- combinatorial circuit - many bits as input simultaneously
- ALU - three inputs, A, B, func
- clocked registers store individual bits or words
- RAM stores several words and uses address to retrieve them
	- stored in register file

### 4.3.1-4
- SEQ - sequential processor
- stages
	- fetch
		- read icode,ifun <- byte 1
		- maybe read rA, rB <- byte 2
		- maybe read valC <- 8 bytes
	- decode
		- read operands usually from rA, rB - sometimes from %esp
		- call these valA, valB
	- execute
		- adds something, called valE
		- for jmp tests condition codes
	- memory
		- reads something from memory called valM or writes to memory
	- write back
		- writes up to two results to regfile
	- PC update
- popl reads two copies so that it can increment before updating the stack pointer
	 components: combinational logic, clocked registers (the program counter and condition code register), and random-access memories		
	- reading from RAM is fast
	- only have to consider PC, CC, writing to data memory, regfile
- processor never needs to read back the state updated by an instruction in order to complete the processing of this instruction.
- based on icode, we can compute three 1-bit signals :
	1. instr_valid: Does this byte correspond to a legal Y86 instruction? This signal is used to detect an illegal instruction.
	2. need_regids: Does this instruction include a register specifier byte? 
	3. need_valC: Does this instruction include a constant word?
	
### 4.4 pipelining
- the task to be performed is divided into a series of discrete stages
- increases the throughput - # customers served per unit time
- might increase latency - time required to service an individual customer.
- when pipelining, have to add time for each stage to write to register
- time is limited by slowest stage
- more stages has diminishing returns for throughput because there is constant time for saving into registers
	- latency increases with stages
	- throughput approaches 1/(register time)
- we need to deal with dependencies between the stages

### 4.5.3, 4.5.8
- several copies of values such as valC, srcA
- registers dD, eD, mM, wW - lowercase letter is input, uppercase is output
- we try to keep all the info of one instruction within a stage
- merge signals for valP in call and valP in jmp as valA
- load/use hazard - (try using before loaded) one instruction reads a value from memory while the next instruction needs this value as a source operand
- we can stop this by stalling and forwarding (the use of a stall here is called a load interlock)

### 5 - optimization
- eliminate unnecessary calls, tests, memory references
- instruction-level parallelism
- profilers - tools that measure the performance of different parts of the program
- critical paths - chains of data dependencies that form during repeated executions of a loop
- compilers can only apply safe operations
- watch out for memory aliasing - two pointers desginating same memory location
- functions can have side effects - calling them multiple times can have different results
- small boost from replacing function call by body of function (although this can be optimized in compiler sometimes)
- measure performance with CPE - cycles per element
- reduce procedure calls (ex. length in for loop check)
- loop unrolling - increase number of elements computed on each iteration
- enhance parallelism 
	- multiple accumulators
- limiting factors
	- register spilling - when we run out of registers, values stored on stack
	- branch prediction - has misprediction penalties, but these are uncommon
		- trinary operator could make things faster
- understand memory performance
- using macros lets compiler optimizem more, lessens bookkeeping

### 6.1.1, 6.2, 6.3
- SRAM is bistable as long as power is on - will fall into one of 2 positions
- DRAM loses its value ~10-100 ms
	- memory controller sends row,col (i,j) to DRAM and DRAM sends back contents
	- matrix organization reduces number of inputs, but slower because must use 2 steps to load row then column
- memory modules
- enhanced DRAMS
- nonvolatile memory 
	- ROM - read-only memories - firmwared
- accessing main memory
	- buses - collection of parallel wires that carry address, data, control
- accessing main memory
- locality
	- locality of references to program data
		- visiting things sequentially (like looping through array) - stride-1 reference pattern or sequential reference pattern
	- locality of instruction fetches
		- like in a loop, the same instructions are repeated
- memory hierarchy
	- block-sizes for caching can differ between different levels
	- when accessing memory from cache, we either get cache hit or cache miss
	- if we miss we replace or evict a block
		- can use random replacement or least-recently used
	- cold cache - cold misses / compulsory misses - when cache is empty
		- need a placement policy for level k+1 -> k (could be something like put block i into i mod 4)
	- conflict miss - miss because placement policy gets rid of block you need - ex. block 0 then 8 then 0 with above placement policy
		 capacity misses - the cache just can't hold enough	

### 6.4, 6.5 - cache memories & writing cache-friendly code
- Miss rate. The fraction of memory references during the execution of a program, or a part of a program, that miss. It is computed as #misses/#references.
- Hit rate. The fraction of memory references that hit. It is computed as 1 − miss rate.
- Hit time. The time to deliver a word in the cache to the CPU, including the time for set selection, line identification, and word selection. Hit time is on the order of several clock cycles for L1 caches.
- Miss penalty. Any additional time required because of a miss. The penalty for L1 misses served from L2 is on the order of 10 cycles; from L3, 40 cycles; and from main memory, 100 cycles.
- Traditionally, high-performance systems that pushed the clock rates would opt for smaller associativity for L1 caches (where the miss penalty is only a few cycles) and a higher degree of associativity for the lower levels, where the miss penalty is higher
- In general, caches further down the hierarchy are more likely to use write-back than write-through

### 8.1 Exceptions
- exceptions - partly hardware, partly OS
- when an event occurs, indirect procedure call (the exception) through a jump table called exception table to OS subroutine (exception handler).
- three possibilities
	- returns to I_curr
	- returns to I_next
	- program aborts
- exception table - entry k contains address for handler code for exception k
- processor pushes address, some additional state
- four classes
	1. interrupts (the faulting instruction)
		- signal from I/O device, Async, return next instruction
	2. traps
		- intentional exception (interface for making system calls), Sync, return next
	3. faults
		- potentially recoverable error (ex. page fault exception), Sync, might return curr
	4. aborts
		- nonrecoverable error, Sync, never returns
- examples
	- general protection fault - seg fault
	- machine check - fatal hardware error
	
### 8.2 Processes
- process - instance of program in execution
	- every program runs in the context of some process (context has code, data stack, pc, etc.)
1. logic control flow - like we have exclusive use of processor
	- processes execute partially and then are preempted (temporarily suspended)
	- concurrency/multitasking/time slicing - if things trade off
	- parallel - concurrent and on separate things
	- kernel uses context switches 
2. private address space - like we have exclusive use of memory
	- each process has stack, shared libraries, heap, executable

### 8.3 System Call Error Handling
- system level calls return -1, set the global integer variable errno
	- this should be checked for
	
### 9-9.5 Virtual Memory
- address translation - converts virtual to physical address
	- translated by the MMU
- VM partitions virtual memory into fixed-size blocks called virtual pages partitioned into three sets
	1. unallocated
	2. cached
	3. uncached
- virtual pages tend to be large because cache misses are large
- DRAM will be fully associative, write-back
- each process has a page table - maps virtual pages to physical pages
	- managed by OS
	- has PTEs
	- PTE - valid bit, n-bit address field
		- valid bit - whether its currently cached in DRAm
		- if yes, address is the start of corresponding physical page
		- if valid bit not set && null address - has not been allocated
		- if valid bit not set && real address - points to start of virtual page on disk
	- PTE - 3 permission bits
		- SUP - does it need to be in kernel (supervisor) mode?
		- READ - read access
		- WRITE - write access
- page fault - DRAM cache miss
	- read valid bit is not set - triggers handler in kernel
- demand paging - waiting until a miss occurs to swap in a page
- malloc creates room on disk
- thrashing - not good locality - pages are swapped in and out continuoously
- virtual address space is typically larger
	- multiple virtual pages can be mapped to the same shared physical page (ex. everything points to printf)
- VM simplifies many things
	- linking
		- each process follows same basic format for its memory image
	- loading
		- loading executables / shared object files
	- sharing
		- easier to communicate with OS
	- memory allocation
		- physical pages don't have to be contiguous
- memory protection
	- private memories are easily isolated

### 9.6 Address Translation
- low order 4 bits serve
2,3 - fault
8c: 1000 1100
b6