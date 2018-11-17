# GMiner

## Introduction
<b>GMiner</b> is an algorithm for finding frequent itemsets using computing power of GPUs.</br>

<b>GMiner</b> has the following characteristics:
* Scalable with respect to the size of datasets, which do not fit into GPU device memory
* Scalable with respect to the number of GPUs by evenly distributing amount of work to each GPU
* Fast and robust comparing to the state-of-the art methods (especially, handling large-scale datasets)



## Research Paper
[GMiner: A fast GPU-based frequent itemset mining method for large-scale data](https://www.sciencedirect.com/science/article/pii/S0020025518300690)

## Implementation
Implemented C++ code interacting with GPU servers and computing frequent itemsets according to itemsets.

## Installation

* This version requires (1) g++ v.4.8 or greater be installed in the system and set in PATH, (2) Boost C++ Libraries be installed and set in PATH, and (3) CUDA 8 be installed and set in PATH (we are not testing GMiner on other configurations)
* For compilation, type *./build.sh* and then get “*GMiner*” in the same directory
* For cleaning executable files, type *./clean.sh*

## Input File Format

In an input dataset, each transaction is stored in a single line (row). In the transaction, items are non-negative integers and separated by a space.

## Output File Format
The output includes a number of lines. Each line includes a single frequent itemset and its support ratio in a range of [0,1].

## How to run
### Command
./GMiner -i <input_path> -o <output_path> -s <min_sup> -w <is_write_output>

### Parameters
* *input_path* (-i): path of the input file (in default “webdocs” in the directory).
* *output_path* (-o): path of the output file (in default “out”)
* *min_sup* (-s) : minimum support (in range [0.0,1.0], in default 0.1)
* *is_write_output* (-w): whether it writes outputs or not (0: no, 1: yes, in default 0)
