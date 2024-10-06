## Table of Contents
1. [Introduction](#introduction)
2. [Buffers](#buffers)
   - [Chunks and Buffers](#chunks-and-buffers)
   - [Using Scratch Buffers for Flexible Temporary Storage](#using-scratch-buffers-for-flexible-temporary-storage)
   - [Creating and Assigning Buffer Slices](#creating-and-assigning-buffer-slices)
3. [Chunks](#chunks)
   - [Chunk Types](#chunk-types)
   - [References](#references)
   - [Operations](#operations)
     - [split()](#splitself-num)
     - [group()](#groupself-other)
     - [copy()](#copyself-dst-bufferindex-sendtb-recvtb-ch)
     - [reduce()](#reduceself-other_chunkref-sendtb-recvtb-ch)
     - [get_origin_index()](#get_origin_indexself-index)
     - [get_origin_rank()](#get_origin_rankself-index)
     - [get_dst_index()](#get_dst_indexself-index)
     - [get_dst_rank()](#get_dst_rankself-index)
     - [print_chunk_info()](#print_chunk_infoself-index)
   - [Examples](#examples)
     - [Transferring the Whole Buffer](#transferring-the-whole-buffer)
4. [Collectives](#collectives)
   - [Overview](#overview)
     - [Collective](#collective)
   - [Collective Types](#collective-types)
     - [AllToAll](#alltoall)
     - [AllGather](#allgather)
     - [AllReduce](#allreduce)
     - [ReduceScatter](#reducescatter)

---
# Introduction
MSCCLang is a high-level language for specifying collective communication algorithms in an intuitive chunk-oriented form. The language is available as a Python-integrated DSL.

# Buffers
MSCCLang is designed to expose GPU memory as named buffers that can be accessed and manipulated during the execution of collective operations. Each rank in the program has access to three distinct types of GPU memory buffers:

- **Input Buffer**: A buffer containing the input data for the collective operation.
- **Output Buffer**: An uninitialized buffer intended to store the output data after the operation completes.
- **Scratch Buffer**: An uninitialized buffer that serves as temporary storage during computations.

These buffers are divided into *chunks*.
The primary goal of an MSCCLang program is to ensure that the **output buffer** on each GPU contains the correct chunks once the collective operation is complete.

## Chunks and buffers
Chunks represent contiguous spans of data with a uniform size in the same buffer.
1. **Number of chunks**: The user specifies how many chunks each buffer is divided into.
2. **Chunk Size**: The size of each chunk is determined at runtime when actual data is available.
3. **Aliased Buffers**: Users can choose to alias the input and output buffers. This allows for *in-place* collective operations, where the input buffer is reused as the output buffer, minimizing memory usage and optimizing performance.

## Using Scratch Buffers for Flexible Temporary Storage
MSCCL allows to organize the scratch buffer into named slices. This gives users the possibility to treat each slice of the global scratch buffer as if it was an independent scratch buffer. 
Interanlly each slice has an offset that represents the starting point within the global scratch buffer of the rank it belongs to.

### Creating and Assigning Buffer Slices
To create a slice of the global scratch buffer named "new_scratch_buffer" of a rank do:
   ```python
   c1.copy(rank, "new_scratch_buffer", index)
   ```
   Such a buffer can then be treated normally as you would do with the classic "scratch" buffer, for example to address that buffer you would do:
  ```python
  c2 = chunk(rank, "new_scratch_buffer", index, size)
   ```


# Chunks
### Chunk Types
Chunks can take one of the following forms:

1. **Input Chunks**: These chunks are initialized at runtime with data from the input buffer. Each input chunk is uniquely identified by a `(rank, index)` pair, where `rank` refers to the GPU and `index` refers to the chunk within that rank's input buffer.

2. **Reduction Chunks**: These chunks are produced as a result of combining two or more input chunks using a point-wise reduction operation (e.g., addition). A reduction chunk is uniquely identified by the list of input chunks that were combined to form it.

3. **Uninitialized Chunks**: These chunks do not contain any initialized data. When the program starts, both the output and scratch buffers consist of uninitialized chunks. These chunks are later replaced with initialized data during the execution of the collective operation.

## References
A reference is a reference to a single or multiple chunks in a Buffer.
A reference is globally determined by 4 variables:
- **Rank** The rank to which the chunks referenced belong to
- **Buffer** The buffer to which the chunks referenced belong to
- **Index** The offset in the buffer of the first referenced chunk
- **Size** The number of chunks referenced

A reference is created with the **chunk** operation:
```python
# c1 is a reference to the whole input buffer of rank 0
c1 = chunk(0, Buffer.input, 0, size=chunk_factor)
```

### Operations
Every operation on chunks is usually done by acting upon references to those chunks, these operations are exposed as methods of the Ref class:

#### `split(self, num)`
- **Description**: Splits the current chunk into `num` smaller chunks, each having an equal portion of the original chunk's size.
- **Parameters**:
  - `num (int)`: The number of smaller chunks to divide the original chunk into. The size of the original chunk must be divisible by `num`.
- **Returns**: 
  - A list of `Ref` objects, each representing a part of the original chunk. Each new `Ref` will point to a portion of the original chunk, defined by its size and starting index.

#### `group(self, other)`
- **Description**: Combines (concatenates) two `Ref` objects into a single chunk. This operation is only allowed if both chunks belong to the same rank and buffer.
- **Parameters**:
  - `other (Ref)`: Another `Ref` object that will be concatenated with the current one.
- **Returns**: 
  - A new `Ref` object that represents the combined chunk, spanning the original two chunks.

#### `copy(self, dst, buffer=None, index=-1, sendtb=-1, recvtb=-1, ch=-1)`
- **Description**: Copies the current chunk to a destination rank (`dst`) and buffer. If the destination index or buffer is not specified, the method assumes the chunk is copied to the same location in the destination rank as in the source rank.
- **Parameters**:
  - `dst (int)`: The destination rank where the chunk will be copied.
  - `buffer`: The destination buffer.
  - `index (int)`: The index in the destination buffer where the chunk will be placed (optional, defaults to the same index as the source).
  - `sendtb (int)`: The rank sending the chunk.
  - `recvtb (int)`: The rank receiving the chunk.
  - `ch (int)`: Channel to use for the communication (optional).
- **Returns**: 
  - A `Ref` object pointing to the chunk at the destination after it has been copied.

#### `reduce(self, other_chunkref, sendtb=-1, recvtb=-1, ch=-1)`
- **Description**: Reduces the current chunk with another chunk (`other_chunkref`), combining their data into the current `Ref`.
- **Parameters**:
  - `other_chunkref (Ref)`: The other chunk to reduce with the current one.
  - `sendtb (int)`: The sending rank
  - `recvtb (int)`: the receiving rank
  - `ch (int)`: Channel to use for communication (optional).
- **Returns**: 
  - The updated `Ref` object, now representing the reduced chunk.

#### `get_origin_index(self, index=0)`
- **Description**: Retrieves the original index of the chunk within its rank
- **Parameters**:
  - `index (int)`: The offset within the chunk to retrieve the origin index (optional, default is 0).
- **Returns**: 
  - The original index of the chunk.

#### `get_origin_rank(self, index=0)`
- **Description**: Retrieves the original rank of the chunk
- **Parameters**:
  - `index (int)`: The offset within the chunk to retrieve the origin rank (optional, default is 0).
- **Returns**: 
  - The original rank of the chunk.

#### `get_dst_index(self, index=0)`
- **Description**: Retrieves the destination index where the chunk will be placed after operations such as copying or reducing.
- **Parameters**:
  - `index (int)`: The offset within the chunk to retrieve the destination index (optional, default is 0).
- **Returns**: 
  - The destination index of the chunk.

#### `get_dst_rank(self, index=0)`
- **Description**: Retrieves the destination rank of the chunk after operations such as copying or reducing.
- **Parameters**:
  - `index (int)`: The offset within the chunk to retrieve the destination rank (optional, default is 0).
- **Returns**: 
  - The destination rank of the chunk.

#### `print_chunk_info(self, index=0)`
- **Description**: Prints detailed information about the chunk at the specified index, including its origin rank, origin index, destination rank, and destination index.
- **Parameters**:
  - `index (int)`: The offset within the chunk to retrieve and print its information (optional, default is 0).

## Examples
### Transfering the whole buffer
We want to copy the input buffer of 0 into the input buffer of 1. We have to 'select' all the chunks in the buffer.input. To do so we use the chunk_factor which are the initial number of chunks in the input buffer
``` python
chunk(0, Buffer.input, index=0, size=chunk_factor).copy(1, Buffer.input, index=0, sendtb=0, recvtb=1)
```


# Collectives
Collectives handled by MSCCL must be created implementing the class Collective.
These collectives are designed to handle both in-place and out-of-place communication scenarios and support configurable chunk factors for fine-tuning communication granularity.
Each collective communication operation handles the initialization of communication buffers and validation of communication patterns across multiple ranks.
## Overview

### `Collective`
The base class for all collective communication operations. It provides common initialization parameters and helper methods.
```python
class Collective():
    def __init__(self, num_ranks, chunk_factor, inplace):
        self.num_ranks = num_ranks
        self.chunk_factor = chunk_factor
        self.inplace = inplace
        self.name = "custom"

    def init_buffers(self):
        pass

    def check(self, prog):
        pass
```

#### Attributes:
- `num_ranks`: Number of participating ranks in the collective operation.
- `chunk_factor`: The number of chunks in which the buffers are divided.
- `inplace`: Boolean flag to indicate whether the operation is in-place.
- `name`: The name of the collective operation.

#### Methods:
- `__init__(self, num_ranks, chunk_factor, inplace)`: Initializes the collective operation with the specified number of ranks, chunk factor, and whether it's in-place or out-of-place.
- `init_buffers(self)`: Abstract method to initialize the input and output buffers. Implementation is specific to the collective operation.
- `check(self, prog)`: Abstract method to verify if the buffers after communication match the expected results.
- `get_buffer_index(self, rank, buffer, index)`: Returns the buffer and index used during communication based on the rank and buffer type.

---

### `AllToAll`
A collective communication operation where every rank sends data to every other rank. Each rank divides its data into chunks, and these chunks are distributed across ranks.

#### Methods:
- `init_buffers(self)`: Initializes input and output buffers for all ranks. Each rank has its own input buffer where chunks are distributed among the output buffers of other ranks. Supports in-place and out-of-place configurations.
- `check(self, prog)`: Verifies the correctness of the output buffers after the all-to-all operation. Checks if the data has been correctly transferred between ranks.
  
---

### `AllGather`
A collective communication where all ranks contribute to a shared data buffer. Every rank gathers chunks from all other ranks.

#### Methods:
- `init_buffers(self)`: Initializes the input and output buffers for all ranks. If in-place, input buffers are aliases of the output buffers. Otherwise, they are separate.
- `check(self, prog)`: Verifies the output buffers by checking if all ranks correctly received the gathered chunks.
- `get_buffer_index(self, rank, buffer, index)`: For in-place all-gather operations, the input buffer points to the corresponding index in the output buffer.

---

### `AllReduce`
A collective operation where each rank reduces data from all other ranks and stores the result in every rank’s output buffer. Each chunk is reduced by applying some reduction operation (e.g., sum).

#### Methods:
- `init_buffers(self)`: Initializes the input and output buffers. If in-place, input buffers are aliases for output buffers. Otherwise, separate buffers are used.
- `check(self, prog)`: Verifies the output buffers after the all-reduce operation. Ensures that each chunk contains the correct reduced result from all ranks.
- `get_buffer_index(self, rank, buffer, index)`: Manages buffer indices based on whether the operation is in-place or out-of-place.

---

### `ReduceScatter`
This operation performs a reduction across all ranks and scatters the reduced results such that each rank receives part of the reduced data.

#### Methods:
- `init_buffers(self)`: Initializes buffers for reduce-scatter communication. Buffers are either in-place or out-of-place, depending on the operation.
- `check(self, prog)`: Verifies if the chunks received by each rank match the expected results after reduction.
- `get_buffer_index(self, rank, buffer, index)`: For in-place operations, the output buffer refers to a segment of the input buffer.

---

### Buffers and Chunks

Each collective class manages buffers that hold data chunks. These chunks are transferred and manipulated during the communication operations. `Chunk` and `ReduceChunk` objects represent the units of data communication between ranks.

### Example Usage
To initialize an `AllReduce` operation with 4 ranks, a chunk factor of 2, and in-place communication:
```python
collective = AllReduce(num_ranks=4, chunk_factor=2, inplace=True)
```
