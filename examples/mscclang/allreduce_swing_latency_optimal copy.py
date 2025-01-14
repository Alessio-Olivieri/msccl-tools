# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import argparse, math
import logging

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def pi(r, s, n):
    p = (1 - math.pow(-2, s+1))/3
    peer = (r + p) % n if r % 2 == 0 else (r - p) % n
    return int(peer)

def allreduce(size, instances, protocol):
    logger = logging.getLogger(__name__)

# Configure logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info(f"Starting allreduce with size={size}, instances={instances}, protocol={protocol}")
    topology = fully_connected(size)
    logical_chunk = size
    collective = AllReduce(size, logical_chunk, True)
    
    with MSCCLProgram("allreduce_swing_latency_optimal", topology, collective, instances, protocol):
        size_log2 = 2 ** (int(math.log2(size))) #The closest power of 2 (smaller) to size
        size_extra = size-size_log2

        aliases = [] #The actual ranks that execute swing
        siblings = [] #Log2 ranks and extra ranks pairs

        r = 0
        while r<size:
            if size_extra>0:
                aliases.append(r)
                siblings.append((r,r+1))
                r+=2
                size_extra-=1
            else:
                aliases.append(r)
                r+=1

        for r, extra_rank in siblings:
            chunk(r, Buffer.input, 0, size=size).reduce(chunk(extra_rank, Buffer.input, 0, size=size))

        # run swing on log2 ranks
        for step in range(int(math.log2(size_log2))):
            logger.debug(f"Starting step {step}")
            # Each rank saves the current input buffers in a scratch buffer
            for r in range(size_log2):
                chunk(aliases[r], Buffer.input, index=0, size=size).copy(aliases[r], "scratch", 0)
            # Each rank sends it's buffer to it's peer and the peer performs a reduce
            for r in range(size_log2):
                peer = pi(r, step, size_log2)
                logger.debug(f"copying input buffer of {aliases[r]} into scratch of {aliases[peer]}")
                chunk(aliases[peer], Buffer.input, index=0, size=size).reduce(chunk(aliases[r], "scratch", 0, size=size))
        
        for r, extrar in siblings:
            chunk(r, Buffer.input, 0, size=size).copy(extrar, Buffer.input, 0)

        XML()
        Check()  
        Print_instruction_dag()
        # Crint()


parser = argparse.ArgumentParser()
# parser.add_argument('num_gpus', type=int, help ='number of gpus')
# parser.add_argument('instances', type=int, help='number of instances')
# parser.add_argument('--protocol', type=str, default='LL', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
# args = parser.parse_args()
# allreduce(args.num_gpus, args.instances, args.protocol)
allreduce(4,1, 'LL')
