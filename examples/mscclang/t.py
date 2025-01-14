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
    collective = AllReduce(size, logical_chunk, False)
    
    with MSCCLProgram("allreduce_swing_latency_optimal", topology, collective, instances, protocol):
        to_reduce = chunk(1,Buffer.input, 0, 4).copy(1, Buffer.output, 0)
        chunk(2, Buffer.input, 0, 4).reduce(to_reduce)
        XML()
        Print_instruction_dag()

parser = argparse.ArgumentParser()
# parser.add_argument('num_gpus', type=int, help ='number of gpus')
# parser.add_argument('instances', type=int, help='number of instances')
# parser.add_argument('--protocol', type=str, default='LL', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
# args = parser.parse_args()
# allreduce(args.num_gpus, args.instances, args.protocol)
allreduce(8,1, 'LL')
