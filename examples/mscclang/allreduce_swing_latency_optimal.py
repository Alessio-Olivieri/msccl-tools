# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import argparse, math, logging
import logging

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce


# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def allreduce(size, instances, protocol):
    logging.info(f"Starting allreduce with size={size}, instances={instances}, protocol={protocol}")
    topology = fully_connected(size)
    logical_chunk = size
    collective = AllReduce(size, logical_chunk, True)
    
    with MSCCLProgram("allreduce_swing_latency_optimal", topology, collective, instances, protocol):
        def pi(r, s, n):
            p = (1 - math.pow(-2, s+1))/3
            peer = (r + p) % n if r % 2 == 0 else (r - p) % n
            logging.debug(f"pi calculation for rank {r}, step {s}, size {n}: peer={peer}")
            return int(peer)
        
        for step in range(int(math.log2(size))):
            logging.debug(f"Starting step {step}")
            for r in range(size):
                peer = pi(r, step, size)
                logging.debug(f"Rank {r} sending to peer {peer} at step {step}")
                c = chunk(r, Buffer.input, 0)
                c.copy(peer, Buffer.output, 0)

            for r in range(size):
                received_chunk = chunk(r, Buffer.output, 0)
                input_chunk = chunk(r, Buffer.input, 0)
                input_chunk.reduce(received_chunk)
                logging.debug(f"Rank {r} reduced chunk at step {step}")

        logging.info("Allreduce completed, generating XML")
        XML()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()
allreduce(args.num_gpus, args.instances, args.protocol)
