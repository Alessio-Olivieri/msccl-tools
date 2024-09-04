# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import argparse, math

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

def allreduce(size, instances, protocol):
    topology = fully_connected(size)
    logical_chunk = size
    collective = AllReduce(size, logical_chunk, True)
    with MSCCLProgram("allreduce_swing_latency_optimal", topology, collective, instances, protocol):
        def pi(r, s, n):
            # Use a similar peer-determination logic as in the Python version
            if r % 2 == 0:
                return (r + (1 - math.pow(-2, s + 1)) / 3) % n
            else:
                return (r - (1 - math.pow(-2, s + 1)) / 3) % n
            
        for step in range(int(math.log2(size))):
            for r in range(size):
                
                peer = pi(r, step, step)
                c = chunk(r, Buffer.input, 0)  # Access the full buffer
                c_peer = c.copy(peer, Buffer.output, 0)  # Send to the peer
                c_out = c_peer.reduce(chunk(peer, Buffer.input, 0))  # Reduce with local buffer


        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()
allreduce(args.num_gpus, args.instances, args.protocol)
