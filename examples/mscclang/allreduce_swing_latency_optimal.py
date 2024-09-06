# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import argparse, math

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

def p(s):
    # This calculates p(s) = sum((-2)^i for i in range(s+1))
    return sum([(-2)**i for i in range(s+1)])

def pi(r, s, n):
    if r % 2 == 0:
        peer = (r + p(s)) % n
    else:
        peer = (r - p(s)) % n
    return int(peer)

def allreduce(size, instances, protocol):
    topology = fully_connected(size)
    logical_chunk = size
    collective = AllReduce(size, logical_chunk, True)
    with MSCCLProgram("allreduce_swing_latency_optimal", topology, collective, instances, protocol):
        for step in range(int(math.log2(size))):
            for r in range(size):
                # Each rank sends its buffer
                peer = pi(r, step, size)
                c = chunk(r, Buffer.input, 0)
                c.copy(peer, Buffer.output, 0)

            for r in range(size):
                # Each rank reduces its buffer with the received buffer
                received_chunk = chunk(r, Buffer.output, 0)
                input_chunk = chunk(r, Buffer.input, 0)
                received_chunk.reduce(input_chunk)


        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()
allreduce(args.num_gpus, args.instances, args.protocol)
