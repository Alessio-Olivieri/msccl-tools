import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.collectives import *
from msccl.language.collectives import AllReduce
import math
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def pi(r, s, n):
    p = (1 - math.pow(-2, s+1))/3
    peer = (r + p) % n if r % 2 == 0 else (r - p) % n
    return int(peer)

def log2_allreduce_swing(size, buffer_size):
    # run swing on log2 ranks
    for step in range(int(math.log2(size))):
        logger.debug(f"Starting step {step}")
        # Each rank sends the nth chunk to the nth rank into scratch space
        for r in range(size):
            peer = pi(r, step, size)
            logger.debug(f"copying buffer of {r} into scratch of {peer}")
            c = chunk(r, Buffer.input, index=0, size=buffer_size)
            c.copy(peer, 'scratch', 0, recvtb=peer, sendtb=r)

        for rank in range(size):
            logger.debug(f"reducing scratch buffer of {rank} into input buffer")
            c = chunk(rank, Buffer.input, 0, size=buffer_size)
            c.reduce(chunk(rank, 'scratch', 0, size=buffer_size))

def send_buffers(size, buffer_size):
    peer = size - 1
    for r in range(size, buffer_size):
            logger.debug(f"copying buffer of {r} into scratch of {peer}")
            c = chunk(r, Buffer.input, index=0, size=buffer_size)
            c.copy(peer, 'scratch', 0, recvtb=peer, sendtb=r)
            peer -= 1

    for rank in range(size):
        logger.debug(f"reducing scratch buffer of {rank} into input buffer")
        c = chunk(rank, Buffer.input, 0, size=buffer_size)
        c.reduce(chunk(rank, 'scratch', 0, size=buffer_size))

def recv_buffers(size, buffer_size):
    # The log2_size ranks send to the input buffer (because it's inplace) of the extra ranks
    peer = size - 1
    for r in range(size, buffer_size):
            logger.debug(f"copying buffer of {r} into input of {peer}")
            c = chunk(peer, Buffer.input, index=0, size=buffer_size)
            c.copy(r, Buffer.input, 0, recvtb=r, sendtb=peer)
            peer -= 1

def allreduce_swing(size, instances):
    # Logical topology
    topology = fully_connected(size)
    collective = AllReduce(size, size, inplace=True)

    with MSCCLProgram("allreduce_swing", topology, collective, instances):
        
        size_log2 = 2 ** (int(math.log2(size))) #The closest power of 2 (smaller) to size


        send_buffers(size_log2, size)# size - size_log2 ranks send to size ranks their buffers
        
        log2_allreduce_swing(size_log2, size)# size_log2 nodes run swing

        recv_buffers(size_log2, size)# size - size_log2 ranks recive from size ranks the whole buffer

        Check()
        XML()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')

args = parser.parse_args()

allreduce_swing(args.num_gpus, args.instances)