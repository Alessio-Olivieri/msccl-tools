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

def allreduce_swing(size, instances):
    # Logical topology
    topology = fully_connected(size)
    collective = AllReduce(size, size, inplace=True)

    with MSCCLProgram("allreduce_swing", topology, collective, instances):
        
        size_log2 = 2 ** (int(math.log2(size))) #The closest power of 2 (smaller) to size

        #The extra ranks send to the log2 ranks
        peer = size_log2-1
        for r in range(size_log2, size):
            logger.debug(f"sending input buffer of {r} into scratch of {peer}")
            peer_scratch = chunk(r, Buffer.input, 0, size=size).copy(peer, 'scratch', 0, sendtb=r, recvtb=peer)
            chunk(peer, Buffer.input, 0, size=size).reduce(peer_scratch)
            peer -= 1

        # run swing on log2 ranks
        for step in range(int(math.log2(size_log2))):
            logger.debug(f"Starting step {step}")
            # Each rank sends it's buffer to it's peer and the peer performs a reduce
            for r in range(size_log2):
                peer = pi(r, step, size_log2)
                logger.debug(f"copying input buffer of {r} into scratch of {peer}")
                chunk(r, Buffer.input, index=0, size=size).copy(peer, 'scratch', 0, recvtb=peer, sendtb=r)
            for r in range(size_log2):
                logger.debug(f"reducing scratch buffer of {r} into the input buffer of {r}")
                chunk(r, Buffer.input, 0, size=size).reduce(chunk(r, 'scratch', 0, size=size))

        #The extra ranks recive from the log2 ranks
        peer = size_log2-1
        for r in range(size_log2, size):
            chunk(peer, Buffer.input, 0, size=size).copy(r, Buffer.input, 0, recvtb=r, sendtb=peer)
            

        
             

        # log2_allreduce_swing(size_log2, size)# size_log2 nodes run swing

        # recv_buffers(size_log2, size)# size - size_log2 ranks recive from size ranks the whole buffer

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')

args = parser.parse_args()

allreduce_swing(args.num_gpus, args.instances)