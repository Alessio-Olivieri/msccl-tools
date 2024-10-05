import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.collectives import *
from msccl.language.collectives import AllReduce
import math
import logging


# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def pi(r, s, n):
    p = (1 - math.pow(-2, s+1))/3
    peer = (r + p) % n if r % 2 == 0 else (r - p) % n
    return int(peer)

def get_rs_idxs(r, s, n):
    if s >= math.log2(n): 
        return []
    else:
        l =[]
        for step in range(s, int(math.log2(n))):
            peer = pi(r, step, n)
            l += [peer]
            l += get_rs_idxs(peer, step+1, n)
        return l

# This version sends non contiguous blocks of data all at once.
def allreduce_swing_optimized(size, instances):
    # Logical topology
    topology = fully_connected(size)
    collective = AllReduce(size, size, inplace=False)

    with MSCCLProgram("allreduce_swing", topology, collective, instances):

        for r in range(size):
            chunk(r, Buffer.input,index=0, size=size).copy(r, Buffer.output, index=0)

        # Reduce scatter
        for s in range(int(math.log2(size))):
            for r in range(size):
                peer = pi(r, s, size)
                to_send = get_rs_idxs(peer, s+1, size) + [peer] 

                # Step 1: Allocate scratch buffer size
                scratch_size = len(to_send)

                # Step 2: Gather non-contiguous chunks into scratch buffer of sending GPU
                for i, block_id in enumerate(to_send):
                    # logger.debug(f"[{r}] gathering output[{block_id}] input into scratch_send{r}[{i}]")
                    chunk(r, Buffer.output, block_id, size=1).copy(r, f'scratch_send{r}', index=i)

                # Step 3: Perform a single copy operation to a scratch buffer in destination GPU
                # logger.debug(f"[{r}] sending scratch_send to scratch_receive of {peer}")
                chunk(r, f'scratch_send{r}', 0, size=scratch_size).copy(peer, f'scratch_receive{peer}', index=0, sendtb=r, recvtb=peer)

            for r in range(size):   
                to_receive = get_rs_idxs(r, s+1, size) + [r] 
                # Step 4: Distribute data on the receiving GPUs
                for i, block_id in enumerate(to_receive):
                    # logger.debug(f"[{r}] reducing scratch_receive{r}[{i}] into output[{block_id}]")
                    chunk(r, Buffer.output, index=block_id, size=1).reduce(chunk(r, f'scratch_receive{r}', index=i, size=1))
        
        # All gather
        received = [[] for i in range(size)]
        for s in range(int(math.log2(size))-1, -1, -1):
            for r in range(size):
                peer = pi(r, s, size)
                to_send = [r] + received[r] #sends his block and the block received
                received[peer] = received[peer] + to_send #update the received of the peer

                # Step 1: Allocate scratch buffer size
                scratch_size = len(to_send)

                # Step 2: Gather non-contiguous chunks into scratch buffer of sending GPU
                for i, block_id in enumerate(to_send):
                    # logger.debug(f"[{r}] gathering output[{block_id}] input into scratch_send[{i}]")
                    chunk(r, Buffer.output, block_id, size=1).copy(r, f'scratch_send{r}', index=i)

                # Step 3: Perform a single copy operation to a scratch buffer in destination GPU
                # logger.debug(f"[{r}] sending scratch_send to scratch_receive of {peer}")
                chunk(r, f'scratch_send{r}', 0, size=scratch_size).copy(peer, f'scratch_receive{peer}', index=0, sendtb=r, recvtb=peer)

                # Step 4: Distribute data on the receiving GPU
                for i, block_id in enumerate(to_send):
                    # logger.debug(f"[{peer}] reducing scratch_receive[{i}] into output[{block_id}]")
                    chunk(peer, f'scratch_receive{peer}', index=i, size=1).copy(peer, Buffer.output, index=block_id)
            
        XML()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('pairs', type=bool, default=False, nargs='?')
args = parser.parse_args()

allreduce_swing_optimized(args.num_gpus, args.instances)