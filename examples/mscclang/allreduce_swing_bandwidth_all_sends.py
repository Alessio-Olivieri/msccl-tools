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

def allreduce_swing_all_sends(size, instances):
    # Logical topology
    topology = fully_connected(size)
    collective = AllReduce(size, size, inplace=True)

    with MSCCLProgram("allreduce_swing", topology, collective, instances):

        # Reduce scatter
        for s in range(int(math.log2(size))):
            #Sending blocks
            for r in range(size):
                peer = pi(r, s, size)
                # Here we calculate all blocks that should be sent
                to_send = get_rs_idxs(peer, s+1, size) + [peer] 
                for block_id in to_send:
                    # logger.debug(f"[{r}] sending chunk {block_id}I into scratch of {peer}")
                    chunk(r, Buffer.input, block_id, size=1).copy(peer, 'scratch', index=block_id, sendtb=r, recvtb=peer)

            #reducing blocks
            for r in range(size):
                peer = pi(r, s, size)
                # Here we calculate all blocks that should be sent
                to_reduce = get_rs_idxs(peer, s+1, size) + [peer] 
                for block_id in to_reduce:
                    # logger.debug(f"[{peer}] Reducing chunk {block_id} in input buffer")
                    chunk(peer, Buffer.input, block_id, size=1).reduce(chunk(peer, 'scratch', block_id, size=1))
        
        # All gather
        received = [[] for i in range(size)]
        for s in range(int(math.log2(size))-1, -1, -1):
            for r in range(size):
                peer = pi(r, s, size)
                to_send = [r] + received[r] #sends his block and the block received
                received[peer] = received[peer] + to_send #update the received of the peer
                for block_id in to_send:
                    chunk(r, Buffer.input, index=block_id, size=1).copy(peer, Buffer.input, index=block_id)
                
            
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('pairs', type=bool, default=False, nargs='?')
args = parser.parse_args()

allreduce_swing_all_sends(args.num_gpus, args.instances)