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

def allreduce_swing(size, instances, pairs):
    # Logical topology
    topology = fully_connected(size)
    collective = AllReduce(size, size, inplace=True)

    with MSCCLProgram("allreduce_swing", topology, collective, instances):
        if not pairs:
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
        
        if pairs:

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

            for r, extrar in siblings:
                peer_scratch = chunk(extrar, Buffer.input, 0, size=size).copy(r, 'scratch', 0, sendtb=extrar, recvtb=r)
                chunk(r, Buffer.input, 0, size=size).reduce(peer_scratch)

            # run swing on log2 ranks
            for step in range(int(math.log2(size_log2))):
                logger.debug(f"Starting step {step}")
                # Each rank sends it's buffer to it's peer and the peer performs a reduce
                for r in range(size_log2):
                    peer = pi(r, step, size_log2)
                    logger.debug(f"copying input buffer of {aliases[r]} into scratch of {aliases[r]}")
                    chunk(aliases[r], Buffer.input, index=0, size=size).copy(aliases[peer], 'scratch', 0, recvtb=aliases[peer], sendtb=aliases[r])
                for r in range(size_log2):
                    logger.debug(f"reducing scratch buffer of {aliases[r]} into the input buffer of {aliases[r]}")
                    chunk(aliases[r], Buffer.input, 0, size=size).reduce(chunk(aliases[r], 'scratch', 0, size=size))
            
            for r, extrar in siblings:
                chunk(r, Buffer.input, 0, size=size).copy(extrar, Buffer.input, 0, recvtb=extrar, sendtb=r)

        XML()
        Check()            

def allreduce_swing_optimized(size, instances, pairs):
    # This version of the latency optimal makes so that each ranks start reducing when both peer and ranks both sent and recived the correct chunks
    # Logical topology
    topology = fully_connected(size)
    collective = AllReduce(size, size, inplace=True)

    with MSCCLProgram("allreduce_swing", topology, collective, instances):
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

        for r, extrar in siblings:
            peer_scratch = chunk(extrar, Buffer.input, 0, size=size).copy(r, 'scratch', 0, sendtb=extrar, recvtb=r)
            chunk(r, Buffer.input, 0, size=size).reduce(peer_scratch)

        # run swing on log2 ranks
        for step in range(int(math.log2(size_log2))):
            ranks = list(range(size))
            peers = [pi(i, step, size) for i in ranks]
            done = [0]*size
            logger.debug(f"Starting step {step}")
            # Each rank sends it's buffer to it's peer and the peer performs a reduce
            for r in range(size_log2):
                done[r] = 1
                peer = pi(r, step, size_log2)
                logger.debug(f"copying input buffer of {aliases[r]} into scratch of {aliases[r]}")
                chunk(aliases[r], Buffer.input, index=0, size=size).copy(aliases[peer], 'scratch', 0, recvtb=aliases[peer], sendtb=aliases[r])
                if done[peer]:# if r and peer exchanged their data they can start the reduce
                    logger.debug(f"reducing scratch buffer of {aliases[peer]} into the input buffer of {aliases[peer]}")
                    chunk(aliases[peer], Buffer.input, 0, size=size).reduce(chunk(aliases[peer], 'scratch', 0, size=size))
                    logger.debug(f"reducing scratch buffer of {aliases[r]} into the input buffer of {aliases[r]}")
                    chunk(aliases[r], Buffer.input, 0, size=size).reduce(chunk(aliases[r], 'scratch', 0, size=size))
        
        for r, extrar in siblings:
            chunk(r, Buffer.input, 0, size=size).copy(extrar, Buffer.input, 0, recvtb=extrar, sendtb=r)
        
        XML()
        Check()



parser = argparse.ArgumentParser()
# parser.add_argument('num_gpus', type=int, help ='number of gpus')
# parser.add_argument('instances', type=int, help='number of instances')
# parser.add_argument('pairs', type=bool, default=False, nargs='?')
# args = parser.parse_args()

# allreduce_swing(args.num_gpus, args.instances, args.pairs)
allreduce_swing_optimized(9, 1, True)