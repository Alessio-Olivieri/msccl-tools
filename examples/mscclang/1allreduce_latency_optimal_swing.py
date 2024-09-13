import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.collectives import *
from msccl.language.collectives import AllReduce
import math
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def send_whole_buffer(source_rank, dest_rank, buffer_name='input'):
    # Reference the source buffer (input buffer in this case)
    src_buffer = buffer_name

    # Initialize the chunk index
    chunk_index = 0
    
    print(f"Starting to send buffer '{buffer_name}' from rank {source_rank} to rank {dest_rank}")
    
    # Dynamically figure out the number of chunks
    while True:
        try:
            # Log before each chunk is sent
            print(f"Copying chunk {chunk_index} from rank {source_rank} to rank {dest_rank}")
            
            # Access each chunk and copy it
            chunk(source_rank, src_buffer, chunk_index).copy(dest_rank, src_buffer, chunk_index)
            
            # Log after chunk is successfully copied
            print(f"Successfully copied chunk {chunk_index}")
            
            chunk_index += 1
        except:
            # Log when no more chunks are available (i.e., when the loop breaks)
            print(f"No more chunks to copy. Finished sending buffer '{buffer_name}' after {chunk_index} chunks.")
            break


def allreduce_swing(size, instances):
    # Logical topology
    topology = fully_connected(size)
    collective = AllReduce(size, size, inplace=True)

    with MSCCLProgram("allreduce_swing", topology, collective, instances):
        def pi(r, s, n):
            p = (1 - math.pow(-2, s+1))/3
            peer = (r + p) % n if r % 2 == 0 else (r - p) % n
            logger.debug(f"pi calculation for rank {r}, step {s}, size {n}: peer={peer}")
            return int(peer)
        
        send_whole_buffer(0,1)
        
        # for step in range(int(math.log2(size))):
        #     logger.debug(f"Starting step {step}")

        #     # Each rank sends the nth chunk to the nth rank into scratch space
        #     for r1 in range(size):
        #         for r2 in range(size):
        #             if r1 != r2:
        #                 index = r2 * size
        #                 c = chunk(r1, Buffer.input, index, size=size)
        #                 c.copy(r2, 'scratch', sendtb=r2, recvtb=r1)

        # Check()
        # XML()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')

args = parser.parse_args()

allreduce_swing(args.num_gpus, args.instances)