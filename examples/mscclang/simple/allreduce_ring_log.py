import argparse
import logging
from msccl.language import *
from msccl.topologies import *
from msccl.collectives import *
from msccl.language.collectives import AllReduce

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def allreduce_ring(size, instances):
    # Logical topology
    topology = fully_connected(size)
    collective = AllReduce(size, size, inplace=True)

    logging.info(f"Starting AllReduce with {size} GPUs and {instances} instances.")
    
    with MSCCLProgram("allreduce_ring_inplace", topology, collective, instances):
        for r in range(size):
            index = r
            logging.debug(f"Chunking at rank {r}, index {index}.")
            
            # (rank, buffer, index)
            c = chunk(r, Buffer.input, index)
            next = (r + 1) % size
            
            logging.debug(f"Initial chunk created at rank {r}. Next rank is {next}.")
            
            # Chunk travels around the ring being reduced
            while next != r:
                logging.debug(f"Reducing chunk at rank {r} with chunk from rank {next}.")
                c1 = chunk(next, buffer=Buffer.input, index=r)
                c = c1.reduce(c)
                logging.debug(f"Chunk reduced at rank {r} by rank {next}.")
                next = (next + 1) % size
            
            logging.debug(f"Starting to send fully reduced chunk around the ring starting from rank {r}.")
            
            # Send the fully reduced chunk around the ring
            while next != (r - 1) % size:
                logging.debug(f"Copying chunk from rank {r} to next rank {next}.")
                c = c.copy(next, buffer=Buffer.input, index=r)
                next = (next + 1) % size

        logging.info(f"All chunks have been reduced and sent across the ring for {size} GPUs.")
        
        Check()
        XML()

    logging.info("MSCCL program completed successfully.")

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')

args = parser.parse_args()

logging.info(f"Program started with {args.num_gpus} GPUs and {args.instances} instances.")
allreduce_ring(args.num_gpus, args.instances)
logging.info("Program execution finished.")
