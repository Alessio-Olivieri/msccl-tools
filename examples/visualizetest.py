# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.collectives import *
from msccl.language.collectives import AllReduce


def test(size, instances):
    # Logical topology
    topology = fully_connected(size)
    collective = AllReduce(size, size, inplace=False)

    with MSCCLProgram("allreduce_ring_inplace", topology, collective, instances):
        chunk(0, Buffer.input, index=1, size=3).copy(0, "pippo", index=0)
        chunk(0, Buffer.input, index=1, size=3).copy(0, "pippo1", index=0)
        chunk(0, Buffer.input, index=1, size=3).copy(0, "pippo3", index=0)

        XML()
        Print_instruction_dag()

# parser = argparse.ArgumentParser()
# parser.add_argument('num_gpus', type=int, help ='number of gpus')
# parser.add_argument('instances', type=int, help='number of instances')

# args = parser.parse_args()

# allreduce_ring(args.num_gpus, args.instances)

test(4, 1)
