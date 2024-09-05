#!/bin/bash
# salloc -N 1 -n 4 --gres=gpu:4 --time=02:00:00 --exclusive --account=IscrC_SHARP_0 -p boost_usr_prod 
module load cuda
module load openmpi
export MPI_HOME=/leonardo/prod/opt/libraries/openmpi/4.1.6/gcc--12.2.0
export LD_LIBRARY_PATH=msccl/build/lib/:$LD_LIBRARY_PATH

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=test.xml
export NCCL_ALGO=MSCCL,RING,TREE