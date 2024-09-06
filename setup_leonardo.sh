#!/bin/bash
# salloc -N 1 -n 4 --gres=gpu:4 --time=02:00:00 --exclusive --account=IscrC_SHARP_0 -p boost_usr_prod 
module load cuda
module load openmpi
export MPI_HOME=/leonardo/prod/opt/libraries/openmpi/4.1.6/gcc--12.2.0

# Currently Loaded Modulefiles:
# 1) profile/base   2) cintools/1.0   3) nccl/2.19.1-1--gcc--12.2.0-cuda-12.1   4) cuda/12.1   5) gcc/12.2.0   6) openmpi/4.1.6--gcc--12.2.0  

#after this you can compile the tests with $ make MPI=1 NCCL_HOME=../msccl/build/ -j 
#then you can directly run the tests, for example
# mpirun -np 1 nccl-tests/build/all_reduce_perf \
#         --minbytes 128 \
#         --maxbytes 32MB \
#         --stepfactor 2 \
#         --ngpus 1 \
#         --check 1 \
#         --iters 100 \
#         --warmup_iters 100 \
#         --cudagraph 100 \
#         --blocking 0


# to compile python stuff
module load python/3.12 #python version ^3.12
# python3 -m venv my_venv
source my_venv/bin/activate


export LD_LIBRARY_PATH=msccl/build/lib/:$LD_LIBRARY_PATH

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=test.xml
export NCCL_ALGO=MSCCL,RING,TREE