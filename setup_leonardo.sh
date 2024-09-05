#!/bin/bash
# salloc -N 1 -n 4 --gres=gpu:4 --time=02:00:00 --exclusive --account=IscrC_SHARP_0 -p boost_usr_prod 
module load cuda
module load openmpi
export NVCC_LOCATION=$(which nvcc)