# MSCCL on Leonardo

Adapted from: [https://www.adamweingram.com/tutorial/setup-msccl/](https://www.adamweingram.com/tutorial/setup-msccl/)

To compile everything:

```bash
module load cuda
export NVCC_LOCATION=$(which nvcc)
export CUDA_HOME=$(echo "${NVCC_LOCATION}" | sed 's/\/bin\/nvcc//g')
export MPI_HOME=${OPENMPI_HOME}
module load nccl
git clone https://github.com/microsoft/msccl.git
cd msccl
make -j src.build
cd ..
git clone --depth 1 --branch v2.13.9 https://github.com/nvidia/nccl-tests.git
cd nccl-tests
make MPI=1 -j 
cd ..
```

To prepare msccl-tools:

```bash
git clone https://github.com/microsoft/msccl-tools.git
cd msccl-tools/
module load python 
source my_venv/bin/activate
pip3 install .
deactivate
cd ..
```

To generate XMLs:

```bash
source msccl-tools/my_venv/bin/activate
python msccl-tools/examples/mscclang/allreduce_a100_allpairs.py --protocol=LL 8 2 > test.xml
deactivate
```

Before running (you need to redo this if you logout/login):

```bash
export LD_LIBRARY_PATH=msccl/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=test.xml
export NCCL_ALGO=MSCCL,RING,TREE
```

When you need to run something, first get the nodes (4 GPUs on one node in this case)

```bash
salloc -p boost_usr_prod -N 1 -n 4 --gres=gpu:4 --time=02:00:00 --exclusive --account=IscrC_SHARP_0
```

Then run with:

```bash
srun -n 4 -N 1 nccl-tests/build/all_reduce_perf --minbytes 128 --maxbytes 32MB --stepfactor 2 --ngpus 1 --check 1 --iters 100 --warmup_iters 100 --cudagraph 100 --blocking 0
```