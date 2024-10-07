#!/bin/bash

# Array of sizes
size=4

# Activate the virtual environment
source msccl-tools/my_venv/bin/activate


# Generate XML files from the specified Python programs
echo "Generating XMLs for size: $size"
python msccl-tools/examples/mscclang/allreduce_swing_latency_optimal.py $size 1 > allreduce_swing_latency_optimal.xml
python msccl-tools/examples/mscclang/allreduce_swing_bandwidth_at_once.py $size 1 > allreduce_swing_bandwidth_at_once.xml
python msccl-tools/examples/mscclang/allreduce_swing_bandwidth_all_sends.py $size 1 > allreduce_swing_bandwidth_all_sends.xml
python msccl-tools/examples/mscclang/allreduce_swing_latency_sync.py $size 1 > allreduce_swing_latency_sync.xml

# Setup the environment
echo "Setting up environment for size: $size"
module load cuda
module load openmpi
module load nccl
export LD_LIBRARY_PATH=msccl/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ALGO=MSCCL,RING,TREE

# List of XML files
xml_files=(allreduce_swing_latency_optimal.xml allreduce_swing_bandwidth_at_once.xml allreduce_swing_bandwidth_all_sends.xml allreduce_swing_latency_sync.xml)

# Run tests for each XML file
for xml in "${xml_files[@]}"; do
    # Create the output file name based on XML file and size
    output_file="${xml%.xml}_size${size}.txt"
    echo "Running tests for $xml with size: $size, outputting to $output_file"

    # Clear the output file before starting
    > $output_file

    export MSCCL_XML_FILES=$xml

    # Run the test 5 times and append results to the output file
    for i in {1..5}; do
        echo "Test run $i for $xml with size: $size" | tee -a $output_file
        srun -n $size -N $(($size / 4)) nccl-tests/build/all_reduce_perf \
            --minbytes 128 --maxbytes 32MB --stepfactor 2 \
            --ngpus 1 --check 1 --iters 100 --warmup_iters 100 \
            --cudagraph 100 --blocking 0 | tee -a $output_file
    done
done

