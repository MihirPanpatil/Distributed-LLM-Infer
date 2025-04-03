#!/bin/bash

# Detect all available resources
cpu_cores=$(nproc --all)
gpu_count=0
gpu_type="None"

# Check for GPUs
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    gpu_type="NVIDIA"
elif command -v rocminfo &> /dev/null; then
    gpu_count=$(rocminfo | grep 'GPU' | wc -l)
    gpu_type="AMD"
elif command -v clinfo &> /dev/null; then
    gpu_count=$(clinfo | grep 'Device Name' | wc -l)
    gpu_type="OpenCL"
fi

# Output both GPU and CPU resources
# Format: hostname slots=[total_workers] gpu_slots=N cpu_slots=M
total_slots=$((gpu_count > 0 ? gpu_count : cpu_cores))
echo "$HOSTNAME slots=$total_slots gpu_slots=$gpu_count cpu_slots=$cpu_cores gpu_type=$gpu_type"