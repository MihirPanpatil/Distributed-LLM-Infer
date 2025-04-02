#!/bin/bash

# Detect GPU Type
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    gpu_type="NVIDIA"
elif command -v rocminfo &> /dev/null; then
    gpu_count=$(rocminfo | grep 'GPU' | wc -l)
    gpu_type="AMD"
elif command -v clinfo &> /dev/null; then
    gpu_count=$(clinfo | grep 'Device Name' | wc -l)
    gpu_type="OpenCL (Intel/AMD Integrated)"
else
    gpu_count=0
    gpu_type="None"
fi

echo "$HOSTNAME slots=$gpu_count type=$gpu_type"