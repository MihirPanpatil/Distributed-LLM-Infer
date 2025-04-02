# Worker Node Setup for Distributed DeepSpeed Training

This directory contains the necessary scripts required on each worker node participating in the distributed training setup orchestrated by the master node.

## Files

1.  **`detect_gpus.sh`**:
    *   A bash script designed to detect the number and type (NVIDIA, AMD, OpenCL/Integrated) of GPUs available on the worker node where it is run.
    *   It outputs a single line formatted for the DeepSpeed hostfile, e.g., `worker-hostname slots=4 type=NVIDIA`.
    *   **Requires**: Appropriate command-line tools installed for GPU detection (`nvidia-smi` for NVIDIA, `rocminfo` for AMD, `clinfo` for OpenCL). If none are found, it reports 0 slots.
    *   **Usage**: This script is intended to be executed remotely by the master node's `generate_hostfile.sh` script via SSH. Ensure this script is placed in a consistent, accessible location on all worker nodes (e.g., user's home directory or a project directory) and is executable (`chmod +x detect_gpus.sh`).

## Setup Instructions for Worker Nodes

1.  **Prerequisites**:
    *   Ensure all prerequisites (Python, PyTorch, DeepSpeed, Transformers, GPU drivers/runtimes like CUDA/ROCm, and GPU detection tools like `nvidia-smi`/`rocminfo`/`clinfo`) are installed.
    *   Ensure the operating system and core libraries (Python, PyTorch version) are consistent with the master node and other worker nodes.

2.  **SSH Access**:
    *   Ensure the master node has passwordless SSH access to this worker node. This usually involves adding the master node's public SSH key to the worker node's `~/.ssh/authorized_keys` file.

3.  **Place `detect_gpus.sh`**:
    *   Copy this `worker_node_setup` directory (or at least the `detect_gpus.sh` script) to a known location on this worker node (e.g., `~/worker_node_setup/detect_gpus.sh`).
    *   Make the script executable: `chmod +x detect_gpus.sh`.

4.  **Network Configuration**:
    *   Ensure the worker node is reachable from the master node via its hostname or IP address over the local network.
    *   Check firewall settings to allow SSH connections (typically port 22) and potentially other ports required for distributed training communication (often handled by libraries like NCCL).

5.  **Wait for Master**:
    *   Once set up, the worker node is ready. The master node will initiate the training process, and DeepSpeed will manage launching the necessary training processes on this worker node via SSH. No manual execution of the training script is typically needed on the worker nodes when using the standard DeepSpeed launcher with a hostfile.