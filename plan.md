# Distributed Training Setup with DeepSpeed on a Local Network (Unknown GPU Type and Count)

This guide provides a step-by-step process to set up distributed training for large language models (e.g., LLaMA, Mistral, Falcon, OPT, BLOOM, GPT-NeoX, Qwen, Gemma) across multiple computers on a local network when the number and type of GPUs per node are unknown.

## Table of Contents

1. [Overview and Architecture](#overview-and-architecture)
2. [Prerequisites and Environment Setup](#prerequisites-and-environment-setup)
3. [Master Node Setup](#master-node-setup)
4. [Worker Node Setup](#worker-node-setup)
5. [Automatically Detecting GPU Availability](#automatically-detecting-gpu-availability)
6. [Creating a Dynamic Hostfile](#creating-a-dynamic-hostfile)
7. [DeepSpeed Configuration](#deepspeed-configuration)
8. [Integrating DeepSpeed into Your Training Script](#integrating-deepspeed-into-your-training-script)
9. [Launching Distributed Training](#launching-distributed-training)
10. [Monitoring and Debugging](#monitoring-and-debugging)
11. [Conclusion](#conclusion)

---

## Overview and Architecture

In a distributed training setup with unknown GPU types and counts, we dynamically detect available GPUs on each node (whether NVIDIA, AMD, or integrated) and create a configuration that DeepSpeed can use to allocate resources efficiently.

### Architecture Diagram

```
                +-----------------------+
                |       Master Node     |
                |   (Hostfile & Launcher)|
                +-----------+-----------+
                            |
       +-------------------------------------------+
       |                                           |
+--------------+                           +--------------+
|   Node 1     |                           |   Node 2     |
| (Detect GPUs)|                           | (Detect GPUs)|
+--------------+                           +--------------+
       |                                           |
   [DeepSpeed Processes]                     [DeepSpeed Processes]
```

Each node detects its available GPUs and reports back to the master node, which builds a dynamic hostfile for DeepSpeed.

---

## Prerequisites and Environment Setup

### Hardware Requirements
- **GPUs:** Can be NVIDIA, AMD, or integrated GPUs (Intel or AMD).
- **Network:** A reliable local network (preferably 10GbE or better).

### Software Requirements
- **Operating System:** Ubuntu or any Linux distro with GPU support.
- **Python:** Python 3.8 or later.
- **CUDA, ROCm, or OpenCL:** Install the appropriate GPU runtime libraries.
- **PyTorch:** Install a compatible version:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For NVIDIA GPUs
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2  # For AMD GPUs
  ```
- **DeepSpeed:**
  ```bash
  pip install deepspeed
  ```
- **Hugging Face Transformers (optional):**
  ```bash
  pip install transformers
  ```

---

## Master Node Setup

1. Install all prerequisites as described in the **Prerequisites** section.
2. Enable passwordless SSH access to worker nodes:
   ```bash
   ssh-keygen -t rsa -b 4096
   ssh-copy-id user@worker-node-ip
   ```
3. Set up a script to dynamically detect available nodes and GPUs (see [Creating a Dynamic Hostfile](#creating-a-dynamic-hostfile)).
4. Create and maintain the `hostfile.txt` which contains the detected worker nodes and their GPU configurations.
5. Start the DeepSpeed training job using the dynamically generated hostfile.

---

## Worker Node Setup

1. Install all prerequisites as described in the **Prerequisites** section.
2. Ensure passwordless SSH access is enabled from the master node.
3. Run the GPU detection script (`detect_gpus.sh`) so that the master node can gather the node’s GPU configuration.
4. Ensure that the worker node is reachable via SSH and can execute DeepSpeed commands remotely.
5. Once the master node initiates training, the worker node will receive training tasks automatically.

---

## Automatically Detecting GPU Availability

Since we don’t know how many GPUs each node has or their types, we create a script to automatically detect available GPUs:

```bash
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
```

Each node should run this script and report its GPU count and type to the master node.

---

## Creating a Dynamic Hostfile

The master node gathers GPU information from each participating node and builds a `hostfile.txt` dynamically:

```bash
#!/bin/bash

# List of node hostnames or IP addresses
nodes=("node1" "node2")  # Replace with actual node names or IPs

# Clear previous hostfile
> hostfile.txt

for node in "${nodes[@]}"; do
    gpu_info=$(ssh $node "./detect_gpus.sh")
    echo "$gpu_info" >> hostfile.txt
done

cat hostfile.txt
```

---

## DeepSpeed Configuration

Create a DeepSpeed configuration file (`ds_config.json`):

```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 0.00015,
      "weight_decay": 0.01
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2
  }
}
```

---

## Launching Distributed Training

### Using the Generated Hostfile

```bash
deepspeed --hostfile=hostfile.txt train_script.py --deepspeed --deepspeed_config ds_config.json
```

### No-SSH Mode (Optional)

Run manually on each node:
```bash
deepspeed --hostfile=hostfile.txt --no_ssh --node_rank=<NODE_RANK> --master_addr=<MASTER_IP> --master_port=<PORT> train_script.py --deepspeed --deepspeed_config ds_config.json
```

---

## Conclusion

This guide demonstrated how to:
- Set up the master node and worker nodes separately.
- Detect GPUs dynamically when their counts and types are unknown.
- Generate a hostfile automatically.
- Configure and launch DeepSpeed across multiple nodes.
- Monitor and debug training runs.

This approach allows efficient scaling of large language model training across a distributed system with mixed GPU availability.

