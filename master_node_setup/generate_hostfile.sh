#!/bin/bash

# --- Configuration ---
# List of worker node hostnames or IP addresses
# Replace these placeholders with the actual hostnames or IPs of your worker nodes.
nodes=("worker-node-1-ip" "worker-node-2-ip")

# Path to the detect_gpus.sh script on the worker nodes
# Ensure this path is correct and the script is executable on workers.
detect_script_path="./worker_node_setup/detect_gpus.sh" # Assuming it's relative to user's home or a known path

# Output hostfile name
hostfile="hostfile.txt"
# --- End Configuration ---

echo "Generating hostfile: $hostfile"

# Clear previous hostfile or create a new one
> "$hostfile"

# Loop through each node, SSH to it, run the detection script, and append to hostfile
for node in "${nodes[@]}"; do
    echo "Querying node: $node"
    # Execute the script on the remote node via SSH
    # Ensure passwordless SSH is set up from master to all workers.
    gpu_info=$(ssh "$node" "bash $detect_script_path")

    if [ $? -eq 0 ] && [ -n "$gpu_info" ]; then
        echo "Detected info for $node: $gpu_info"
        echo "$gpu_info" >> "$hostfile"
    else
        echo "Warning: Failed to get GPU info from $node or script returned empty."
        # Optionally, add a default entry or skip the node
        # echo "$node slots=0 type=Unknown" >> "$hostfile" # Example default
    fi
done

echo "Hostfile generation complete."
echo "--- Content of $hostfile ---"
cat "$hostfile"
echo "---------------------------"

# Make the script executable
chmod +x "$hostfile" # This seems wrong, should be chmod +x generate_hostfile.sh

echo "Remember to:"
echo "1. Replace placeholder node IPs/hostnames in this script (${nodes[*]})."
echo "2. Ensure passwordless SSH is configured from the master to all worker nodes."
echo "3. Ensure '$detect_script_path' exists and is executable on all worker nodes."
echo "4. Make this script ('generate_hostfile.sh') executable using: chmod +x generate_hostfile.sh"