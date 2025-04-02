# Master Node Setup for Distributed DeepSpeed Training

This directory contains the necessary scripts and configuration files to run on the master node for initiating distributed LLM training using DeepSpeed across a local network.

## Files

1.  **`generate_hostfile_py.py`**:
    *   A Python script using the `paramiko` library to automatically detect available GPUs on worker nodes via SSH and generate the `hostfile.txt` required by DeepSpeed.
    *   **Requires**: Python 3 and `paramiko` library (`pip install paramiko`) installed on the master node. SSH access (key-based or password) from the master node to all worker nodes.
    *   **Configuration**: Worker node IPs/hostnames, the path to `detect_gpus.sh` on workers, and SSH credentials can be configured via command-line arguments or by editing the defaults within the script.
    *   **Usage**: Run `python generate_hostfile_py.py [options]` on the master node *before* starting the training. Use `python generate_hostfile_py.py --help` to see available options (e.g., `--nodes`, `--script-path`, `--ssh-user`, `--ssh-key`, `--ssh-password`).

2.  **`ds_config.json`**:
    *   The DeepSpeed configuration file. It defines parameters for training optimization, such as the ZeRO optimization stage, mixed-precision settings (FP16/BF16), optimizer type, learning rate scheduler, and gradient accumulation.
    *   The provided configuration uses ZeRO Stage 2 with CPU offloading for optimizer states and parameters, and enables FP16. Adjust these settings based on your specific hardware (GPU memory) and model size. Many values are set to `"auto"` allowing DeepSpeed to infer optimal values.

3.  **`train_script.py`**:
    *   The main Python script for training. It uses Hugging Face `transformers` to load the specified LLM and integrates with DeepSpeed for distributed training.
    *   **Arguments**:
        *   `--model_name_or_path`: (Required) Specify the Hugging Face model identifier (e.g., `mistralai/Mistral-7B-v0.1`) or local path.
        *   `--num_epochs`: Number of training epochs (default: 1).
        *   `--max_steps`: Maximum training steps (overrides epochs if > 0).
        *   `--deepspeed`: Flag required by DeepSpeed launcher.
        *   `--deepspeed_config`: Path to the `ds_config.json` file.
        *   `--local_rank`: Automatically provided by the DeepSpeed launcher.
    *   **Dataset**: Currently uses placeholder dummy data. You **must** modify the `prepare_dataset` function to load and preprocess your actual training data.

4.  **`hostfile.txt`** (Generated):
    *   This file is created by `generate_hostfile_py.py`. It lists the worker nodes and the number of GPU slots detected on each, formatted for DeepSpeed. Example:
        ```
        worker-node-1-ip slots=4 type=NVIDIA
        worker-node-2-ip slots=2 type=AMD
        ```
 5.  **`run_master_setup.sh`**:
    *   An automation script to simplify the master node setup process.
    *   It checks for the `paramiko` dependency (and offers to install it), removes the old `generate_hostfile.sh` (if found), runs `generate_hostfile_py.py` (passing along any arguments you provide), and displays the final `deepspeed` command to launch training.
    *   **Usage**: Make it executable (`chmod +x run_master_setup.sh`) and run it: `./run_master_setup.sh [options_for_generate_hostfile_py.py]`. For example: `./run_master_setup.sh --nodes <ip1> <ip2> --script-path <path_on_worker>`.

## How to Run Training

1.  **Prerequisites**:
    *   Ensure all prerequisites (Python 3, PyTorch, DeepSpeed, Transformers, GPU drivers/runtimes like CUDA/ROCm) are installed on the master and all worker nodes.
    *   Ensure SSH access is configured from the master node to all worker nodes. The setup script (`run_master_setup.sh`) uses Python's `paramiko` library, which supports SSH keys (recommended for automation) or password authentication (using the `--ssh-password` flag). **Using SSH keys is strongly recommended for a smoother setup.** See the section below on how to set this up.
    *   Copy the `worker_node_setup` directory (containing `detect_gpus.sh`) to a consistent location on all worker nodes (e.g., the user's home directory, matching the `--script-path` argument used in the next step, default `~/worker_node_setup/detect_gpus.sh`) and ensure `detect_gpus.sh` is executable (`chmod +x detect_gpus.sh`).

### Setting Up Passwordless SSH (Using Keys - Recommended)

For the master node to automatically connect to worker nodes without asking for a password each time, you can set up SSH keys. Think of it like giving the master node a special key that unlocks access to the worker nodes.

**You only need to do these steps once.**

**Step 1: Create an SSH Key Pair on the Master Node**

   *   Open a terminal on the **master node**.
   *   Run the following command:
     ```bash
     ssh-keygen -t rsa -b 4096
     ```
   *   It will ask where to save the key. Just press **Enter** to accept the default location (usually `~/.ssh/id_rsa`).
   *   It will ask for a passphrase. **For passwordless login, just press Enter twice (leaving it empty).** *Warning: This means anyone who gets access to your master node user account can also access the workers.*
   *   This creates two files in your `~/.ssh/` directory:
      *   `id_rsa`: Your **private key** (keep this secret!)
      *   `id_rsa.pub`: Your **public key** (this is what you share)

**Step 2: Copy the Public Key to Each Worker Node**

   *   You need to tell each worker node to trust the master node's public key. The easiest way is using the `ssh-copy-id` command **from the master node's terminal**.
   *   For **each worker node**, run this command, replacing `worker_user` with the username on the worker and `worker_ip_or_hostname` with its actual IP address or hostname:
     ```bash
     ssh-copy-id worker_user@worker_ip_or_hostname
     ```
   *   It will likely ask for the `worker_user`'s password **one last time**. Enter it.
   *   This command automatically adds your master node's public key (`~/.ssh/id_rsa.pub`) to the `~/.ssh/authorized_keys` file on the worker node.

   *   **Repeat this `ssh-copy-id` command for every worker node.**

   *   *(Alternative if `ssh-copy-id` fails):* You can manually copy the *content* of the master node's `~/.ssh/id_rsa.pub` file and paste it into a new line in the `~/.ssh/authorized_keys` file on each worker node. You might need to create the `.ssh` directory (`mkdir ~/.ssh`) and the `authorized_keys` file (`touch ~/.ssh/authorized_keys`) on the worker first, and set correct permissions (`chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys`).

**Step 3: Test the Connection**

   *   From the **master node's terminal**, try logging into a worker node:
     ```bash
     ssh worker_user@worker_ip_or_hostname
     ```
   *   It should log you in **without asking for a password**. If it works, type `exit` to come back to the master node.
   *   Test this for each worker node.

Now, the `run_master_setup.sh` script (when using the default key file `~/.ssh/id_rsa` or specifying it with `--ssh-key`) should be able to connect to the workers automatically.

2.  **Run Master Setup Script**:
    *   Make the setup script executable: `chmod +x run_master_setup.sh`.
    *   Execute the script, passing necessary arguments for hostfile generation (like worker node IPs/hostnames and the path to `detect_gpus.sh` on workers). Example:
        ```bash
        ./run_master_setup.sh --nodes 192.168.1.101 192.168.1.102 --script-path ~/worker_node_setup/detect_gpus.sh --ssh-user worker_user --ssh-key ~/.ssh/id_rsa_worker
        ```
    *   (The script will check for `paramiko` and offer to install it if missing).
    *   Verify the generated `hostfile.txt` looks correct based on the script's output.

3.  **Launch Training**:
    *   The `run_master_setup.sh` script will print the command needed to launch training. Copy, paste, and edit it as needed (especially replacing `<your_model_identifier>`). Example structure:
        ```bash
        deepspeed --hostfile=hostfile.txt train_script.py \
            --model_name_or_path <your_model_identifier> \
            --deepspeed \
            --deepspeed_config ds_config.json \
            # Add other arguments like --num_epochs or --dataset_path if needed
        ```
    *   Replace `<your_model_identifier>` with the desired model (e.g., `google/gemma-7b`).

4.  **Monitor**: Observe the output on the master node's terminal for training progress and any errors. Logs might also appear on worker nodes depending on the setup.