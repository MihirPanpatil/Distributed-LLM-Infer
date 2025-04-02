Here is a comprehensive, step-by-step explanation of the entire procedure, from setting up the environment to running training and adapting for inference, using the files we've created.

**Goal:** To train or run inference on Large Language Models (LLMs) like LLaMA, Mistral, Falcon, etc., across multiple computers (nodes) on your local network using DeepSpeed for efficiency, even if the computers have different types or numbers of GPUs.

**Overview:** We have created two main sets of files:
*   `master_node_setup/`: Contains scripts and configurations run *on* the main computer (master node) that controls the process.
*   `worker_node_setup/`: Contains a helper script needed *on* all the other computers (worker nodes) participating in the task.

---

**Phase 1: One-Time Environment Setup (Done on ALL Nodes)**

This phase prepares all computers involved (the master and all workers).

1.  **Install Prerequisites:**
    *   **On every node (master and all workers):** Ensure you have the necessary software installed. This typically includes:
        *   Python (version 3.8 or later recommended).
        *   PyTorch (compatible with your GPUs and desired DeepSpeed version).
        *   DeepSpeed (`pip install deepspeed`).
        *   Hugging Face Transformers (`pip install transformers`).
        *   Appropriate GPU drivers (NVIDIA drivers for NVIDIA GPUs, ROCm for AMD GPUs).
        *   GPU detection tools (`nvidia-smi` for NVIDIA, `rocminfo` for AMD, `clinfo` for others - usually installed with drivers).
    *   *Consistency is key!* Try to use the same major versions of PyTorch and DeepSpeed across all nodes.

2.  **Set Up SSH Access (Master -> Workers):**
    *   The master node needs to connect to worker nodes automatically to start tasks. The recommended way is using SSH keys (passwordless SSH).
    *   **Follow the detailed, step-by-step guide added in `master_node_setup/README.md` under the section "Setting Up Passwordless SSH (Using Keys - Recommended)".** This involves generating a key pair on the master and copying the public key to each worker using `ssh-copy-id`.
    *   *Alternatively*, if you cannot use keys, you can use password authentication by providing the `--ssh-password` flag when running the setup script later, but this is less secure and requires manual password entry.

3.  **Deploy Worker Script:**
    *   **On each worker node:** Copy the entire `worker_node_setup` directory to a known location. A common place is the user's home directory (e.g., `/home/worker_user/worker_node_setup/`).
    *   **On each worker node:** Make the detection script executable. Open a terminal on the worker and run:
        ```bash
        chmod +x ~/worker_node_setup/detect_gpus.sh
        ```
        (Adjust the path `~/worker_node_setup/` if you copied it elsewhere).

---

**Phase 2: Preparing for a Training Run (Done on Master Node)**

This phase is done each time you want to start a new training job or update the list of participating nodes.

1.  **Navigate to Setup Directory:**
    *   Open a terminal on the **master node**.
    *   Change directory to where the `master_node_setup` directory is located. For example: `cd /path/to/your/project/master_node_setup`

2.  **Run the Master Setup Script:**
    *   Make the script executable (only needs to be done once):
        ```bash
        chmod +x run_master_setup.sh
        ```
    *   Execute the script. You **must** provide details about your worker nodes using command-line arguments. Key arguments (see `./run_master_setup.sh --help` for all options, inherited from `generate_hostfile_py.py`):
        *   `--nodes`: List of worker IP addresses or hostnames (e.g., `--nodes 192.168.1.101 192.168.1.102 worker3.local`).
        *   `--script-path`: The *full path* to `detect_gpus.sh` *on the worker nodes* (e.g., `--script-path /home/worker_user/worker_node_setup/detect_gpus.sh` or `~/worker_node_setup/detect_gpus.sh`). This must match where you copied it in Phase 1.
        *   `--ssh-user`: The username to log into the worker nodes (e.g., `--ssh-user worker_user`).
        *   `--ssh-key`: Path to the private SSH key on the master node to use for authentication (e.g., `--ssh-key ~/.ssh/id_rsa`). This is used if you set up passwordless SSH.
        *   `--ssh-password`: Use this flag *instead* of `--ssh-key` if you want to be prompted for the SSH password for the workers.
    *   **Example Execution:**
        ```bash
        ./run_master_setup.sh --nodes 192.168.1.101 192.168.1.102 --script-path ~/worker_node_setup/detect_gpus.sh --ssh-user worker_user --ssh-key ~/.ssh/id_rsa
        ```
    *   **What it does:**
        *   Checks if the `paramiko` Python library is installed (needed for SSH) and offers to install it via `pip`.
        *   Runs the Python script `generate_hostfile_py.py` using the arguments you provided.
        *   `generate_hostfile_py.py` connects to each worker via SSH, runs the `detect_gpus.sh` script there, and collects the output.
        *   It creates the `hostfile.txt` file, listing the workers and their detected GPU slots (e.g., `192.168.1.101 slots=4 type=NVIDIA`).
        *   Prints the generated `hostfile.txt` content for verification.
        *   Prints a template command for launching the DeepSpeed training job.

3.  **Modify Training Script for Your Data:**
    *   **Crucial Step:** Open the `train_script.py` file in a text editor.
    *   Find the function `prepare_dataset`.
    *   **Replace the placeholder dummy data logic** with code that loads and preprocesses *your actual training dataset*. This will involve using libraries like `datasets` from Hugging Face or standard Python file I/O, and tokenizing the data using the loaded `tokenizer`. The function should return a PyTorch-compatible dataset (like `torch.utils.data.Dataset` or `torch.utils.data.TensorDataset`).

---

**Phase 3: Launching Distributed Training (Done on Master Node)**

1.  **Use the Launch Command:**
    *   Copy the `deepspeed` command template printed by the `run_master_setup.sh` script at the end of Phase 2.
    *   **Modify the command:**
        *   Replace `<your_model_identifier>` with the actual Hugging Face model name or path you want to train (e.g., `mistralai/Mistral-7B-v0.1`, `google/gemma-7b`, `/path/to/local/model`).
        *   Add any other arguments needed by `train_script.py`, such as `--num_epochs <number>` or arguments related to your specific dataset loading.
    *   **Example Launch Command:**
        ```bash
        deepspeed --hostfile=hostfile.txt train_script.py \
            --model_name_or_path mistralai/Mistral-7B-v0.1 \
            --num_epochs 3 \
            --deepspeed \
            --deepspeed_config ds_config.json \
            # --dataset_path /path/to/my/data # Example if you added this arg
        ```
2.  **Execute the Command:**
    *   Run this complete command in the terminal on the master node (from the `master_node_setup` directory).
    *   DeepSpeed will read `hostfile.txt`, connect to the listed worker nodes via SSH, and start the `train_script.py` process on the specified number of GPUs across all nodes.

---

**Phase 4: Monitoring Training**

1.  **Master Node Terminal:** Watch the output in the terminal where you launched the `deepspeed` command. You should see:
    *   Initialization messages from DeepSpeed on different ranks (processes).
    *   Output from `train_script.py`, including loading messages and periodic loss/step updates (as configured in the script).
    *   Any errors encountered during training.
2.  **Worker Node Logs (Optional):** Depending on the configuration and potential errors, logs or error messages might also appear in system logs or user directories on the worker nodes.

---

**Phase 5: Adapting for Inference**

The current setup is focused on *training*. To perform *inference* (using a trained model to generate text, answer questions, etc.) efficiently with DeepSpeed, especially for large models, you'll need to adapt:

1.  **Save a Trained Model:**
    *   During or after training, you need to save the model weights. DeepSpeed's `model_engine.save_checkpoint("my_checkpoint_dir")` saves the model, optimizer state, etc., in a DeepSpeed-specific format.
    *   To get a standard Hugging Face model format (useful for easier inference later), you might need to consolidate the weights, especially if using ZeRO Stage 3. This often involves loading the checkpoint and then saving using the underlying Hugging Face model's `save_pretrained` method (e.g., `model_engine.module.save_pretrained("my_hf_model_dir")`). Consult DeepSpeed documentation for the exact procedure based on your ZeRO stage.

2.  **Create an Inference Script:**
    *   Create a new Python script (e.g., `inference_script.py`) in the `master_node_setup` directory.
    *   This script should:
        *   Load the tokenizer and the *trained* model checkpoint (either the DeepSpeed checkpoint or the consolidated Hugging Face format).
        *   **Initialize DeepSpeed for Inference:** Use `model_engine = deepspeed.init_inference(...)` instead of `deepspeed.initialize(...)`. This requires a different DeepSpeed configuration.
        *   Define an inference-specific DeepSpeed JSON config (e.g., `ds_inference_config.json`). This config often uses ZeRO Stage 3 (to fit large models in memory across GPUs) and might specify `mp_size` (tensor-parallel degree) for further optimization.
        *   Include logic to take input text, tokenize it, and use the `model_engine.module.generate(...)` method to produce output.
        *   Decode the generated output tokens back into text.

3.  **Create Inference Config (`ds_inference_config.json`):**
    *   Create a JSON file similar to `ds_config.json` but tailored for inference. Example focus points:
        *   `"zero_optimization": {"stage": 3}` (Common for large models)
        *   May not need optimizer/scheduler sections.
        *   Might include inference-specific settings like `mp_size`, `dtype`.

4.  **Launch Distributed Inference:**
    *   You can still use the `deepspeed` launcher with your `hostfile.txt` if you want to distribute the inference workload across multiple nodes/GPUs (useful if the model is too large for one node or you need high throughput).
    *   The launch command would look like:
        ```bash
        deepspeed --hostfile=hostfile.txt inference_script.py \
            --model_name_or_path /path/to/your/trained_hf_model_dir \
            --deepspeed \
            --deepspeed_config ds_inference_config.json \
            # Add any other args needed by inference_script.py
        ```

---

**Conclusion:**

This workflow allows you to leverage multiple machines for demanding LLM tasks. The key phases are the initial environment setup (including SSH), preparing the master node using the automation script, modifying the training script for your data, launching the job via `deepspeed`, and finally adapting the process for inference using DeepSpeed's inference engine if needed. Remember to consult the `README.md` files and the scripts' help options for specific configuration details.

```tool_code