#!/bin/bash

# Master setup script for DeepSpeed distributed training

# --- Configuration ---
PYTHON_CMD="python3" # Command to run python
HOSTFILE_GEN_SCRIPT="generate_hostfile_py.py"
OLD_HOSTFILE_GEN_SCRIPT="generate_hostfile.sh"
HOSTFILE_OUTPUT="hostfile.txt" # Should match the default in the python script or be passed as arg
DS_CONFIG="ds_config.json"
TRAIN_SCRIPT="train_script.py"
# --- End Configuration ---

echo "--- Starting Master Node Setup ---"

# 1. Check for paramiko dependency
echo "[Step 1/4] Checking for 'paramiko' Python library..."
if $PYTHON_CMD -c "import paramiko" &> /dev/null; then
    echo "'paramiko' is installed."
else
    echo "Warning: 'paramiko' library not found."
    read -p "Attempt to install 'paramiko' using pip? (y/n): " install_confirm
    if [[ "$install_confirm" == [yY] || "$install_confirm" == [yY][eE][sS] ]]; then
        echo "Installing paramiko..."
        pip install paramiko
        if $PYTHON_CMD -c "import paramiko" &> /dev/null; then
            echo "'paramiko' installed successfully."
        else
            echo "Error: Failed to install 'paramiko'. Please install it manually ('pip install paramiko') and rerun this script."
            exit 1
        fi
    else
        echo "Error: 'paramiko' is required. Please install it manually ('pip install paramiko') and rerun this script."
        exit 1
    fi
fi

# 2. Remove old bash script if it exists
echo "[Step 2/4] Cleaning up old scripts..."
if [ -f "$OLD_HOSTFILE_GEN_SCRIPT" ]; then
    echo "Removing old script: $OLD_HOSTFILE_GEN_SCRIPT"
    rm "$OLD_HOSTFILE_GEN_SCRIPT"
    if [ $? -eq 0 ]; then
        echo "Old script removed."
    else
        echo "Warning: Failed to remove $OLD_HOSTFILE_GEN_SCRIPT."
    fi
else
    echo "Old script $OLD_HOSTFILE_GEN_SCRIPT not found, skipping removal."
fi

# 3. Generate hostfile using the Python script
# Pass all arguments given to this script directly to the python script
echo "[Step 3/4] Generating hostfile using $HOSTFILE_GEN_SCRIPT..."
$PYTHON_CMD "$HOSTFILE_GEN_SCRIPT" "$@"

# Check if hostfile generation was successful (basic check: file exists)
if [ $? -ne 0 ]; then
    echo "Error: Hostfile generation script failed. Please check the output above."
    exit 1
fi

if [ ! -f "$HOSTFILE_OUTPUT" ]; then
     # Check if a different hostfile name was passed as argument
    HOSTFILE_ARG=$(echo "$@" | grep -oP -- '--hostfile\s+\K\S+')
    if [ -n "$HOSTFILE_ARG" ] && [ -f "$HOSTFILE_ARG" ]; then
         HOSTFILE_OUTPUT=$HOSTFILE_ARG
         echo "Using generated hostfile: $HOSTFILE_OUTPUT"
    elif [ -f "hostfile.txt" ]; then # Fallback to default if arg parsing failed but default exists
         HOSTFILE_OUTPUT="hostfile.txt"
         echo "Using generated hostfile: $HOSTFILE_OUTPUT"
    else
        echo "Error: Hostfile '$HOSTFILE_OUTPUT' (or specified via --hostfile) was not found after running the script."
        exit 1
    fi
fi

echo "Hostfile '$HOSTFILE_OUTPUT' generated successfully."

# 4. Display the command to launch training
echo "[Step 4/4] Setup complete. To launch training, use a command like this:"
echo "--------------------------------------------------"
echo "deepspeed --hostfile=$HOSTFILE_OUTPUT $TRAIN_SCRIPT \\"
echo "    --model_name_or_path <your_model_identifier> \\"
echo "    --deepspeed \\"
echo "    --deepspeed_config $DS_CONFIG \\"
echo "    # Add other arguments like --num_epochs or --dataset_path if needed"
echo ""
echo "** Remember to replace <your_model_identifier> with the actual model name! **"
echo "--------------------------------------------------"

echo "--- Master Node Setup Finished ---"

exit 0