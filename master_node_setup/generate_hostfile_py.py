import argparse
import paramiko
import os
import time
import getpass
import subprocess
import socket

# --- Configuration ---
# List of worker node hostnames or IP addresses
# Replace these placeholders with the actual hostnames or IPs of your worker nodes.
DEFAULT_NODES = ["10.255.255.254"]

# Path to the detect_gpus.sh script on the worker nodes
# IMPORTANT: This should be the *absolute path* or path relative to the user's
# home directory on the worker node where the script will be executed.
DEFAULT_DETECT_SCRIPT_PATH = "~/worker_node_setup/detect_gpus.sh" # Example: Assumes it's in worker's home

# Output hostfile name
DEFAULT_HOSTFILE = "hostfile.txt"

# SSH Configuration
DEFAULT_SSH_USER = getpass.getuser() # Use current user by default
DEFAULT_SSH_PORT = 22
DEFAULT_SSH_KEY_FILE = os.path.expanduser("~/.ssh/id_rsa") # Common default private key

# --- End Configuration ---

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate DeepSpeed hostfile by querying worker nodes via SSH.")
    parser.add_argument("--nodes", nargs='+', default=DEFAULT_NODES,
                        help=f"List of worker node hostnames or IPs. Default: {DEFAULT_NODES}")
    parser.add_argument("--script-path", type=str, default=DEFAULT_DETECT_SCRIPT_PATH,
                        help=f"Path to detect_gpus.sh on worker nodes. Default: {DEFAULT_DETECT_SCRIPT_PATH}")
    parser.add_argument("--hostfile", type=str, default=DEFAULT_HOSTFILE,
                        help=f"Output hostfile name. Default: {DEFAULT_HOSTFILE}")
    parser.add_argument("--ssh-user", type=str, default=DEFAULT_SSH_USER,
                        help=f"SSH username for worker nodes. Default: {DEFAULT_SSH_USER}")
    parser.add_argument("--ssh-port", type=int, default=DEFAULT_SSH_PORT,
                        help=f"SSH port for worker nodes. Default: {DEFAULT_SSH_PORT}")
    parser.add_argument("--ssh-key", type=str, default=DEFAULT_SSH_KEY_FILE,
                        help=f"Path to SSH private key file. Default: {DEFAULT_SSH_KEY_FILE}")
    parser.add_argument("--ssh-password", action='store_true',
                        help="Prompt for SSH password instead of using key file.")
    parser.add_argument("--include-master", action='store_true',
                        help="Detect GPUs on the local master node and include it in the hostfile.")
    return parser.parse_args()

def execute_remote_command(hostname, port, username, command, key_filename=None, password=None):
    """Connects to a remote host via SSH and executes a command."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) # Automatically add host keys
    output = None
    error = None
    try:
        print(f"Connecting to {username}@{hostname}:{port}...")
        if password:
            client.connect(hostname, port=port, username=username, password=password, timeout=10)
        else:
            client.connect(hostname, port=port, username=username, key_filename=key_filename, timeout=10)

        print(f"Executing command: {command}")
        stdin, stdout, stderr = client.exec_command(command, timeout=15)
        output = stdout.read().decode('utf-8').strip()
        error = stderr.read().decode('utf-8').strip()
        exit_status = stdout.channel.recv_exit_status()
        print(f"Command exit status: {exit_status}")

        if exit_status != 0:
            error = f"Command exited with status {exit_status}. Stderr: {error}"
            output = None # Don't trust output if command failed

    except paramiko.AuthenticationException:
        error = "Authentication failed. Check username, password, or key file."
    except paramiko.SSHException as ssh_ex:
        error = f"SSH connection error: {ssh_ex}"
    except Exception as e:
        error = f"An unexpected error occurred: {e}"
    finally:
        client.close()
        print(f"Connection to {hostname} closed.")

    return output, error

def detect_local_gpus():
    """Detects GPUs on the local machine."""
    print("\n--- Detecting GPUs on local master node ---")
    gpu_count = 0
    gpu_type = "None"
    hostname = socket.gethostname() # Get the local hostname

    try:
        # Check for nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            gpu_count = len(result.stdout.strip().split('\n'))
            gpu_type = "NVIDIA"
            print(f"Detected {gpu_count} NVIDIA GPUs.")
        else:
            # Check for rocminfo
            result = subprocess.run(['rocminfo'], capture_output=True, text=True, check=False)
            if result.returncode == 0 and 'GPU' in result.stdout:
                 # Counting lines containing 'Agent' and 'Name:' might be more robust for rocminfo structure
                 # Example: Count lines like '  Name:                    gfx...' under 'Agent xxx.' sections
                 # This is a simpler approximation:
                 gpu_count = result.stdout.count('SubVendor ID:') # Count occurrences which often correspond to GPUs
                 if gpu_count == 0: # Fallback if SubVendor ID isn't found reliably
                     gpu_count = result.stdout.count('Marketing Name:')
                 gpu_type = "AMD"
                 print(f"Detected {gpu_count} AMD GPUs (using rocminfo).")

            else:
                # Check for clinfo (less reliable for discrete GPUs sometimes)
                result = subprocess.run(['clinfo'], capture_output=True, text=True, check=False)
                if result.returncode == 0 and 'Device Name' in result.stdout:
                    # Count devices that are likely GPUs (heuristic)
                    gpu_devices = [line for line in result.stdout.split('\n') if 'Device Type' in line and 'GPU' in line]
                    gpu_count = len(gpu_devices)
                    gpu_type = "OpenCL"
                    print(f"Detected {gpu_count} OpenCL GPUs (using clinfo).")
                else:
                    print("No GPUs detected via nvidia-smi, rocminfo, or clinfo.")

    except FileNotFoundError as e:
        print(f"GPU detection command not found: {e.filename}. Assuming 0 GPUs.")
        gpu_count = 0
        gpu_type = "None"
    except Exception as e:
        print(f"Error during local GPU detection: {e}. Assuming 0 GPUs.")
        gpu_count = 0
        gpu_type = "None"

    cpu_cores = os.cpu_count()
    if gpu_count > 0:
        return f"{hostname} gpu_slots={gpu_count} gpu_type={gpu_type} cpu_slots={cpu_cores}"
    else:
        print(f"No GPUs detected, using {cpu_cores} CPU cores")
        return f"{hostname} gpu_slots=0 gpu_type=None cpu_slots={cpu_cores}"

def main():
    args = parse_arguments()

    ssh_password = None
    ssh_key = args.ssh_key
    if args.ssh_password:
        ssh_password = getpass.getpass(f"Enter SSH password for user {args.ssh_user}: ")
        ssh_key = None # Don't use key if password is provided
    elif not os.path.exists(ssh_key):
        print(f"Warning: SSH key file '{ssh_key}' not found. Authentication might fail.")
        # Optionally prompt for password here or let connection fail
        # ssh_password = getpass.getpass(f"SSH key not found. Enter password for {args.ssh_user}: ")
        # ssh_key = None

    print(f"Generating hostfile: {args.hostfile}")
    print(f"Querying nodes: {args.nodes}")
    print(f"Worker script path: {args.script_path}")
    print(f"SSH User: {args.ssh_user}, Port: {args.ssh_port}")
    if ssh_key:
        print(f"SSH Key: {ssh_key}")

    hostfile_content = []
    failed_nodes = []
    master_info = None
    if args.include_master:
        master_info = detect_local_gpus()
        if master_info:
            hostfile_content.append(master_info)
        else:
            print("Master node specified to be included, but no GPUs detected locally.")

    # Prepare command for worker nodes
    # Ensure the detect script is executable (needs to be run on workers)
    # The command assumes 'bash' is available on the worker nodes.
    command_to_run = f"bash {args.script_path}"

    for node in args.nodes:
        print(f"\n--- Querying node: {node} ---")
        gpu_info, err = execute_remote_command(
            hostname=node,
            port=args.ssh_port,
            username=args.ssh_user,
            command=command_to_run,
            key_filename=ssh_key,
            password=ssh_password
        )

        if gpu_info:
            print(f"Detected info for {node}: {gpu_info}")
            # Parse GPU and CPU slots
            gpu_slots = 0
            cpu_slots = 0
            if "gpu_slots=" in gpu_info:
                gpu_slots = int(gpu_info.split("gpu_slots=")[1].split()[0])
            if "cpu_slots=" in gpu_info:
                cpu_slots = int(gpu_info.split("cpu_slots=")[1].split()[0])
            
            # Use GPU slots if available, otherwise CPU slots
            slots = gpu_slots if gpu_slots > 0 else cpu_slots
            if slots > 0:
                hostfile_content.append(f"{node} slots={slots}")
            else:
                print(f"Warning: No usable slots found for {node}")
                failed_nodes.append(node)
        else:
            print(f"Warning: Failed to get GPU info from {node}. Error: {err}")
            failed_nodes.append(node)
            # Optionally add a default entry or skip entirely
            # hostfile_content.append(f"{node} slots=0 type=Unknown # Failed query")

    # Write the hostfile
    try:
        with open(args.hostfile, 'w') as f:
            for line in hostfile_content:
                f.write(line + '\n')
        print(f"\nHostfile '{args.hostfile}' generated successfully.")
    except IOError as e:
        print(f"Error writing hostfile '{args.hostfile}': {e}")
        return # Exit main if cannot write file

    print("\n--- Generated Hostfile Content ---")
    try:
        with open(args.hostfile, 'r') as f:
            print(f.read().strip())
    except IOError:
        print(f"Could not read back hostfile '{args.hostfile}'.")
    print("---------------------------------")

    if failed_nodes:
        print("\nWarning: Failed to retrieve information from the following nodes:")
        for node in failed_nodes:
            print(f"- {node}")
        print("These nodes were excluded from the hostfile.")

    print("\nRemember to:")
    print(f"1. Ensure worker nodes ({', '.join(args.nodes)}) are reachable via SSH from this machine.")
    print(f"2. Verify SSH credentials (user: {args.ssh_user}, key: {ssh_key or 'password used'}) are correct.")
    print(f"3. Ensure the script '{args.script_path}' exists and is executable on all worker nodes (if any specified).")
    print(f"4. Install 'paramiko' library on this master node: pip install paramiko")
    if args.include_master:
        print(f"5. Ensure GPU detection tools (nvidia-smi/rocminfo/clinfo) are installed on the master node for local detection.")

if __name__ == "__main__":
    main()