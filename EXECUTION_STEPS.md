# Distributed Training Execution Steps

## 1. Setup All Nodes
```bash
# On all nodes (master and workers):
pip install deepspeed torch transformers
```

## 2. Prepare Worker Nodes
```bash
# On each worker node:
chmod +x ~/worker_node_setup/detect_gpus.sh
```

## 3. Run Master Setup
```bash
# On master node:
cd master_node_setup
chmod +x run_master_setup.sh
./run_master_setup.sh \
    --nodes worker1 worker2 \ 
    --script-path ~/worker_node_setup/detect_gpus.sh \
    --ssh-user your_username
```

## 4. Launch Training
```bash
deepspeed --hostfile=hostfile.txt train_script.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --num_epochs 3 \
    --deepspeed \
    --deepspeed_config ds_config.json
```

## 5. Monitor Training
- Check master node terminal for progress
- View worker node logs:
```bash
tail -f /var/log/syslog | grep deepspeed
```

## Key Notes:
- System automatically handles mixed GPU/CPU nodes
- CPU nodes will use optimized configurations
- Training scales based on available resources
- Adjust batch sizes in ds_config.json if needed