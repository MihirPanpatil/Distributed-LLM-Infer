import argparse
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import time

# --- Helper Functions ---
def get_argument_parser():
    """Set up command-line argument parsing."""
    parser = argparse.ArgumentParser(description="DeepSpeed Distributed Training Script for LLMs")

    # Model Arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models. "
             "Examples: 'meta-llama/Llama-2-7b-hf', 'mistralai/Mistral-7B-v0.1', 'tiiuae/falcon-7b', "
             "'facebook/opt-1.3b', 'bigscience/bloom-560m', 'EleutherAI/gpt-neox-20b', "
             "'Qwen/Qwen-7B', 'google/gemma-7b'"
    )

    # Inference Arguments
    parser.add_argument(
        "--input_text",
        type=str,
        default=None,
        help="Text input for generation (use either this or --input_file)"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to text file with inputs (one per line)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation"
    )

    # Data Arguments (Placeholders - replace with your actual data loading)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the training dataset (optional, replace with actual data handling)."
    )

    # Training Arguments
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum number of training steps. Overrides num_epochs if set > 0."
    )

    # DeepSpeed Arguments
    # Add deepspeed_config argument using deepspeed.add_config_arguments
    parser = deepspeed.add_config_arguments(parser)

    # Add local_rank argument provided by deepspeed launcher
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')

    return parser

def load_model_and_tokenizer(model_name_or_path, local_rank):
    """Load the model and tokenizer specified by the user."""
    print(f"[Rank {local_rank}] Loading model: {model_name_or_path}")
    try:
        # Load configuration first to potentially adjust settings if needed
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        # Example: You might want to force a specific dtype or setting
        # config.torch_dtype = torch.float16 # Or torch.bfloat16 if supported

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        # Set pad token if not present (common for some models like Llama)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"[Rank {local_rank}] Set pad_token to eos_token ({tokenizer.pad_token})")

        # Load model - DeepSpeed handles device placement later
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=config,
            trust_remote_code=True
            # Note: Avoid device_map='auto' here; DeepSpeed manages placement.
            # torch_dtype can be set here or rely on DeepSpeed fp16/bf16 config
        )
        print(f"[Rank {local_rank}] Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"[Rank {local_rank}] Error loading model/tokenizer: {e}")
        raise

def prepare_dataset(tokenizer, dataset_path):
    """Placeholder for dataset loading and preprocessing."""
    # Replace this with your actual dataset loading logic
    # Example: Load from a file, preprocess, tokenize
    print("Loading dummy dataset...")
    # Example dummy data
    texts = ["DeepSpeed is a library for training large models.",
             "Distributed training allows scaling across multiple GPUs and nodes.",
             "Choose your LLM: LLaMA, Mistral, Falcon, OPT, BLOOM, GPT-NeoX, Qwen, Gemma."] * 10
    # Tokenize the dummy data
    # Ensure padding and truncation are handled appropriately for your model/task
    tokenized_data = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # Create a simple TensorDataset
    dataset = torch.utils.data.TensorDataset(tokenized_data['input_ids'], tokenized_data['attention_mask'])
    print("Dummy dataset prepared.")
    return dataset

# --- Main Training Logic ---
def main():
    parser = get_argument_parser()
    args = parser.parse_args()

    # Initialize DeepSpeed
    # DeepSpeed requires the local_rank argument
    print(f"Initializing DeepSpeed on local rank: {args.local_rank}")
    deepspeed.init_distributed()
    
    # Check if we're doing inference
    is_inference = not any([args.num_epochs > 0, args.max_steps > 0])

    # Ensure local_rank is correctly set for subsequent operations
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank)) # Get rank from env if available
    
    # Handle device assignment
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"[Rank {local_rank}] Using GPU {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        cpu_cores = os.cpu_count()
        print(f"[Rank {local_rank}] No GPU available, using {cpu_cores} CPU cores")
        
        # Adjust batch size for CPU if needed
        if hasattr(args, 'micro_batch_size'):
            args.micro_batch_size = max(1, args.micro_batch_size // 4)  # Reduce batch size for CPU

    print(f"[Rank {local_rank}] DeepSpeed distributed initialized.")
    print(f"[Rank {local_rank}] DeepSpeed Config Path: {args.deepspeed_config}")

    # Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, local_rank)

    # Prepare Dataset (Replace with your actual data loading)
    train_dataset = prepare_dataset(tokenizer, args.dataset_path)
    # DeepSpeed handles the distributed sampler
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=...) # DeepSpeed handles batch size

    # Initialize DeepSpeed Engine
    print(f"[Rank {local_rank}] Initializing DeepSpeed engine...")
    if is_inference:
        # For inference, we don't need optimizer or training data
        model_engine = deepspeed.init_inference(
            model=model,
            config=args.deepspeed_config
        )
    else:
        # For training, use the full initialize function
        model_engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=model.parameters(),
            training_data=train_dataset
        )
    print(f"[Rank {local_rank}] DeepSpeed engine initialized.")
    print(f"[Rank {local_rank}] Effective Batch Size per GPU: {model_engine.train_micro_batch_size_per_gpu()}")
    print(f"[Rank {local_rank}] Gradient Accumulation Steps: {model_engine.gradient_accumulation_steps()}")

    # --- Training/Inference Loop ---
    if is_inference:
        print(f"[Rank {local_rank}] Starting inference...")
        # Dynamic input handling
        inputs = []
        if args.input_text:
            inputs = [args.input_text]
        elif args.input_file:
            with open(args.input_file, 'r') as f:
                inputs = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError("Must provide either --input_text or --input_file for inference")

        # Batch generation
        for i, input_text in enumerate(inputs):
            inputs = tokenizer(input_text, return_tensors="pt").to(model_engine.local_rank)
            outputs = model_engine.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                pad_token_id=tokenizer.eos_token_id
            )
            if local_rank == 0:
                print(f"\nInput ({i+1}/{len(inputs)}): {input_text}")
                print("Generated output:")
                print(tokenizer.decode(outputs[0], skip_special_tokens=True))
                print("-"*50)
    else:
        print(f"[Rank {local_rank}] Starting training loop...")
        global_step = 0
        start_time = time.time()

        for epoch in range(args.num_epochs):
            print(f"[Rank {local_rank}] Starting Epoch {epoch+1}/{args.num_epochs}")
            model_engine.train() # Set model to training mode

            for step, batch in enumerate(train_loader):
                # Move batch to the correct device
                # Input IDs and Attention Mask are typical outputs from tokenizer
                input_ids = batch[0].to(model_engine.local_rank)
            attention_mask = batch[1].to(model_engine.local_rank)
            # Labels are often input_ids shifted for Causal LM tasks
            labels = input_ids.clone()

            # Forward pass
            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss # The model itself calculates the loss when labels are provided

            # Backward pass (handled by DeepSpeed)
            model_engine.backward(loss)

            # Optimizer step (handled by DeepSpeed)
            model_engine.step()

            global_step += 1

            # Print progress periodically
            if global_step % model_engine.steps_per_print() == 0 and local_rank == 0:
                elapsed_time = time.time() - start_time
                print(f"[Rank {local_rank}] Epoch: {epoch+1}, Step: {global_step}, Loss: {loss.item():.4f}, Time: {elapsed_time:.2f}s")
                # You might want to add learning rate logging here if using a scheduler
                # current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else optimizer.param_groups[0]['lr']
                # print(f"  Learning Rate: {current_lr}")


            # Check if max_steps reached
            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f"[Rank {local_rank}] Reached max_steps ({args.max_steps}). Stopping training.")
                break # Exit inner loop

        if args.max_steps > 0 and global_step >= args.max_steps:
            print(f"[Rank {local_rank}] Reached max_steps ({args.max_steps}). Stopping training.")
             # Exit inner loop

    if args.max_steps > 0 and global_step >= args.max_steps:
        break # Exit outer loop

    print(f"[Rank {local_rank}] Finished Epoch {epoch+1}")

        # --- Optional: Saving Checkpoints ---
        # DeepSpeed handles saving model, optimizer, and scheduler states
        # save_dir = f"checkpoints/epoch_{epoch+1}"
        # if local_rank == 0: # Usually save only on rank 0
        #     print(f"Saving checkpoint to {save_dir}...")
        # model_engine.save_checkpoint(save_dir)
        # print(f"[Rank {local_rank}] Checkpoint saved.")


    total_time = time.time() - start_time
    print(f"[Rank {local_rank}] Training finished.")
    print(f"[Rank {local_rank}] Total Training Time: {total_time:.2f} seconds")

    # --- Optional: Save Final Model ---
    # final_save_dir = "final_model"
    # if local_rank == 0:
    #     print(f"Saving final model to {final_save_dir}...")
    #     # Use model_engine.module to access the underlying Hugging Face model for saving
    #     # Ensure ZeRO stage allows parameter gathering if needed (Stage 3 requires consolidation)
    #     # model_engine.save_fp16_model(final_save_dir) # Example for saving FP16 model
    #     # Or save using Hugging Face methods if parameters are consolidated
    #     # model_engine.module.save_pretrained(final_save_dir)
    #     # tokenizer.save_pretrained(final_save_dir)
    # print(f"[Rank {local_rank}] Final model saving process initiated (if applicable).")


if __name__ == "__main__":
    main()