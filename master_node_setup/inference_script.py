import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Configure logging to suppress DeepSpeed INFO/WARNING messages
logging.basicConfig(level=logging.ERROR)
logging.getLogger('deepspeed').setLevel(logging.ERROR)

def get_argument_parser():
    """Set up command-line argument parsing for inference."""
    parser = argparse.ArgumentParser(description="DeepSpeed Distributed Inference Script for LLMs")

    # Model Arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models (e.g. 'gpt2' for 125M parameter model)"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank passed from distributed launcher"
    )

    # Inference Arguments
    parser.add_argument(
        "--input_text",
        type=str,
        default=None,
        help="Text input for generation (use either this or --input_file). Use quotes for text with spaces."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to text file with inputs (one per line)"
    )
    parser.add_argument(
        "--max_length",
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
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.5,
        help="Repetition penalty parameter"
    )



    return parser

def load_model_and_tokenizer(model_path):
    """Load the model and tokenizer specified by the user."""
    # print(f"Loading model: {model_path}")
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token ({tokenizer.pad_token})")

        # Initialize DeepSpeed
        import deepspeed
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # Initialize DeepSpeed Zero3 with optimized config
        model = deepspeed.init_inference(
            model,
            mp_size=1,
            dtype=torch.float16,
            replace_with_kernel_inject=False
        )
        print(f"Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model/tokenizer: # {e}")
        raise

def main():
    parser = get_argument_parser()
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # Skip .to(device) for 8-bit models as they're already on correct device
    print("Model loaded successfully")

    print(f"Starting inference...")
    inputs = []
    if args.input_text:
        inputs = [args.input_text]
    elif args.input_file:
        with open(args.input_file, 'r') as f:
            inputs = [line.strip() for line in f if line.strip()]
    else:
        pass

    if args.input_text or args.input_file:
        # Run once with provided inputs
        print(f"Starting inference...")
        for i, input_text in enumerate(inputs):
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_length,
                temperature=args.temperature,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=4,
                early_stopping=True,
                use_cache=True
            )
            print(f"\nInput ({i+1}/{len(inputs)}): {input_text}")
            print("Generated output:")
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            print("-"*50)
        return
    else:
        # Enhanced interactive mode
        print("\n=== Interactive Inference Mode ===")
        print("Type your prompt and press Enter to generate text.")
        print("Type 'exit' or press Ctrl+C to quit.")
        print("-"*50)
        
        try:
            while True:
                input_text = input("\nPrompt: ")
                if not input_text.strip() or input_text.lower() == 'exit':
                    print("Exiting interactive mode...")
                    return
                    
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_length,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=4,
                    early_stopping=True,
                    use_cache=False
                )
                print("\nGenerated output:")
                print(tokenizer.decode(outputs[0], skip_special_tokens=True))
                print("-"*50)
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            return
    

        

if __name__ == "__main__":
    main()