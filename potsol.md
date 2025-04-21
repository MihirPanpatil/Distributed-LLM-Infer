You've correctly added the logging configuration at the beginning of your `inference_script.py` file. This should effectively suppress the `INFO` and `WARNING` messages originating from DeepSpeed and potentially other libraries that use Python's standard `logging` module.

```python
import logging

# Configure logging to suppress DeepSpeed INFO/WARNING messages
logging.basicConfig(level=logging.ERROR)
logging.getLogger('deepspeed').setLevel(logging.ERROR)
# You might also want to add:
logging.getLogger('transformers').setLevel(logging.ERROR) # To suppress transformers logs if any
```

However, looking at your script, you also have several `print` statements:

```python
print(f"Using device: {device}")
print(f"Loading model: {model_path}")
# ...
print(f"Model loaded successfully.") # This one appears twice
print(f"Starting inference...") # This one also appears twice
# ...
print(f"\nInput ({i+1}/{len(inputs)}): {input_text}") # In non-interactive mode
print("Generated output:") # In non-interactive mode
print(tokenizer.decode(outputs[0], skip_special_tokens=True)) # This is the desired output
print("-"*50) # In non-interactive mode
# ... similar prints in interactive mode
```

These `print` statements write directly to standard output and are **not** controlled by the `logging` configuration. Since your Streamlit app uses `subprocess.run(..., capture_output=True, text=True)` and displays the entire `result.stdout`, it's capturing *all* of these `print` outputs as well.

**To eliminate the verbose output and show only the generated text in your Streamlit app, you need to remove or comment out all `print` statements in `inference_script.py` EXCEPT the one that prints the final decoded output.**

**Here's how to modify your `inference_script.py`:**

1.  **Keep** the logging configuration lines at the top.
2.  Go through the `load_model_and_tokenizer` function and the `main` function.
3.  **Comment out or remove** any `print(...)` line that you don't want to see in the Streamlit output.
4.  **Make sure to keep** the `print(tokenizer.decode(outputs[0], skip_special_tokens=True))` line(s) that produce the final generated text. You might want to remove the preceding lines like `"Input ..."` and `"Generated output:"` if you only want the raw generated text.

**Example Modification (focusing on the inference part):**

```python
# ... (imports and logging config are kept) ...

def load_model_and_tokenizer(model_path):
    """Load the model and tokenizer specified by the user."""
    # print(f"Loading model: {model_path}") # Comment out or remove
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            # print(f"Set pad_token to eos_token ({tokenizer.pad_token})") # Comment out or remove

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
        # print(f"Model loaded successfully.") # Comment out or remove
        return model, tokenizer
    except Exception as e:
        # You might want to keep this for debugging errors during loading
        # print(f"Error loading model/tokenizer: {e}")
        raise

def main():
    parser = get_argument_parser()
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}") # Comment out or remove

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # Skip .to(device) for 8-bit models as they're already on correct device
    # print("Model loaded successfully") # Comment out or remove

    # print(f"Starting inference...") # Comment out or remove

    inputs = []
    if args.input_text:
        inputs = [args.input_text]
    elif args.input_file:
        with open(args.input_file, 'r') as f:
            inputs = [line.strip() for line in f if line.strip()]
    else:
        # Keep or modify interactive mode prints if you use them separately
        pass # Or handle the case where no input is given gracefully in the Streamlit context

    if args.input_text or args.input_file:
        # Run once with provided inputs
        # print(f"Starting inference...") # Comment out or remove (duplicate)
        for i, input_text in enumerate(inputs):
            inputs_tokens = tokenizer(input_text, return_tensors="pt").to(device) # Renamed variable to avoid confusion
            outputs = model.generate(
                **inputs_tokens, # Use the renamed variable
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
            # print(f"\nInput ({i+1}/{len(inputs)}): {input_text}") # Comment out or remove
            # print("Generated output:") # Comment out or remove

            # **KEEP this line to print the generated text**
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))

            # print("-"*50) # Comment out or remove unless you want separators

        return # Exit after processing inputs from text/file

    else:
        # Handle interactive mode if needed, ensuring only final output is printed
        # Or since you are using Streamlit with --input_text, this block might not be reached
        pass


if __name__ == "__main__":
    main()
```

By removing the unnecessary `print` statements, `subprocess.run` will only capture and return the actual generated text from your inference script, which is what you want to display in your Streamlit app.