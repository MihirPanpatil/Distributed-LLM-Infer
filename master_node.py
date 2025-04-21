import grpc
import logging
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed
import inference_pb2
import inference_pb2_grpc

logger = logging.getLogger(__name__)

class DistributedInferenceManager:
    def __init__(self, model_name: str, reload: bool = True, cpu_offload: bool = False, selected_workers: Optional[List[int]] = None):
        self.model_name = model_name
        self.reload = reload
        self.cpu_offload = cpu_offload
        self.selected_workers = selected_workers or []
        self.model_loaded = False
        self.model = None
        self.tokenizer = None
        
    def initialize_model(self):
        """Initialize the model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            config = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
            
            self.model = deepspeed.init_inference(
                config,
                mp_size=1,
                dtype=torch.float16,
                replace_with_kernel_inject=False
            )
            
            self.model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.7, 
                 top_p: float = 0.9, top_k: int = 50) -> str:
        """Generate text from the given prompt using distributed inference."""
        if not self.model_loaded or not self.model:
            raise RuntimeError("Model not initialized")
            
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise
    
    def close(self):
        """Clean up resources."""
        if self.model:
            del self.model
            self.model = None
        self.model_loaded = False
        logger.info("Model resources released")