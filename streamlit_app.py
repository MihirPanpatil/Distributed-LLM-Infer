import streamlit as st
import subprocess
import json
from pathlib import Path

# Initialize the inference script
@st.cache_resource
def get_inference_script_path():
    return str(Path(__file__).parent / "master_node_setup" / "inference_script.py")

script_path = get_inference_script_path()

# Streamlit UI
st.title("LLM Inference Dashboard")

# Input area
prompt = st.text_area("Enter your prompt:", height=150)

# Model selection
model_name = st.selectbox("Model", ["facebook/opt-350m", "gpt2"])

# Generation parameters
col1, col2, col3 = st.columns(3)
with col1:
    max_length = st.slider("Max length", 50, 500, 100)
with col2:
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
with col3:
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9)

# Generate button
if st.button("Generate"):
    if prompt:
        with st.spinner("Generating..."):
            try:
                # Run inference script with parameters
                cmd = [
                    "python", script_path,
                    "--model_path", model_name,
                    "--input_text", prompt,
                    "--max_length", str(max_length),
                    "--temperature", str(temperature),
                    "--top_p", str(top_p)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                output = result.stdout
                st.success("Generation complete!")
                st.text_area("Output:", value=output, height=300)
            except Exception as e:
                st.error(f"Error during generation: {str(e)}")
    else:
        st.warning("Please enter a prompt first.")