#--------------- Functions in QLoRA Fine Tuning ---------------
# 1) Create Quantization Config for QLoRA
# !pip install bitsandbytes==0.44.1 -q -U
import torch
from transformers import BitsAndBytesConfig
def create_quantization_config():
    print("==> Creating Quantization Configuration...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                       # Load model weights in 4-bit precision
        bnb_4bit_use_double_quant=True,          # First quantize weights to 8-bit, then to 4-bit
        bnb_4bit_quant_type="nf4",               # Use NF4 (Normal Float 4-bit) quantization
        bnb_4bit_compute_dtype=torch.bfloat16    # Upcast 4-bit precision weights to bfloat16 during computation
    )
    return bnb_config
