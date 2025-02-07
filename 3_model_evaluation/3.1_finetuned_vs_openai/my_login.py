#--------------- Hugging Face and Weights & Biases login functions ---------------
# 1) Hugging Face account login 
# !pip install huggingface_hub==0.25.1 -q -U      
from huggingface_hub import login
def login_hf():
    print(f'==> Logging in to Hugging Face...')
    login('your Hugging Face Access Token')  # Replace with your real Hugging Face token

# 2) Weights & Biases account login
# !pip install wandb==0.18.1
import wandb
def login_wandb():
    print(f'==> Logging in to Weights & Biases...')
    wandb.login(key='your WandB API key')  # Replace with your real Weights & Biases API key
