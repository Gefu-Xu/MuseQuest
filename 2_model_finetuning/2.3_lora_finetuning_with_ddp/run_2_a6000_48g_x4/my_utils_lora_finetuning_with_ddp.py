#--------------- Functions in LoRA Fine-Tuning with Distributed Data Parallel ---------------
# 1) Override the built-in print() function to prepend a prefix string to all printed messages.
import builtins
def add_prefix_to_print_function(prefix_string):
    original_print = builtins.print                               # Save a reference to the original print function
    def custom_print(*args, **kwargs):                            # Define a custom print function
        args = (f"{prefix_string} {' '.join(map(str, args))}",)   # Prepend the prefix string to the original print message
        original_print(*args, **kwargs)                           # Call the original print function with the modified arguments
    builtins.print = custom_print                                 # Override the built-in print function

# 2) Create Custom Device Map (Multi-GPU) for Distributed Data Parallel
def create_device_map_ddp_multi_gpu(accelerator):
    rank = accelerator.process_index                              # Get the current rank (process) ID from the accelerator
    device_map = {'': rank}                                       # Allocate all model layers to the GPU matching the current rank ID
    return device_map

# 3) Create Custom Device Map (Single-GPU)
def create_device_map_ddp_single_gpu0():   
    device_map = {'': 0}                                          # Allocate all model layers to the single GPU0
    return device_map

# 4) Define a custom callback function to report VRAM usage, print the mini-batch size, and print the first data in the mini-batch.
# Note: At every training step, each GPU receives an individual mini-batch from the dataloader in DDP. Printing the mini-batch size and the first data in the mini-batch at training step 1 helps validate that each GPU processes a different batch from the dataset, ensuring proper data parallelism.
import torch
from transformers import TrainerCallback
class CustomCallbackDDP(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # If the current process is using a GPU and this is the first training step:
        if torch.cuda.is_available() and state.global_step == 1:
            # Report VRAM usage
            gpu_id = torch.cuda.current_device()
            gpu_memory = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
            gpu_max_memory = torch.cuda.max_memory_allocated(gpu_id) / (1024 ** 3)
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 3)
            print(f'Step {state.global_step}: Current VRAM usage: {gpu_memory:.2f} GB, Peak VRAM usage: {gpu_max_memory:.2f} GB, Total GPU VRAM: {total_memory:.2f} GB')

            # Print mini-batch size and the first data in the mini-batch
            train_dataloader = kwargs.get('train_dataloader')     # Retrieve the dataloader from kwargs
            if train_dataloader is not None:
                for batch in train_dataloader:
                    input_data = batch.get('input_ids', None)
                    if input_data is not None:
                        batch_size = input_data.size(0)           # Get the size of the mini-batch
                        first_input_data = input_data[0].tolist() # Convert the first data in the mini-batch to a list
                        print(f'Step {state.global_step}: Mini-batch size: {batch_size}')
                        print(f'Step {state.global_step}: 1st data in mini-batch: {first_input_data}')
                    else:
                        print(f'Step {state.global_step}: No input data available in the mini-batch.')
                    break                                         # Stop after processing the first mini-batch to avoid iterating through the entire dataloader
