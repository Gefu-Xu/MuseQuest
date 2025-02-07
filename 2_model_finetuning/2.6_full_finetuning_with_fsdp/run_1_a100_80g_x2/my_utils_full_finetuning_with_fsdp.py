#--------------- Functions in Full Fine-Tuning with Fully Sharded Data Parallel ---------------
# 1) Override the built-in print() function to prepend a prefix string to all printed messages.
import builtins
def add_prefix_to_print_function(prefix_string):
    original_print = builtins.print                               # Save a reference to the original print function
    def custom_print(*args, **kwargs):                            # Define a custom print function
        args = (f"{prefix_string} {' '.join(map(str, args))}",)   # Prepend the prefix string to the original print message
        original_print(*args, **kwargs)                           # Call the original print function with the modified arguments
    builtins.print = custom_print                                 # Override the built-in print function

# 2) Print utility accelerator info to the terminal
def print_util_accelerator_info(accelerator):
    print(f'==> Printing utility accelerator info...')
    print(f'==> accelerator.distributed_type: {accelerator.distributed_type}')                                 # Current distributed training mode: FSDP
    print(f'==> accelerator.state.num_processes: {accelerator.state.num_processes}')                           # Total number of processes across all devices
    print(f'==> accelerator.state.process_index: {accelerator.state.process_index}')                           # Global index of the current process among all device processes
    print(f'==> accelerator.state.local_process_index: {accelerator.state.local_process_index}')               # Local index of the current process on its assigned device
    print(f'==> accelerator.state.device: {accelerator.state.device}')                                         # Device assigned to the current process
    print(f'==> accelerator.state.mixed_precision: {accelerator.state.mixed_precision}')                       # Precision mode used for training (e.g., FP16, BF16, FP32)
    if hasattr(accelerator.state, 'fsdp_plugin'):                                                              # Details of the FSDP plugin, if configured
        print(f'==> accelerator.state.fsdp_plugin: {accelerator.state.fsdp_plugin.__dict__}')
    else:
        print(f'==> accelerator.state.fsdp_plugin: Not Specified')

# 3) Print trainer info to the terminal
def print_trainer_info(trainer):
    print(f'==> Printing trainer info...')
    # Print trainer.model 
    print(f'==> trainer.model is located on device: {next(trainer.model.parameters()).device}')                # Device where the model is located
    # Print trainer.accelerator internal state 
    accelerator = trainer.accelerator       
    print(f'==> trainer.accelerator.distributed_type: {accelerator.distributed_type}')                         # Current distributed training mode: FSDP
    print(f'==> trainer.accelerator.state.num_processes: {accelerator.state.num_processes}')                   # Total number of processes across all devices
    print(f'==> trainer.accelerator.state.process_index: {accelerator.state.process_index}')                   # Global index of the current process among all device processes
    print(f'==> trainer.accelerator.state.local_process_index: {accelerator.state.local_process_index}')       # Local index of the current process on its assigned device
    print(f'==> trainer.accelerator.state.device: {accelerator.state.device}')                                 # device assigned to the current process
    print(f'==> trainer.accelerator.state.mixed_precision: {accelerator.state.mixed_precision}')               # Precision mode used for training (e.g., FP16, BF16, FP32)
    if hasattr(accelerator.state, 'fsdp_plugin'):                                                              # Details of the FSDP plugin, if configured
        print(f'==> trainer.accelerator.state.fsdp_plugin: {accelerator.state.fsdp_plugin.__dict__}')
    else:
        print(f'==> trainer.accelerator.state.fsdp_plugin: Not Specified')
    # Print trainer.train_dataloader
    train_dataloader = trainer.get_train_dataloader()
    print(f'==> trainer.train_dataloader: {train_dataloader}')                                                  # The dataloader used for training
    print(f'==> trainer.train_dataloader._is_accelerate_prepared: {train_dataloader._is_accelerate_prepared}')  # Whether the dataloader is prepared by accelerate
    print(f'==> trainer.train_dataloader.device: {train_dataloader.device}')                                    # Device assigned to the training dataloader
    # Print trainer.eval_dataloader
    eval_dataloader = trainer.get_eval_dataloader()
    print(f'==> trainer.eval_dataloader: {eval_dataloader}')                                                    # The dataloader used for evaluation
    print(f'==> trainer.eval_dataloader._is_accelerate_prepared: {eval_dataloader._is_accelerate_prepared}')    # Whether the dataloader is prepared by accelerate
    print(f'==> trainer.eval_dataloader.device: {eval_dataloader.device}')                                      # Device assigned to the evaluation dataloader
    # Print trainer.optimizer
    if trainer.optimizer:                                                                                       # Details of the optimizer, if defined
        optimizer = trainer.optimizer
        print(f'==> trainer.optimizer type: {type(optimizer)}')                                                 # Type of optimizer used
        print(f"==> trainer.optimizer.param_groups.initial_lr: {optimizer.param_groups[0]['initial_lr']}")      # Initial learning rate of the optimizer
        print(f'==> trainer.optimizer._is_accelerate_prepared: {optimizer._is_accelerate_prepared}')            # Whether the optimizer is prepared by accelerate
        print(f"==> trainer.optimizer on device: {optimizer.param_groups[0]['params'][0].device}")              # Device where the optimizer parameters are located
    else:
        print(f'==> trainer.optimizer is not available.') 

# 4) Print brief model wrapping status to the terminal and save detailed wrapping status to a file
# Note: This function is highly customized for the Idefics2 model
import os
import torch.distributed.fsdp as fsdp
def print_module_wrapping_status_to_file(model, filename):
    print(f'==> Printing module wrapping status...')
    filepath = os.path.dirname(filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    file = open(filename, 'w')

    # Define class names considered as module groups
    module_group_class_name = {
        'Idefics2ForConditionalGeneration', 
        'Idefics2Model', 
        'Idefics2VisionTransformer', 
        'Idefics2VisionEmbeddings',
        'Idefics2Encoder',
        'Idefics2EncoderLayer', 
        'Idefics2Connector', 
        'Idefics2PerceiverResampler',
        'ModuleList', 
        'MistralModel',
        'Idefics2PerceiverLayer', 
        'MistralDecoderLayer'
    }

    wrapped_module = 0
    unwrapped_module = 0
    # Iterate through model modules and categorize them
    for name, module in model.named_modules():
        if name == '':                                                                # Top-level module
            file.write(f'top_module:         {name} ({type(module).__name__})\n')
        elif type(module).__name__ in module_group_class_name:                        # Module group
            file.write(f'module group:       {name} ({type(module).__name__})\n')
        elif '._fsdp_wrapped_module' in name:                                         # FSDP-wrapped submodule
            file.write(f'                    {name}  ({type(module).__name__})\n')
        else:                                                                         # Regular module
            if isinstance(module, fsdp.FullyShardedDataParallel):                     # FSDP-wrapped module
                wrapped_module += 1
                file.write(f'--fsdp_wrap_module: {name} ({type(module).__name__})\n')
            else:                                                                     # Unwrapped module
                unwrapped_module += 1
                file.write(f'--unwrap-module:    {name} ({type(module).__name__})\n')

    # Calculate total modules and percentage of wrapped modules
    total_module = wrapped_module + unwrapped_module
    wrapped_percentage = (100 * wrapped_module / total_module) if total_module > 0 else 0
    print(f'==> Total modules: {total_module}, FSDP wrapped modules: {wrapped_percentage:.2f}%')
    print(f'==> For complete module wrapping status info, refer to {filename}')
    file.write(f'==> Total modules: {total_module}, FSDP wrapped modules: {wrapped_percentage:.2f}%\n')
    file.close()

# 5) Select Layers for Full Fine-Tuning
def select_layers_for_full_fine_tuning(model, module_list):
    print("==> Selecting Layers for Full Fine-Tuning...")
    for name, param in model.named_parameters():
        param.requires_grad = False                                                   # Freeze all parameters by default
        for module in module_list:
            if module in name and 'proj' in name:                                     # Set linear('proj') layers in the specified modules to trainable
                param.requires_grad = True
                break
    return model

# 6) Define a custom callback function to report VRAM usage, print mini-batch details, and display parameter sharding percentage on the current device.
# Note 1: At every training step, each GPU receives an individual mini-batch from the dataloader in FSDP. Printing the mini-batch size and the first data in the mini-batch at training step 1 helps validate that each GPU processes a different batch from the dataset, ensuring proper data parallelism.
# Note 2: At training step 1, printing the trainer's model type validates whether the model is FSDP-wrapped. Additionally, printing the percentage of parameters sharded on the current device validates that layer sharding is evenly distributed across GPUs.
import torch
from transformers import TrainerCallback
class CustomCallbackFSDP(TrainerCallback):
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
            train_dataloader = kwargs.get('train_dataloader')         # Retrieve the dataloader from kwargs
            if train_dataloader is not None:
                for batch in train_dataloader:
                    input_data = batch.get('input_ids', None)
                    if input_data is not None:
                        batch_size = input_data.size(0)               # Get the size of the mini-batch
                        first_input_data = input_data[0].tolist()     # Convert the first data in the mini-batch to a list
                        print(f'Step {state.global_step}: Mini-batch size: {batch_size}')
                        print(f'Step {state.global_step}: 1st data in mini-batch: {first_input_data}')
                    else:
                        print(f'Step {state.global_step}: No input data available in the mini-batch.')
                    break                                             # Stop after processing the first mini-batch to avoid iterating through the entire dataloader
            
            # Print model type to check if wrapped in FSDP
            model = kwargs.get('model')
            print(f'Step {state.global_step}: Trainer model type: {type(model)}')

            # For the first and last 15 model layers, print the percentage of parameters sharded on the current device
            display_size = 15
            named_params = list(model.named_parameters())
            layers_to_print = named_params[:display_size] + named_params[-display_size:]
            for name, param in layers_to_print:
                if hasattr(param, '_unpadded_unsharded_size'):        # FSDP mode; this attribute exists only in FSDP
                    total_params = param._unpadded_unsharded_size[0]  # In FSDP, param._unpadded_unsharded_size[0] is the total parameter count
                    shard_params = param.numel()                      # In FSDP, param.numel() and param._sharded_size[0] represent the sharded parameter count on the current device
                    shard_percentage = (shard_params / total_params) * 100 if total_params > 0 else 0
                    print(f'Step {state.global_step}: [FSDP Mode] Model Layer: {name}, Total Parameter Count: {total_params}, Sharded Parameter Count on device: {shard_params} ({shard_percentage:.2f}%)')
                else:
                    print(f'Step {state.global_step}: [Non-FSDP Mode] Model Layer: {name}, Total Parameter Count: {param.numel()}')  # In non-FSDP, param.numel() is the total parameter count
