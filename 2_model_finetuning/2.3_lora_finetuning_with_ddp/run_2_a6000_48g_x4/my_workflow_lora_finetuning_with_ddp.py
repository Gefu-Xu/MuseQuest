#--------------- A Training Workflow in LoRA Fine-Tuning with Distributed Data Parallel ---------------
def training_workflow():
    import my_login
    import my_utils
    import my_utils_lora_finetuning_with_ddp

    # **Step 1: Rank (Process) Initialization**
    ## 1.1 Initialize an Accelerator at the Beginning of Each Rank (Process)
    import torch
    from accelerate import Accelerator
    accelerator = Accelerator()
    rank = accelerator.process_index                  # Get the current rank (process) ID
    torch.cuda.set_device(rank)                       # Bind the current rank (process) to the GPU with the same ID (e.g., rank 0 to GPU 0, rank 1 to GPU 1, etc.)

    ## 1.2 Override print() to Prepend a Prefix String ('[RANK{current rank ID}:GPU{current GPU ID}]') to All Printed Messages
    gpu_id = torch.cuda.current_device()
    prefix_string = f'[RANK{rank}:GPU{gpu_id}]'
    my_utils_lora_finetuning_with_ddp.add_prefix_to_print_function(prefix_string)
    print(f'==> Process started...')

    # **Step 2: Pre-training**
    ## 2.1 Log in to Required Accounts (e.g., Hugging Face, Weights & Biases)
    if accelerator.is_main_process:                                        # Only use rank 0 to log in to avoid multiple logins
        # Log in to the Hugging Face account to access the Idefics2 model and the dataset
        my_login.login_hf()
        # Log in to the Weights & Biases account to enable tracking of training progress
        my_login.login_wandb()
    accelerator.wait_for_everyone()                                        # Synchronize across all ranks to ensure rank 0 finishes

    ## 2.2 Load the Processor (Tokenizer) and the Base Model
    # Load processor
    processor = my_utils.load_processor()
    # Load model
    base_model = my_utils.load_model(device_map=my_utils_lora_finetuning_with_ddp.create_device_map_ddp_multi_gpu(accelerator))  # Use a custom device map: The model processed in the current rank (process) should be allocated to the GPU bound to this rank. Thus, all model layers are allocated to the GPU matching the current rank ID.
    if accelerator.is_main_process:                                        # Only use rank 0 to write files to avoid multi-process file writing
        my_utils.print_processor_info_to_file(processor, './results/model_info/original_processor_info.txt')
        my_utils.print_model_info_to_file(base_model, './results/model_info/base_model_info.txt')
        my_utils.print_model_parameters_to_file(base_model, './results/model_info/base_model_parameters.txt')
    accelerator.wait_for_everyone()                                        # Synchronize across all ranks to ensure rank 0 finishes

    ## 2.3 Load the Dataset
    # Load the Hugging Face dataset
    dataset = my_utils.load_hf_dataset(hf_path='xugefu/MuseQuest')

    ## 2.4 Perform Inference with the Base Model to Establish a Baseline
    if accelerator.is_main_process:                                        # Only use rank 0 to run inference to avoid multi-process file writing
        # Perform batch inference on the test set and save results to a CSV file
        my_utils.batch_inference_model(base_model, processor, dataset['test'], './results/inference_results/inference_results_before_finetuning.csv', 14)
        # Plot similarity scores from the inference CSV file, using thumbnails, and save the plot image locally
        my_utils.plot_similarity_scores([('Original', './results/inference_results/inference_results_before_finetuning.csv')],
                                        thumbnail_dir='./results/inference_results/thumbnail',
                                        plot_file_name='./results/inference_results/similarity_scores_before_finetuning.jpg'
        )
    accelerator.wait_for_everyone()                                        # Synchronize across all ranks to ensure rank 0 finishes

    # **Step 3: Training**
    ## 3.1 Add a Custom Padding Token
    # Set up pad token (<pad>) in the processor
    my_utils.setup_pad_token_in_processor(processor)
    # Set up pad token (<pad>) in the base model
    my_utils.setup_pad_token_in_model(base_model, processor)

    ## 3.2 Create a LoRA Adapter
    # Create LoRA adapter with rank=8 and lora_alpha=64
    lora_model = my_utils.create_lora_adapter(base_model, 8, 64)
    if accelerator.is_main_process:                                         # Only use rank 0 to write files to avoid multi-process file writing
        my_utils.print_model_info_to_file(lora_model, './results/model_info/lora_model_info.txt')
        my_utils.print_model_parameters_to_file(lora_model, './results/model_info/lora_model_parameters.txt')
    accelerator.wait_for_everyone()                                         # Synchronize across all ranks to ensure rank 0 finishes

    ## 3.3 Set Up the Trainer
    custom_callback = my_utils_lora_finetuning_with_ddp.CustomCallbackDDP() # Optional: Reports VRAM usage and validates data parallelism by printing mini-batch details during the first training step.
    trainer = my_utils.setup_trainer(lora_model, processor, dataset, my_utils.collate_fn, epoch_num=3, batch_size=14, callbacks=[custom_callback])
    # Additional trainer setup in LoRA Fine-Tuning with DDP
    trainer.args.gradient_checkpointing_kwargs = {'use_reentrant': False}   # In DDP, use a non-reentrant backward pass to skip extra autograd graph traversals, making the backward pass faster.
    trainer.args.ddp_find_unused_parameters = False                         # Skip unnecessary checks for unused parameters to reduce overhead during backward computation

    ## 3.4 Train the Model
    my_utils.run_training(lora_model, trainer)

if __name__ == "__main__":
    training_workflow()
