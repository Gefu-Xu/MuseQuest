#--------------- A Training Workflow in Full Fine-Tuning with Fully Sharded Data Parallel ---------------
def training_workflow():
    import my_login
    import my_utils
    import my_utils_full_finetuning_with_fsdp

    # **Step 1: Rank (Process) Initialization**
    ## 1.1 Initialize an (Utility) Accelerator at the Beginning of Each Rank (Process)
    import torch
    from accelerate import Accelerator
    accelerator = Accelerator()
    rank = accelerator.process_index                                         # Get the current rank (process) ID
    torch.cuda.set_device(rank)                                              # Bind the current rank (process) to the GPU with the same ID (e.g., rank 0 to GPU 0, rank 1 to GPU 1, etc.)

    ## 1.2 Override print() to Prepend a Prefix String ('[RANK{current rank ID}:GPU{current GPU ID}]') to All Printed Messages
    gpu_id = torch.cuda.current_device()
    prefix_string = f'[RANK{rank}:GPU{gpu_id}]'
    my_utils_full_finetuning_with_fsdp.add_prefix_to_print_function(prefix_string)
    print(f'==> Process started...')

    # **Step 2: Pre-training**
    ## 2.1 Log in to Required Accounts (e.g., Hugging Face, Weights & Biases)
    if accelerator.is_main_process:                                           # Only use rank 0 to log in to avoid multiple logins
        # Log in to the Hugging Face account to access the Idefics2 model and the dataset
        my_login.login_hf()
        # Log in to the Weights & Biases account to enable tracking of training progress
        my_login.login_wandb()
    accelerator.wait_for_everyone()                                           # Synchronize across all ranks to ensure rank 0 finishes

    ## 2.2 Print (Utility) Accelerator Info
    if accelerator.is_main_process:                                           # Use rank 0 only to print info to prevent redundant prints from multiple processes
        # Print (utility) accelerator info to verify that FSDP is properly configured and reflected in the accelerator state
        my_utils_full_finetuning_with_fsdp.print_util_accelerator_info(accelerator)
    accelerator.wait_for_everyone()                                           # Synchronize across all ranks to ensure rank 0 finishes

    ## 2.3 Load the Processor (Tokenizer) and the Base Model
    # Load processor
    processor = my_utils.load_processor()
    # Load model
    base_model = my_utils.load_model(device_map=None)                         # device_map is derived from the accelerate configuration
    if accelerator.is_main_process:                                           # Only use rank 0 to write files to avoid multi-process file writing
        my_utils.print_processor_info_to_file(processor, './results/model_info/original_processor_info.txt')
        my_utils.print_model_info_to_file(base_model, './results/model_info/base_model_info.txt')
        my_utils.print_model_parameters_to_file(base_model, './results/model_info/base_model_parameters.txt')
    accelerator.wait_for_everyone()                                           # Synchronize across all ranks to ensure rank 0 finishes

    ## 2.4 Load the Dataset
    # Load the Hugging Face dataset
    dataset = my_utils.load_hf_dataset(hf_path='xugefu/MuseQuest')

    ## 2.5 Perform Inference with the Base Model to Establish a Baseline
    base_model.to(f'cuda:{rank}')                                             # Once FSDP training starts, model weights are automatically sharded and moved to GPUs. However, if we run inference before training and device_map is set to 'None', we must manually move the model to GPUs before running inference.
    if accelerator.is_main_process:                                           # Only use rank 0 to run inference to avoid multi-process file writing
        # Perform batch inference on the test set and save results to a CSV file
        my_utils.batch_inference_model(base_model, processor, dataset['test'], './results/inference_results/inference_results_before_finetuning.csv', 10)
        # Plot similarity scores from the inference CSV file, using thumbnails, and save the plot image locally
        my_utils.plot_similarity_scores([('Original', './results/inference_results/inference_results_before_finetuning.csv')],
                                        thumbnail_dir='./results/inference_results/thumbnail',
                                        plot_file_name='./results/inference_results/similarity_scores_before_finetuning.jpg'
        )
    accelerator.wait_for_everyone()                                           # Synchronize across all ranks to ensure rank 0 finishes

    # **Step 3: Training**
    ## 3.1 Add a Custom Padding Token
    # Set up pad token (<pad>) in the processor
    my_utils.setup_pad_token_in_processor(processor)
    # Set up pad token (<pad>) in the base model
    my_utils.setup_pad_token_in_model(base_model, processor)

    ## 3.2 Select Layers for Full Fine-Tuning
    # - Freeze the `vision_model` module entirely.
    # - Fine-tune linear ('proj') layers in the `text_model` and `connector` modules.
    fine_tuning_module_list = ['text_model', 'connector']
    full_fine_tuning_model = my_utils_full_finetuning_with_fsdp.select_layers_for_full_fine_tuning(base_model, fine_tuning_module_list)
    if accelerator.is_main_process:                                            # Only use rank 0 to write files to avoid multi-process file writing
        my_utils.print_model_info_to_file(full_fine_tuning_model, './results/model_info/full_finetuning_model_info.txt')
        my_utils.print_model_parameters_to_file(full_fine_tuning_model, './results/model_info/full_finetuning_model_parameters.txt')
    accelerator.wait_for_everyone()                                            # Synchronize across all ranks to ensure rank 0 finishes

    ## 3.3 Set Up the Trainer
    custom_callback = my_utils_full_finetuning_with_fsdp.CustomCallbackFSDP()  # Optional: Report VRAM usage, print mini-batch details, and display parameter sharding percentage on the current device during the first training step.
    trainer = my_utils.setup_trainer(full_fine_tuning_model, processor, dataset, my_utils.collate_fn, epoch_num=3, batch_size=10, callbacks=[custom_callback], use_fsdp=True)  # use_fsdp=True: disable gradient checkpointing-related settings in TrainingArguments and use activation_checkpointing from the accelerate configuration instead.
    # Additional trainer setup in Full Fine-Tuning with FSDP
    trainer.args.learning_rate = 2e-5                                        # Scale up learning rate for 4-GPU training

    ## 3.4 Print Trainer Info and Module Wrapping Status Before Training
    # Note 1: The Trainer initializes an internal accelerator. It also uses 'dataset', 'batch_size', and 'collate_fn' to create train_dataloader and eval_dataloader.
    # Note 2: The internal accelerator moves (via prepare()) the dataloaders from CPU to GPU after Trainer setup, and moves the model and optimizer to GPU once training starts.
    if accelerator.is_main_process:                                            # Only use rank 0 to write files to avoid multi-process file writing
        my_utils_full_finetuning_with_fsdp.print_trainer_info(trainer)
        my_utils_full_finetuning_with_fsdp.print_module_wrapping_status_to_file(trainer.model, './results/model_info/module_wrapping_status_before_training.txt')
    accelerator.wait_for_everyone()                                            # Synchronize across all ranks to ensure rank 0 finishes

    ## 3.5 Train the Model
    my_utils.run_training(full_fine_tuning_model, trainer)

    ## 3.6 Print Trainer Info and Module Wrapping Status After Training
    # Note: Once training starts, model weights are sharded and moved to the appropriate GPU during the first forward pass by the internal accelerator. Meanwhile, the optimizer is initialized and moved to the GPU by the internal accelerator as well. The module status now becomes wrapped.
    if accelerator.is_main_process:                                            # Only use rank 0 to write files to avoid multi-process file writing
        my_utils_full_finetuning_with_fsdp.print_trainer_info(trainer)
        my_utils_full_finetuning_with_fsdp.print_module_wrapping_status_to_file(trainer.model, './results/model_info/module_wrapping_status_after_training.txt')
    accelerator.wait_for_everyone()                                            # Synchronize across all ranks to ensure rank 0 finishes

if __name__ == "__main__":
    training_workflow()
