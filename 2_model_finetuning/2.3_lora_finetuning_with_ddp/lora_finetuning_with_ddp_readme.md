# LoRA Fine-Tuning with Distributed Data Parallel (DDP)

---

## 1. LoRA Fine-Tuning with DDP Overview

LoRA (Low-Rank Adaptation) with Distributed Data Parallel (DDP) is a technique that combines the efficiency of LoRA fine-tuning with the scalability of distributed training. DDP enables LoRA fine-tuning to be performed across multiple GPUs, allowing for faster training of large language models and the ability to handle larger batch sizes.

In our experiments, each GPU hosts a complete copy of the model and processes a portion of the training data in parallel. At the end of each training step, gradients are synchronized across all GPUs to ensure consistent updates to the model weights. This parallelized approach significantly improves training speed and efficiency when fine-tuning large models.

<img src="./assets/LoRA_finetuning_with_ddp.png" height="300" alt="LoRA Fine-Tuning with DDP" style="border: 1px solid #ccc; padding: 5px; display: block;">

In the example shown above, for LoRA fine-tuning with DDP using 2 GPUs, the training data batch is split into 2 mini-batches. Each GPU hosts the full model and processes its assigned mini-batch independently. By doubling the training resources, this approach nearly doubles the training speed (1.91x), as shown in the 'Results' section below.

## 2. LoRA Fine-Tuning with DDP Workflow

In this project, the workflow for LoRA fine-tuning with DDP follows the same steps as those used in standard LoRA fine-tuning. For more details, refer to the 'Fine-Tuning Workflow' section in the [LoRA Fine-Tuning Readme](../2.1_lora_finetuning/lora_finetuning_readme.md#2-lora-fine-tuning-workflow).

The key difference in the LoRA fine-tuning with DDP notebook is that the `pre-training` and `training` steps are encapsulated within a workflow function. This workflow function is then executed using either `notebook_launcher` or `accelerate launch`.

## 3. Fine-Tuning Runs and GPU Selection 

`Run 1`: `run_1_a6000_48g_x2` (GPUs: 2 x RTX A6000 48GB)  
`Run 2`: `run_2_a6000_48g_x4` (GPUs: 4 x RTX A6000 48GB)  

## 4. File Structure in Run Folder 

`./lora_finetuning_with_ddp_nblaunch.ipynb`: A notebook for LoRA fine-tuning with DDP. In this notebook, the workflow function is executed using `notebook_launcher`.  
`./lora_finetuning_with_ddp_acclaunch.ipynb`: A notebook for LoRA fine-tuning with DDP. In this notebook, the workflow function is executed using `accelerate launch`.  

`./my_utils_lora_finetuning_with_ddp.py`: Contains utility functions specific to LoRA fine-tuning with Distributed Data Parallel (DDP). For example, it includes a function to override the built-in print() function to prepend a prefix string to all printed messages, functions to create custom device maps, and a custom callback function to report VRAM usage, as well as print the mini-batch size and the first data in the mini-batch.  
`./my_workflow_lora_finetuning_with_ddp.py`: Contains a workflow function that encapsulates the pre-training and training steps for LoRA fine-tuning, along with the modifications required for DDP.  

The remaining files in the run folder are similar to those used in LoRA fine-tuning. For more details, refer to the 'File Structure' section in the [LoRA Fine-Tuning Readme](../2.1_lora_finetuning/lora_finetuning_readme.md#4-file-structure-in-run-folder).

## 5. Settings and Hyperparameters in LoRA Fine-Tuning with DDP

#### Device Map

A custom device map was defined for multi-GPU training in Distributed Data Parallel. During training, multiple ranks (processes) are initiated in parallel, and each rank assigns its model layers to the GPU it is bound to.

#### Gradient Checkpointing

Standard LoRA fine-tuning uses `reentrant gradient checkpointing`. In LoRA fine-tuning with DDP, `non-reentrant gradient checkpointing` is enabled to allow faster backward passes and reduce computational overhead by disabling checks for unused parameters ([Discussion Link](https://github.com/huggingface/trl/issues/1303)).

#### Learning Rate Scaling

Unlike standard multi-GPU training, our LoRA fine-tuning with DDP experiments do not apply learning rate scaling. Typically, the effective batch size increases linearly with the number of GPUs allocated for training, and the learning rate is scaled proportionally to the batch size. However, due to the nature of LoRA fine-tuning, less aggressive updates are required as fewer parameters are modified. Consequently, no scaling or minimal scaling is sufficient, as LoRA adapters generally perform well with smaller learning rates.

Other settings and hyperparameters are similar to those described in standard LoRA fine-tuning. For more details, refer to the 'Settings and Hyperparameters' section in the [LoRA Fine-Tuning Readme](../2.1_lora_finetuning/lora_finetuning_readme.md#5-settings-and-hyperparameters-in-lora-fine-tuning).

## 6. LoRA Fine-Tuning with DDP Results

#### Overall Fine-Tuned Model Quality and Training Speed (LoRA Fine-Tuning with DDP vs LoRA Fine-Tuning)

As shown below, the similarity scores for both `LoRA with DDP` and `standard LoRA` are approximately 0.04. This demonstrates that LoRA with DDP retains model quality while significantly accelerating training, achieving a 1.91x speedup with double GPUs and a 3.59x speedup with quadruple GPUs.

| **Approach**                             | **Training Epochs** | **Batch Size** | **Effective Batch Size** | **Learning Rate Scaling<sup>[4]</sup>** | **Similarity Score** |
|------------------------------------------|---------------------|----------------|--------------------------|-----------------------------------------|----------------------|
| LoRA Fine-Tuning <sup>[1]</sup>          | 3                   | 14             | 14 x1                    | 1x (1e-5)                               | 0.037                |
| LoRA Fine-Tuning with DDP <sup>[2]</sup> | 3                   | 14             | 14 x2                    | 1x (1e-5)                               | 0.039                |
| LoRA Fine-Tuning with DDP <sup>[3]</sup> | 3                   | 14             | 14 x4                    | 1x (1e-5)                               | 0.042                |

| **Approach**                             | **GPU Setup**      | **CUDA Scaling Factor** | **VRAM Scaling Factor** | **Training Time** | **Training Speedup Factor** |
|------------------------------------------|--------------------|-------------------------|-------------------------|-------------------|-----------------------------|
| LoRA Fine-Tuning <sup>[1]</sup>          | RTX A6000 (48G) x1 | 1x (10,752 x1)          | 1x (48G x1)             | 23:51             | 1x (23:51)                  |
| LoRA Fine-Tuning with DDP <sup>[2]</sup> | RTX A6000 (48G) x2 | 2x (10,752 x2)          | 2x (48G x2)             | 12:29             | 1.91x (12:29)               |
| LoRA Fine-Tuning with DDP <sup>[3]</sup> | RTX A6000 (48G) x4 | 4x (10,752 x4)          | 4x (48G x4)             | 06:39             | 3.59x (06:39)               |


[1]: LoRA Fine-Tuning with 1 RTX A6000 (48G) GPU. The experiment results are from the [Complete LoRA Fine-Tuning Notebook](../2.1_lora_finetuning/run_1_a6000_48g_x1/lora_finetuning_complete.ipynb).  
[2]: LoRA Fine-Tuning with 2 RTX A6000 (48G) GPUs. The experiment results are from the [LoRA Fine-Tuning with DDP Notebook (2 x A6000)](./run_1_a6000_48g_x2/lora_finetuning_with_ddp_nblaunch.ipynb).  
[3]: LoRA Fine-Tuning with 4 RTX A6000 (48G) GPUs. The experiment results are from the [LoRA Fine-Tuning with DDP Notebook (4 x A6000)](./run_2_a6000_48g_x4/lora_finetuning_with_ddp_nblaunch.ipynb).  
[4]: In parallel training with multiple GPUs, such as in Distributed Data Parallel (DDP), the effective batch size typically increases linearly with the number of GPUs, often necessitating an adjustment to the learning rate. However, LoRA is less sensitive to these batch size changes compared to full fine-tuning because its additional low-rank layers are relatively small. Additionally, LoRA performs well with smaller learning rates, which help prevent overfitting and maintain stability in the low-rank updates. Therefore, we retain the original learning rate during LoRA DDP training.

## 7. References

`[HF: Designing a device map]`: (https://huggingface.co/docs/accelerate/main/en/concept_guides/big_model_inference#designing-a-device-map) 
`[HF: Distributed Data Parallel]`: (https://huggingface.co/docs/transformers/en/perf_train_gpu_many#dataparallel-vs-distributeddataparallel)
`[HF: accelerate launch and notebook_launcher]`: https://huggingface.co/docs/transformers/en/accelerate