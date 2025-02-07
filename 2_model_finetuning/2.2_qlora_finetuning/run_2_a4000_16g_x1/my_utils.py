#--------------- Processor and Base Model ---------------
# 1) Load processor
# !pip install transformers==4.46.0 -q -U
from transformers import AutoProcessor
def load_processor(model_path='HuggingFaceM4/idefics2-8b'):
    print(f'==> Loading processor...')
    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=model_path,
        do_image_splitting=False                   # No image splitting in order to reduce memory usage
    )
    return processor

# 2) Load model
# !pip install hf_transfer==0.1.8 -q -U 
# !pip install accelerate==0.34.2 -q -U
# !pip install flash-attn==2.6.3 -q -U
import os
import torch
from transformers import Idefics2ForConditionalGeneration
def load_model(model_path='HuggingFaceM4/idefics2-8b', device_map='auto', quantization_config=None):
    print(f'==> Loading model...')
    # Enable HF_TRANSFER to optimize data transfer from the Hugging Face Hub
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    model = Idefics2ForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_path,
        # If device_map='auto', the accelerate library automatically distributes the model across devices in the order: GPU0 -> GPU1 -> CPU -> disk.
        device_map=device_map,
        # Use torch.bfloat16 (supported by Ampere architecture GPUs) to save VRAM. The default data type is torch.float32.
        torch_dtype=torch.bfloat16,
        # "flash_attention_2" significantly reduces VRAM usage and improves the efficiency of LLaMA architecture models.
        # For QLoRA, mixed precision issues were encountered, so the 'eager' attention implementation is used instead, providing moderate VRAM savings.
        attn_implementation='eager' if quantization_config else 'flash_attention_2',
        # Enable the quantization configuration when using QLoRA.
        quantization_config=quantization_config,
        # Use the default cache directory specified by the Hugging Face library.
        cache_dir=''
    )
    return model

# 3) Print brief model info to the terminal and save detailed model info to a file
import os
def print_model_info_to_file(model, filename):
    print(f'==> Printing model info...')
    filepath = os.path.dirname(filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    file = open(filename, 'w')
    print(f'==> Model type: {type(model).__name__}')
    print(f'==> Model is located on device: {model.device}')
    # If model is not on CPU, print how model layers are distributed on GPUs
    print(f"==> Model device map: {getattr(model, 'hf_device_map', 'All layers are on CPU')}")
    print(f'==> For complete model info (type, architecture, config, generation config, device map), refer to {filename}.')
    file.write(f'==> Model type: {type(model).__name__}\n')
    file.write(f'==> Model architecture: {model}\n')
    # model.config includes bos_token_id, eos_token_id, max_position_embeddings, vocab_size, etc.
    file.write(f'==> Model config: {model.config}\n')
    # model.generation_config includes bos_token_id, eos_token_id, max_length, pad_token_id, temperature, etc.
    file.write(f'==> Model generation config: {model.generation_config}\n')
    file.write(f'==> Model is located on device: {model.device}\n')
    file.write(f"==> Model device map: {getattr(model, 'hf_device_map', 'All layers are on CPU')}\n")
    file.close()

# 4) Print model parameter samples to the terminal and save the complete model parameter info to a file
def print_model_parameters_to_file(model, filename):
    print(f'==> Printing model parameters...')
    filepath = os.path.dirname(filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    file = open(filename, 'w')
    trainable_parameter = 0
    non_trainable_parameter = 0
    display_size = 15
    total_block = len(list(model.named_parameters()))
    for idx, (n, p) in enumerate(model.named_parameters()):
        # Print the first and last 15 blocks' parameters as samples to the terminal
        if idx < display_size or idx > total_block - display_size:
            print(f'{n} ({p.numel()}, {p.dtype}, {p.device.type}, {p.requires_grad})')
        # Print all model parameters to the file
        file.write(f'{n} ({p.numel()}, {p.dtype}, {p.device.type}, {p.requires_grad})\n')
        if p.requires_grad:
            trainable_parameter += p.numel()
        else:
            non_trainable_parameter += p.numel()
    total_parameter = trainable_parameter + non_trainable_parameter
    print(f'==> Total parameters: {total_parameter}, Trainable parameters: {100 * trainable_parameter / total_parameter}%')
    print(f'==> Parameters in the first {display_size} blocks and the last {display_size} blocks are displayed here. For complete model parameter info, refer to {filename}.')
    file.write(f'==> Total parameters: {total_parameter}, Trainable parameters: {100 * trainable_parameter / total_parameter}%\n')
    file.close()

# 5) Print brief processor info to the terminal and save detailed processor info to a file
def print_processor_info_to_file(processor, filename):
    print(f'==> Printing processor info...')
    filepath = os.path.dirname(filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    file = open(filename, 'w')
    print(f'==> Regular token count: {processor.tokenizer.vocab_size}, regular+special token count: {len(processor.tokenizer)}, bos_token: {processor.tokenizer.bos_token}, eos_token: {processor.tokenizer.eos_token}, pad_token: {processor.tokenizer.pad_token}')
    print(f'==> For complete processor info, refer to {filename}.')
    file.write(f'==> Regular token count: {processor.tokenizer.vocab_size}, regular+special token count: {len(processor.tokenizer)}, bos_token: {processor.tokenizer.bos_token}, eos_token: {processor.tokenizer.eos_token}, pad_token: {processor.tokenizer.pad_token}\n')
    file.write(f'==> Processor info: {processor}\n')
    file.close()

#--------------- Hugging Face Dataset ---------------
#1) Load dataset (source: https://huggingface.co/datasets/xugefu/MuseQuest)
# !pip install datasets==3.0.0 -q -U
from datasets import load_dataset
def load_hf_dataset(hf_path="xugefu/MuseQuest"):
    print(f'==> Loading hf dataset...')
    dataset = load_dataset(path=hf_path)
    return dataset

#--------------- Inferencing ---------------
# 1) Calculate semantic similarity score between two sentence batches
#   Note: Although this function supports sentence batches, in our use case, we calculate a score between 2 sentences only
def calculate_semantic_similarity_score(model, processor, sentence_batch1, sentence_batch2):
    # Tokenizing sentence_batch1 (e.g., generated answer batch) and sentence_batch2 (e.g., ground truth batch)
    # sentence_batch_pt['input_ids'] dimensions: (batch_size, sequence_length)
    sentence_batch1_pt = processor(text=sentence_batch1, return_tensors="pt").to(model.device)
    sentence_batch2_pt = processor(text=sentence_batch2, return_tensors="pt").to(model.device)

    # Get sentence_batch1 and sentence_batch2 contextual embeddings through a model forward pass; embedding is from the last hidden layer
    # embedding_vec dimensions: (batch_size, hidden_size)
    with torch.no_grad():
        embedding_vec_batch1 = model(**sentence_batch1_pt, output_hidden_states=True).hidden_states[-1].mean(dim=1)
        embedding_vec_batch2 = model(**sentence_batch2_pt, output_hidden_states=True).hidden_states[-1].mean(dim=1)

    # Calculate semantic similarity score (cosine similarity) for each record in sentence_batch1 and sentence_batch2
    # cosine_sim dimensions: (batch_size), value range: float[1, -1]
    cosine_sim = torch.cosine_similarity(embedding_vec_batch1, embedding_vec_batch2, dim=1)
    # Calculate overall semantic similarity score for sentence_batch1 and sentence_batch2 at the batch level
    # semantic_loss dimensions: (1), value range: float[0, 2]
    semantic_loss = 1.0 - cosine_sim.mean()
    return semantic_loss

# 2) Batch inference, save image thumbnails, and save inference results to a CSV file
# Note: The saved image thumbnails and the CSV file will be used in plot_similarity_scores() next step
import pandas as pd
import json
def batch_inference_model(model, processor, dataset, report_csv, batch_size=4):
    print(f'==> Performing batch inference and saving results to {report_csv} ...')
    
    # Prepare directories for image thumbnails and CSV file
    report_csv_dir = os.path.dirname(report_csv)
    thumbnail_dir = os.path.join(report_csv_dir, 'thumbnail')
    if not os.path.exists(report_csv_dir):
        os.makedirs(report_csv_dir)
    if not os.path.exists(thumbnail_dir):
        os.makedirs(thumbnail_dir)

    # Start inference
    model.eval()  # model.eval() disables dropout and switches batch norm to use batch statistics instead of accumulated statistics
    model.config.use_cache = True  # Enable cache to speed up token-by-token generation
    report_df = pd.DataFrame(columns=['data_id', 'item_name', 'image_name', 'user_question', 'model_generated_answer', 'ground_truth_answer', 'semantic_similarity_score'])
    accumulated_semantic_similarity_score = 0
    with torch.no_grad():  # Disable gradient tracking to reduce VRAM usage
        total_batch = (len(dataset) - 1) // batch_size + 1  # Calculate total number of batches
        batch_data = []
        for i, dataset_data in enumerate(dataset):
            batch_data.append(dataset_data)
            # If batch_data[] is full or we reach the last data point, start inference
            if len(batch_data) == batch_size or i == len(dataset) - 1:
                cur_batch_idx = i // batch_size
                print(f'==> [Batch: {cur_batch_idx + 1}/{total_batch}] Data in batch: {len(batch_data)}')
                # Save thumbnails of current batch
                batch_image = [[data['image']] for data in batch_data]
                batch_image_resized = [data['image'].resize((100, 100)) for data in batch_data]
                batch_thumbnail = [data['image'].resize((20, 20)) for data in batch_data]
                for idx, data in enumerate(batch_data):
                    batch_thumbnail[idx].save(os.path.join(thumbnail_dir, data["image_name"]))
                # Tokenize 'user message + image' for the current batch
                batch_dialog_json = [json.loads(data['dialog']) for data in batch_data]
                batch_user_msg = [dialog_json['messages'][0] for dialog_json in batch_dialog_json]
                batch_assistant_msg = [dialog_json['messages'][1] for dialog_json in batch_dialog_json]
                batch_user_question = [user_msg['content'][1]['text'] for user_msg in batch_user_msg]
                batch_ground_truth_answer = [assistant_msg['content'][0]['text'] for assistant_msg in batch_assistant_msg]
                batch_user_msg_formatted_with_next_prompt = [  # 'User:<image>xxx<end_of_utterance>\nAssistant:'
                    processor.apply_chat_template([user_msg], add_generation_prompt=True, tokenize=False)
                    for user_msg in batch_user_msg
                ]
                batch_tokenized_input = processor(
                    text=batch_user_msg_formatted_with_next_prompt,
                    images=batch_image,
                    return_tensors="pt",
                    padding=True  # Padding to the longest sequence in the batch
                ).to(model.device)
                
                # Run batch inference
                batch_generated_output = model.generate(**batch_tokenized_input, max_new_tokens=512)
                batch_generated_text = processor.batch_decode(batch_generated_output, skip_special_tokens=True)
                
                # Process inference results
                for idx, data in enumerate(batch_data):
                    # Display inference result (generated answer) and evaluate it against ground truth using semantic similarity scores
                    generated_answer = batch_generated_text[idx].split('Assistant:')[1].strip()
                    semantic_similarity_score = calculate_semantic_similarity_score(model, processor, generated_answer, batch_ground_truth_answer[idx])
                    print(f'==> [Batch: {cur_batch_idx + 1}/{total_batch}][Data: {idx + 1}/{len(batch_data)}] Inference for data_id: {data["data_id"]}, item_name: {data["item_name"]}, image_name: {data["image_name"]}:')
                    batch_image_resized[idx].show()
                    print(f'==> Original user message: {batch_user_msg[idx]}')
                    print(f'==> User message + assistant prompt (text): {batch_user_msg_formatted_with_next_prompt[idx]}')
                    print(f'==> Model generated answer: {generated_answer}')
                    print(f'==> Ground truth answer: {batch_ground_truth_answer[idx]}')
                    print(f'==> Semantic similarity score: {semantic_similarity_score}')
                    
                    # Save results in the CSV file
                    data_df = pd.DataFrame([{'data_id': data['data_id'],
                                             'item_name': data['item_name'],
                                             'image_name': data['image_name'],
                                             'user_question': batch_user_question[idx],
                                             'model_generated_answer': generated_answer,
                                             'ground_truth_answer': batch_ground_truth_answer[idx],
                                             'semantic_similarity_score': semantic_similarity_score.item()}])
                    report_df = pd.concat([report_df, data_df], ignore_index=True)
                    accumulated_semantic_similarity_score += semantic_similarity_score.item()
                batch_data = []  # Clear batch_data[] for the next batch run
                torch.cuda.empty_cache()  # Free GPU VRAM after each batch
                
    # Calculate and save average similarity score for the dataset
    ave_semantic_similarity_score = round(accumulated_semantic_similarity_score / len(dataset), 3)
    print(f'==> Average semantic similarity score: {ave_semantic_similarity_score} (0 is the best, 2 is the worst)')
    report_df = pd.concat([report_df, pd.DataFrame([{'item_name': 'average_semantic_similarity_score', 'semantic_similarity_score': ave_semantic_similarity_score}])], ignore_index=True)
    report_df.to_csv(report_csv, index=False, encoding='utf-8')
    torch.cuda.empty_cache()  # Clean up GPU VRAM after all batch inferences are complete

# 3) Plot multiple inference run results in a diagram
# !pip install matplotlib==3.9.2 -q -U
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
def plot_similarity_scores(inference_result_csv_list, thumbnail_dir=None, plot_file_name='./results/inference_results/default_plot.jpg'):
    # Note1: inference_result_csv_list format example: [('inference_run_tag1', 'inference_result1.csv'), ('inference_run_tag2', 'inference_result2.csv'), ...]
    # Note2: thumbnail_dir is the directory created during batch_inference_model() execution
    # Note3: y-axis: df['data_id'], x-axis: df['semantic_similarity_score'], y-tick: df['data_id'] + df['item_name'], legend: inference_run_tag

    print(f'==> Plotting similarity scores...')
    ytick_dict = {}  # Union lookup table for all inference results: key: df['data_id'], value: df['data_id'] + df['item_name']
    image_dict = {}  # Union lookup table for all inference results: key: df['data_id'], value: df['image_name']
 
    fig, ax = plt.subplots(figsize=(8, 30))    
    # Plot 1 curve from 1 CSV file
    for inference_run_tag, inference_result_csv in inference_result_csv_list:
        df = pd.read_csv(inference_result_csv)        
        # 1. Extract average similarity score from the last row, then remove the last row to keep all rows uniform
        ave_semantic_similarity_score = df['semantic_similarity_score'].iloc[-1]
        df_trimmed = df.iloc[:-1]
        # 2. Sort by 'data_id' for y-axis, plot (y, x) curve, and plot a reference line using average similarity score
        df_sorted = df_trimmed.sort_values(by='data_id')
        x = df_sorted['semantic_similarity_score']
        y = df_sorted['data_id']
        lines = ax.plot(x, y, marker='o', linestyle='-', label=inference_run_tag)
        ax.axvline(x=ave_semantic_similarity_score, color=lines[0].get_color(), linestyle='--', label=f'{inference_run_tag}(ave)={ave_semantic_similarity_score:.3f}')
        # 3. Update y-tick info and image info in lookup tables
        ytick = '[' + df_sorted['data_id'].astype(int).astype(str) + '] ' + df_sorted['item_name']
        ytick_dict.update(dict(zip(y, ytick)))
        image_dict.update(dict(zip(y, df_sorted['image_name'])))

    # Figure title and global settings
    plt.title('Semantic Similarity Score Comparison')                     # Set figure title
    ax.axvline(x=0, color='red', linestyle='--', label='ground_truth=0')  # Add a reference line for ground truth
    ax.legend()                                                           # Show legend
    # x-axis settings
    plt.xlim(-0.2, 0.8)                                                   # Set x-axis range
    plt.xlabel('Semantic Similarity Score')                               # Set x-axis label
    # y-axis settings
    plt.ylabel('Item')                                                    # Set y-axis label
    y_of_all_plots = sorted(ytick_dict.keys())                            # All y values (from df['data_id'])
    yticks_of_all_plots = [ytick_dict[y] for y in y_of_all_plots]         # All y-ticks (from df['data_id'] + df['item_name'])
    plt.yticks(y_of_all_plots, yticks_of_all_plots)                       # Attach y-ticks to y-axis   
    for data_id, image_name in image_dict.items():                        # Attach thumbnails (if available, from df['image_name']) next to y-axis
        image_path_name = os.path.join(thumbnail_dir, image_name)
        if os.path.exists(image_path_name):
            img = plt.imread(image_path_name)
            imagebox = OffsetImage(img, zoom=1)
            ab = AnnotationBbox(imagebox, (-0.15, data_id), frameon=False)
            ax.add_artist(ab)
        else:
            print(f"Image file {image_path_name} not found. Skipping.")
    # Save and show the figure
    plt.savefig(plot_file_name, bbox_inches='tight')                      # Use tight bounding box to avoid cut off
    plt.show()                                                            # Display figure; plt.show() clears the figure, so we save figure before this step

#--------------- Set Up Pad Token ---------------
# 1) Set up pad token in processor
def setup_pad_token_in_processor(processor, pad_token='<pad>'):
    print(f'==> Setting up pad token (<pad>) in processor...')
    print(f'==> [Original] Regular token count: {processor.tokenizer.vocab_size}, regular + special token count: {len(processor.tokenizer)}, pad token in tokenizer: {processor.tokenizer.pad_token}')
    if pad_token not in processor.tokenizer.get_vocab():
        print(f'==> Adding new pad token: {pad_token}')
        processor.tokenizer.add_special_tokens({'pad_token': pad_token})
    processor.tokenizer.pad_token = pad_token
    processor.tokenizer.padding_side = 'left'     # Flash Attention version of the Mistral model requires left padding
    print(f'==> [Updated] Regular token count: {processor.tokenizer.vocab_size}, regular + special token count: {len(processor.tokenizer)}, pad token in tokenizer: {processor.tokenizer.pad_token}')

# 2) Set up pad token in model
def setup_pad_token_in_model(model, processor):
    print(f'==> Setting up pad token (<pad>) in model...')
    print(f'==> Configuring pad token in model.config and model.model.text_model for input sequence padding...')
    pad_token_id = processor.tokenizer.pad_token_id 
    model.config.pad_token_id = pad_token_id
    model.config.perceiver_config.pad_token_id = pad_token_id
    model.config.text_config.pad_token_id = pad_token_id
    model.config.vision_config.pad_token_id = pad_token_id
    model.model.text_model.embed_tokens.padding_idx = pad_token_id
    print(f'==> Configuring pad token in model.generation_config for output sequence padding...')
    model.generation_config.pad_token_id = pad_token_id
    print(f'==> Checking if resizing of model embeddings is needed...')
    current_embedding_size = model.get_input_embeddings().num_embeddings
    if current_embedding_size != len(processor.tokenizer):
        new_embedding_size = len(processor.tokenizer)
        model.resize_token_embeddings(new_embedding_size)
        print(f'==> Resized model embeddings from {current_embedding_size} to {new_embedding_size}.')
    else:
        print(f'==> No resizing needed for token embeddings.')

#--------------- LoRA Adapter ---------------
# 1) Create a LoRA adapter
# !pip install peft==0.13.0 -q -U
from peft import LoraConfig, get_peft_model
def create_lora_adapter(base_model, r, lora_alpha):
    print(f'==> Creating LoRA adapter...')
    peft_config = LoraConfig( 
        # Define 'target_modules' to select linear layers in 'text_model' and 'connector' modules for fine-tuning
        # Example regex usage link: https://github.com/huggingface/peft/blob/v0.12.0/src/peft/tuners/lora/config.py#L162
        target_modules= '.*(text_model|connector).*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*$',  
        r=r,                                      # LoRA rank
        lora_alpha=lora_alpha,                    # LoRA scaling factor        
        bias="none",                              # Typically in LoRA, we fine-tune weights, not biases
        lora_dropout=0.1,                         # Dropout to help prevent overfitting
        task_type="CAUSAL_LM",                    # Set for causal language modeling tasks
        init_lora_weights="gaussian",             # Initialize LoRA A matrix with Gaussian values; LoRA B matrix with zeros
        use_rslora=True,                          # Use Rank-Stabilized LoRA for stability
    )
    lora_model = get_peft_model(base_model, peft_config)   
    return lora_model

# 2) Load LoRA adapter from a checkpoint and set LoRA to trainable state
from peft import PeftModel
def load_lora_adapter_from_checkpoint(base_model, lora_checkpoint):
    # Setting 'is_trainable=True' puts LoRA in a trainable state
    # Source: https://discuss.huggingface.co/t/correct-way-to-save-load-adapters-and-checkpoints-in-peft/77836/2
    print(f'==> Loading LoRA adapter from checkpoint...')
    lora_model = PeftModel.from_pretrained(base_model, lora_checkpoint, is_trainable=True)
    return lora_model

#--------------- Trainer ---------------
# 1) Define custom DataCollator (collate_fn()) to process data in a batch
def collate_fn(dataset, processor, inspection_mode=False):
    '''
    Example data formats used in this function are detailed below for reference:
    1. Format of input dataset:
        {
            'image': image0, 
             ...
            'dialog': '{"messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "aaa"} ]}, 
                                    {"role": "assistant", "content": [{"type": "text", "text": "bbb"} ]}]
                      }'
        }
        {
            'image': image1, 
             ...
            'dialog': '{"messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "ccc"} ]}, 
                                    {"role": "assistant", "content": [{"type": "text", "text": "ddd"} ]}]
                      }'
        }
    2. Format of texts[]:
        [
            'User:<image>aaa<end_of_utterance>\nAssistant: bbb<end_of_utterance>\n', 
            'User:<image>ccc<end_of_utterance>\nAssistant: ddd<end_of_utterance>\n'
        ]
    3. Format of images[]:
        [
            [<PIL.Image.Image image mode=RGB size=800x800 at 0x17B44EC7D90>], 
            [<PIL.Image.Image image mode=RGB size=800x800 at 0x17B44EC7F10>]
        ]
    4. Format of output batch (right side padding is applied here just for demonstration):
        {
        'input_ids':            tensor([ [texts[0].tensor ............................ + <pad><pad>...<pad>], [texts[1].tensor ............................ + <pad><pad>...<pad>] ]), 
        'labels':               tensor([ [texts[0].tensor(<image> and question masked) + -100 -100 ...-100 ], [texts[1].tensor(<image> and question masked) + -100 -100 ...-100 ] ]), 
        'attention_mask':       tensor([ [1    1    1  ......................  1    1  +   0    0  ...  0  ], [1    1    1  ......................  1    1  +   0    0  ...  0  ] ]), 
        'pixel_values':         tensor([ [images[0].pixels ................................... ],             [images[1].pixels ................................... ]             ]), 
        'pixel_attention_mask': tensor([ [1    1    1  ......................  1    1    1    1],             [1    1    1  ......................  1    1    1    1]             ])
        }
    '''

    # Step 1: Process the data batch by extracting images to images[], extracting and formatting texts to texts[]
    texts = [processor.apply_chat_template(json.loads(data['dialog'])['messages'], add_generation_prompt=False, tokenize=False) for data in dataset]
    images = [[data['image']] for data in dataset]

    # Step 2: Tokenize texts[] and images[] with padding applied
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)   # padding=True pads sequences to the longest in the batch
  
    # Step 3: Create batch["labels"]
    # During training:
    # batch["input_ids"] serves as the model input:       'User:<image>question<end_of_utterance>\nAssistant: answer<end_of_utterance>\n<pad><pad>'
    # batch["labels"] serves as the target output:        'User:xxxxxxxxxxxxxxx<end_of_utterance>\nAssistant: answer<end_of_utterance>\nxxxxxxxxxx'
    # batch["labels"] is identical to batch["input_ids"], but with '<image>' token, 'user question' and '<pad>' token masked (set to -100) to ignore them in loss computation.
    labels = batch['input_ids'].clone()
    fake_token_around_image_token_id = processor.tokenizer.convert_tokens_to_ids('<fake_token_around_image>')  # fake_token_around_image_token_id: 32000
    image_token_id = processor.tokenizer.convert_tokens_to_ids('<image>')                                      # image_token_id: 32001
    end_of_utterance_token_id = processor.tokenizer.convert_tokens_to_ids('<end_of_utterance>')                # end_of_utterance_token_id: 32002
    pad_token_id = processor.tokenizer.pad_token_id                                                            # pad_token_id: 32003
    # Step 3.1: Mask 'user question' section
    for row in labels:
        # Check for exactly two `<end_of_utterance>` tokens; otherwise data format is invalid
        end_of_utterance_token_id_count = torch.sum(row == end_of_utterance_token_id).item()
        if end_of_utterance_token_id_count != 2:
            raise ValueError(f'==> [Error] Expected 2 occurrences of <end_of_utterance> in data, but found {end_of_utterance_token_id_count}.')
        # Mask 'user question' section in the current row
        updating = False
        for i in range(1, len(row)):
            if row[i - 1] == fake_token_around_image_token_id and row[i] != image_token_id:  # 'user question' starts after image section
                updating = True
            elif row[i] == end_of_utterance_token_id:                                        # 'user question' ends at the first <end_of_utterance>
                break
            if updating:
                row[i] = -100 
    # Step 3.2: Mask '<pad>' tokens
    labels[labels == pad_token_id] = -100
    # Step 3.3: Mask '<image>' tokens
    labels[labels == image_token_id] = -100

    batch['labels'] = labels

    if inspection_mode: 
        print(f'==> [custom DataCollator] Format of input dataset: \n{dataset[0]}\n{dataset[1]}')
        print(f'==> [custom DataCollator] Text count in the dataset: {len(texts)}, image count in the dataset: {len(images)}')
        print(f'==> [custom DataCollator] Format of texts[]: \n{texts}')
        print(f'==> [custom DataCollator] Format of images[]: \n{images}')
        print(f'==> [custom DataCollator] Format of output batch[\'input_ids\']: \n{batch["input_ids"]}')
        print(f'==> [custom DataCollator] Format of output batch[\'labels\']: \n{batch["labels"]}')
        print(f'==> [custom DataCollator] Format of output batch[\'attention_mask\']: \n{batch["attention_mask"]}')
        print(f'==> [custom DataCollator] Format of output batch[\'pixel_values\']: \n{batch["pixel_values"]}')
        print(f'==> [custom DataCollator] Format of output batch[\'pixel_attention_mask\']: \n{batch["pixel_attention_mask"]}')

    return batch

# 2) Define a custom callback function to report VRAM usage
from transformers import TrainerCallback
class CustomCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Report VRAM usage only if using a GPU and during the first training step
        if torch.cuda.is_available() and state.global_step == 1:
            num_gpus = torch.cuda.device_count()
            for gpu_id in range(num_gpus):
                gpu_memory = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
                gpu_max_memory = torch.cuda.max_memory_allocated(gpu_id) / (1024 ** 3)
                total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 3)
                print(f'[GPU{gpu_id}] Step {state.global_step}: Current VRAM usage: {gpu_memory:.2f} GB, Peak VRAM usage: {gpu_max_memory:.2f} GB, Total GPU VRAM: {total_memory:.2f} GB')

# 3) Set up hyperparameters and other training settings in a Hugging Face Trainer
from transformers import Trainer, TrainingArguments
def setup_trainer(model, processor, dataset, collate_fn, epoch_num, batch_size, callbacks=None, use_fsdp=False):
    print(f'==> Setting up trainer...')
    trainer = Trainer(
        model=model,
        data_collator=lambda batch: collate_fn(batch, processor=processor),              # Use a custom DataCollator
        train_dataset=dataset["train"],                                                  # Training dataset
        eval_dataset=dataset["validation"],                                              # Validation dataset
        callbacks=callbacks,                                                             # Optional: Custom callbacks (e.g., for monitoring and reporting VRAM during training)
        args=TrainingArguments(
            # 1. Hyperparameters                               
            optim="adamw_torch",                                                         # Use the AdamW optimizer
            weight_decay=0.01,                                                           # Apply a 1% penalty on weight magnitude at each training step to reduce overfitting
            max_grad_norm=0.3,                                                           # Clip gradients to stabilize training and avoid large updates
            warmup_ratio=0.03,                                                           # During the first 3% of training steps, linearly increase the learning rate from 0 to the designated value (e.g., 1e-5)
            learning_rate=1e-5,                                                          # For the remaining 97% of steps, the learning rate starts at 1e-5
            lr_scheduler_type="cosine",                                                  # The learning rate then follows a cosine decay schedule, gradually decreasing toward the end of training
            log_level="debug",                                                           # Provide detailed training logs
            bf16=True,                                                                   # Use bf16 to reduce VRAM usage (requires Ampere or newer GPUs); default is fp32
            remove_unused_columns=False,                                                 # Retain all dataset columns, including those not explicitly consumed by the model's forward method.
            gradient_checkpointing=None if use_fsdp else True,                           # FSDP: defer to FSDP config; otherwise, enable gradient checkpointing to reduce VRAM usage by recomputing activations
            gradient_checkpointing_kwargs=None if use_fsdp else {'use_reentrant': True}, # FSDP: defer to FSDP config; otherwise, use reentrant autograd for backward passes

            # 2. Training settings (train_steps = (epoch_num * total_samples_in_dataset) / (batch_size * gradient_accumulation_steps))
            num_train_epochs=epoch_num,                                                  # Total number of epochs
            per_device_train_batch_size=batch_size,                                      # Training batch size per device
            per_device_eval_batch_size=batch_size,                                       # Evaluation batch size per device
            gradient_accumulation_steps=1,                                               # Set to 1 to allow frequent updates when training on limited data

            # 3. Evaluation settings                               
            eval_strategy="steps",                                                       # Perform evaluation periodically based on training steps
            eval_steps=0.05,                                                             # Perform evaluation every 5% of total training steps

            # 4. Logging settings                               
            logging_strategy="steps",                                                    # Log metrics periodically based on training steps
            logging_steps=0.05,                                                          # Log metrics every 5% of total training steps
            logging_dir='./results/tensorboard_logs',                                    # Directory for saving TensorBoard logs
            report_to="all",                                                             # Report metrics to all supported platforms (e.g., TensorBoard, WandB)

            # 5. Model checkpoint settings                               
            save_strategy="steps",                                                       # Save model checkpoints periodically based on training steps
            save_steps=0.05,                                                             # Save model checkpoints every 5% of total training steps
            save_total_limit=20,                                                         # Retain the latest 20 model checkpoints on disk
            output_dir='./results/training_checkpoints',                                 # Directory for saving model checkpoints
        ),
    )
    return trainer

#--------------- Training ---------------
# 1) Run model training
def run_training(model, trainer):
    print(f'==> Starting training...')
    model.config.use_cache = False  # Disable the model's caching mechanism during training to recompute hidden states from scratch at each step.
    torch.cuda.empty_cache()        # Clear PyTorch cache to maximize VRAM availability for training.
    trainer.train()                 # Start training. Automatically sets the model to training mode (model.train()).
    torch.cuda.empty_cache()        # Clear PyTorch cache again after training to free up VRAM.
    model.config.use_cache = True   # Re-enable the model's caching mechanism to store and reuse hidden states, speeding up inference.
