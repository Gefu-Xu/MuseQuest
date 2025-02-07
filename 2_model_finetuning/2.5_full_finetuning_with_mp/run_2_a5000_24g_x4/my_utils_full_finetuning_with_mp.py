#--------------- Functions in Full Fine-Tuning with Model Parallel ---------------
# 1) Create Custom Device Map for Model Parallel
def create_device_map_mp_2_gpu():
    custom_device_map={
        'model.vision_model': 0,
        'model.connector': 0,
        'model.text_model.embed_tokens': 0,
        'model.text_model.layers.0': 0,
        'model.text_model.layers.1': 0,
        'model.text_model.layers.2': 0,
        'model.text_model.layers.3': 0,
        'model.text_model.layers.4': 0,
        'model.text_model.layers.5': 0,
        'model.text_model.layers.6': 0,
        'model.text_model.layers.7': 0,
        'model.text_model.layers.8': 0,
        'model.text_model.layers.9': 0,
        'model.text_model.layers.10': 0,
        'model.text_model.layers.11': 0,
        'model.text_model.layers.12': 0,
        'model.text_model.layers.13': 1,
        'model.text_model.layers.14': 1,
        'model.text_model.layers.15': 1,
        'model.text_model.layers.16': 1,
        'model.text_model.layers.17': 1,
        'model.text_model.layers.18': 1,
        'model.text_model.layers.19': 1,
        'model.text_model.layers.20': 1,
        'model.text_model.layers.21': 1,
        'model.text_model.layers.22': 1,
        'model.text_model.layers.23': 1,
        'model.text_model.layers.24': 1,
        'model.text_model.layers.25': 1,
        'model.text_model.layers.26': 1,
        'model.text_model.layers.27': 1,
        'model.text_model.layers.28': 1,
        'model.text_model.layers.29': 1,
        'model.text_model.layers.30': 1,
        'model.text_model.layers.31': 1,
        'model.text_model.norm': 1,
        'lm_head': 1
    }
    return custom_device_map

def create_device_map_mp_4_gpu():
    custom_device_map={
        'model.vision_model': 0,
        'model.connector': 0,
        'model.text_model.embed_tokens': 0,
        'model.text_model.layers.0': 0,
        'model.text_model.layers.1': 0,
        'model.text_model.layers.2': 0,
        'model.text_model.layers.3': 0,
        'model.text_model.layers.4': 0,
        'model.text_model.layers.5': 1,
        'model.text_model.layers.6': 1,
        'model.text_model.layers.7': 1,
        'model.text_model.layers.8': 1,
        'model.text_model.layers.9': 1,
        'model.text_model.layers.10': 1,
        'model.text_model.layers.11': 1,
        'model.text_model.layers.12': 1,
        'model.text_model.layers.13': 1,
        'model.text_model.layers.14': 2,
        'model.text_model.layers.15': 2,
        'model.text_model.layers.16': 2,
        'model.text_model.layers.17': 2,
        'model.text_model.layers.18': 2,
        'model.text_model.layers.19': 2,
        'model.text_model.layers.20': 2,
        'model.text_model.layers.21': 2,
        'model.text_model.layers.22': 2,
        'model.text_model.layers.23': 3,
        'model.text_model.layers.24': 3,
        'model.text_model.layers.25': 3,
        'model.text_model.layers.26': 3,
        'model.text_model.layers.27': 3,
        'model.text_model.layers.28': 3,
        'model.text_model.layers.29': 3,
        'model.text_model.layers.30': 3,
        'model.text_model.layers.31': 3,
        'model.text_model.norm': 3,
        'lm_head': 3
    }
    return custom_device_map

# 2) Select Layers for Full Fine-Tuning
def select_layers_for_full_fine_tuning(model, module_list):
    print("==> Selecting Layers for Full Fine-Tuning...")
    for name, param in model.named_parameters():
        param.requires_grad = False               # Freeze all parameters by default
        for module in module_list:
            if module in name and 'proj' in name: # Set linear('proj') layers in the specified modules to trainable
                param.requires_grad = True
                break
    return model
    
