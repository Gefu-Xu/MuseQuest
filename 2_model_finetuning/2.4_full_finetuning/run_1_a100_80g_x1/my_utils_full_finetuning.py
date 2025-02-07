#--------------- Functions in Full Fine-Tuning ---------------
# 1) Select Layers for Full Fine-Tuning
def select_layers_for_full_fine_tuning(model, module_list):
    print("==> Selecting Layers for Full Fine-Tuning...")
    for name, param in model.named_parameters():
        param.requires_grad = False               # Freeze all parameters by default
        for module in module_list:
            if module in name and 'proj' in name: # Set linear('proj') layers in the specified modules to trainable
                param.requires_grad = True
                break
    return model
    
