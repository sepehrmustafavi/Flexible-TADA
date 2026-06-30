import logging
import torch

logger = logging.getLogger(__name__)

def apply_static_tada(model: torch.nn.Module, trainable_layers: list) -> torch.nn.Module:
    """
    Applies the Static TADA (Baseline) methodology to a pre-trained Transformer model.
    
    In the static version, ONLY the input embeddings and the task-specific 
    classification head are updated. The entire transformer backbone is frozen.
    This method often suffers from Model Collapse when fine-tuning on complex 
    reasoning tasks because the high-level semantic representations cannot adjust.

    Args:
        model (torch.nn.Module): The instantiated HuggingFace base model.
        trainable_layers (list): Keywords representing the embedding layers 
                                 (e.g., ["embeddings"] for RoBERTa, 
                                 ["embed_tokens"] for Qwen).

    Returns:
        torch.nn.Module: The modified model with updated requires_grad attributes.
    """
    
    if not trainable_layers:
        logger.warning("No trainable layers provided for Static TADA. Expected embedding layer names.")

    logger.info(f"Applying Static TADA (Baseline). Target embedding layers: {trainable_layers}")

    # Step 1: Freeze all parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Step 2: Unfreeze ONLY the embedding layers based on the provided keywords
    unfrozen_params_count = 0
    unfrozen_layer_names = []

    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trainable_layers):
            param.requires_grad = True
            unfrozen_params_count += param.numel()
            unfrozen_layer_names.append(name)

    # Logging embedding unfreezing status
    if len(unfrozen_layer_names) == 0:
        logger.error("Static TADA failed to unfreeze any embeddings! Check the layer keywords in YAML.")
    else:
        logger.info(f"Successfully unfrozen {len(unfrozen_layer_names)} embedding tensors.")
        logger.debug(f"Unfrozen embedding tensors: {unfrozen_layer_names}")

    # Step 3: Ensure the classification head is unfrozen
    # Even in baseline TADA, the downstream task head must be trained.
    head_keywords = ["classifier", "score", "qa_outputs"]
    for name, param in model.named_parameters():
        if any(head_key in name for head_key in head_keywords):
            if not param.requires_grad:
                param.requires_grad = True
                logger.info(f"Auto-unfrozen classification head parameter: {name}")

    return model