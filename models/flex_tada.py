import logging
import torch

logger = logging.getLogger(__name__)

def apply_flexible_tada(model: torch.nn.Module, trainable_layers: list) -> torch.nn.Module:
    """
    Applies the Flexible TADA methodology to a pre-trained Transformer model.
    
    The core concept: Freeze the entire backbone to retain pre-trained knowledge, 
    but unfreeze specific representation layers (typically the Input Embedding and 
    the Final Transformer Layer) to allow high-level semantic recalibration.

    Args:
        model (torch.nn.Module): The instantiated HuggingFace base model.
        trainable_layers (list): A list of string keywords representing the layers 
                                 that should remain unfrozen 
                                 (e.g., ["embeddings", "layer.11"]).

    Returns:
        torch.nn.Module: The modified model with updated requires_grad attributes.
    """
    
    if not trainable_layers:
        logger.warning("No trainable layers provided for Flexible TADA. The model will be completely frozen!")

    logger.info(f"Applying Flexible TADA. Target layers to unfreeze: {trainable_layers}")

    # Step 1: Freeze all parameters in the model (The Foundation)
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Step 2: Selectively unfreeze parameters based on the provided keywords (The Innovation)
    unfrozen_params_count = 0
    unfrozen_layer_names = []

    for name, param in model.named_parameters():
        # Check if any keyword from our target list is present in the parameter's name
        if any(keyword in name for keyword in trainable_layers):
            param.requires_grad = True
            unfrozen_params_count += param.numel()
            unfrozen_layer_names.append(name)

    # Step 3: Sanity Logging
    if len(unfrozen_layer_names) == 0:
        logger.error("Flexible TADA failed to unfreeze any layers! Check your layer keywords.")
    else:
        logger.info(f"Successfully unfrozen {len(unfrozen_layer_names)} parameter tensors.")
        # Print the first few and last few to keep logs clean
        if len(unfrozen_layer_names) > 5:
            logger.debug(f"Unfrozen tensors sample: {unfrozen_layer_names[:3]} ... {unfrozen_layer_names[-2:]}")
        else:
            logger.debug(f"Unfrozen tensors: {unfrozen_layer_names}")

    # Step 4: Ensure the task-specific classification head is ALWAYS unfrozen
    # Because we are doing a downstream task (GLUE/SuperGLUE), the classifier 
    # (or score layer) must learn to map representations to specific classes.
    head_keywords = ["classifier", "score", "qa_outputs"]
    for name, param in model.named_parameters():
        if any(head_key in name for head_key in head_keywords):
            if not param.requires_grad:
                param.requires_grad = True
                logger.info(f"Auto-unfrozen classification head parameter: {name}")

    return model