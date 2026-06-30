import logging
import torch

logger = logging.getLogger(__name__)

def apply_flexible_tada(model: torch.nn.Module, trainable_layers: list) -> torch.nn.Module:
    """
    Applies the Flexible TADA methodology to a pre-trained Transformer model.
    """
    trainable_layers = trainable_layers.copy()
    if not trainable_layers:
        logger.warning("No trainable layers provided for Flexible TADA. The model will be completely frozen!")

    # --- 🌟 NEW: Dynamic Last Layer Detection 🌟 ---
    # Find out how many layers the model has to dynamically resolve "last_layer"
    if "last_layer" in trainable_layers:
        trainable_layers.remove("last_layer")
        
        num_layers = 12 # fallback default
        if hasattr(model.config, "num_hidden_layers"):
            num_layers = model.config.num_hidden_layers
        elif hasattr(model.config, "n_layers"):
            num_layers = model.config.n_layers
            
        last_layer_idx = num_layers - 1
        
        # Add architecture-specific keywords for the last layer
        model_type = getattr(model.config, "model_type", "").lower()
        if "qwen" in model_type or "llama" in model_type:
            dynamic_last_layer = f"layers.{last_layer_idx}."
        elif "deberta" in model_type:
            dynamic_last_layer = f"encoder.layer.{last_layer_idx}."
        else: # BERT, RoBERTa, ELECTRA
            dynamic_last_layer = f"layer.{last_layer_idx}."
            
        trainable_layers.append(dynamic_last_layer)
        logger.info(f"Dynamically resolved 'last_layer' to index {last_layer_idx} for {model_type} architecture.")
    # -----------------------------------------------

    logger.info(f"Applying Flexible TADA. Target keywords to unfreeze: {trainable_layers}")

    # Step 1: Freeze all parameters in the model (The Foundation)
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Step 2: Selectively unfreeze parameters based on the provided keywords
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
        if len(unfrozen_layer_names) > 5:
            logger.debug(f"Unfrozen tensors sample: {unfrozen_layer_names[:3]} ... {unfrozen_layer_names[-2:]}")
        else:
            logger.debug(f"Unfrozen tensors: {unfrozen_layer_names}")

    # Step 4: Ensure the task-specific classification head is ALWAYS unfrozen
    head_keywords = ["classifier", "score", "qa_outputs"]
    for name, param in model.named_parameters():
        if any(head_key in name for head_key in head_keywords):
            if not param.requires_grad:
                param.requires_grad = True
                logger.info(f"Auto-unfrozen classification head parameter: {name}")

    return model