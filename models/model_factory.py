import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType

# We will implement these two functions in the next steps
from .flex_tada import apply_flexible_tada
from .baselines import apply_static_tada

logger = logging.getLogger(__name__)

def get_model(config: dict, method: str, num_labels: int, tokenizer):
    """
    Factory function to instantiate the correct model architecture based on the chosen method.

    Args:
        config (dict): The complete configuration dictionary loaded from YAML.
        method (str): The chosen tuning method ('fft', 'lora', 'static_tada', 'flex_tada').
        num_labels (int): Number of classification labels for the downstream task.
        tokenizer: The loaded tokenizer (needed to set pad_token_id for LLMs).

    Returns:
        torch.nn.Module: The prepared PyTorch model ready for training.
    """
    model_name_or_path = config["model"]["model_name_or_path"]
    method = method.lower()
    
    logger.info(f"Loading base model '{model_name_or_path}' with {num_labels} labels...")

    # 1. Load Base Configuration
    model_config = AutoConfig.from_pretrained(
        model_name_or_path, 
        num_labels=num_labels,
        trust_remote_code=config["model"].get("trust_remote_code", False)
    )

    # 2. Fix Padding Token for LLMs (like Qwen2)
    # Generative models usually don't have a default pad token, which causes errors
    # in AutoModelForSequenceClassification. We sync it with the tokenizer.
    if tokenizer.pad_token_id is not None:
        model_config.pad_token_id = tokenizer.pad_token_id
    else:
        model_config.pad_token_id = tokenizer.eos_token_id

    # 3. Load the Base Model Weights
    # We use AutoModelForSequenceClassification because all our GLUE/SuperGLUE 
    # tasks are formulated as classification or regression.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=model_config,
        trust_remote_code=config["model"].get("trust_remote_code", False),
        # If using fp16/bf16 on L40S, we can load the model in half-precision to save memory
        torch_dtype=torch.bfloat16 if config["system"].get("fp16", False) else torch.float32
    )

    logger.info(f"Applying method: {method.upper()}...")

    # 4. Route to the correct tuning method
    method_config = config["methods"].get(method, {})

    if method == "fft":
        # Full Fine-Tuning: All parameters require gradients (Default behavior)
        logger.info("Full Fine-Tuning selected. All parameters are trainable.")
        
        # If Gradient Checkpointing is enabled for FFT (critical for Qwen on L40S)
        if config["training"].get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
            logger.info("Gradient Checkpointing enabled for FFT.")

    elif method == "lora":
        # Low-Rank Adaptation using HuggingFace PEFT library
        logger.info(f"Applying LoRA with r={method_config.get('r', 8)}")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=method_config.get("r", 8),
            lora_alpha=method_config.get("lora_alpha", 16),
            lora_dropout=method_config.get("lora_dropout", 0.1),
            target_modules=method_config.get("target_modules", ["query", "value"])
        )
        model = get_peft_model(model, peft_config)

    elif method == "static_tada":
        # Baseline TADA: Only unfreeze the embedding layer
        trainable_layers = method_config.get("trainable_layers", [])
        model = apply_static_tada(model, trainable_layers)

    elif method == "flex_tada":
        # Our Proposed Method: Unfreeze embeddings + the last transformer layer
        trainable_layers = method_config.get("trainable_layers", [])
        model = apply_flexible_tada(model, trainable_layers)

    else:
        raise ValueError(f"Method '{method}' is not supported. Choose from: fft, lora, static_tada, flex_tada.")

    # 5. Final Sanity Check: Print Trainable Parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable Parameters: {trainable_params:,d} || "
        f"All Parameters: {all_params:,d} || "
        f"Trainable Ratio: {100 * trainable_params / all_params:.4f}%"
    )

    return model