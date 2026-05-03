import logging
import os
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

logger = logging.getLogger(__name__)

def build_trainer(
    model,
    config: dict,
    method: str,
    train_dataset,
    eval_dataset,
    tokenizer,
    data_collator,
    compute_metrics_fn
):
    """
    Builds and configures the HuggingFace Trainer.
    It automatically routes the correct hyperparameters based on the chosen method
    (FFT, LoRA, Flex TADA) and sets up DeepSpeed if required.

    Args:
        model: The PyTorch model (prepared by model_factory).
        config (dict): The main YAML configuration dictionary.
        method (str): The tuning method being evaluated.
        train_dataset: The tokenized training dataset.
        eval_dataset: The tokenized validation dataset.
        tokenizer: The tokenizer (for saving with the model).
        data_collator: The data collator for dynamic padding.
        compute_metrics_fn: The function from utils/metrics.py to calculate Acc/F1.

    Returns:
        Trainer: The fully configured HuggingFace Trainer object.
    """
    
    task_name = config["dataset"]["tasks"][0] # For naming the output directory
    method_config = config["methods"].get(method, {})
    train_config = config["training"]
    sys_config = config["system"]

    # 1. Dynamically extract the specific Learning Rate for the chosen method
    # This is crucial because FFT needs ~1e-5, while LoRA needs ~3e-4
    learning_rate = float(method_config.get("learning_rate", 2e-5))
    logger.info(f"Setting Learning Rate to {learning_rate} for method '{method.upper()}'.")

    # 2. Setup Output Directory
    output_dir = os.path.join("outputs", f"{task_name}_{method}_{sys_config.get('seed', 42)}")
    os.makedirs(output_dir, exist_ok=True)

    # 3. Configure Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Core Hyperparameters
        learning_rate=learning_rate,
        per_device_train_batch_size=train_config.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 16),
        num_train_epochs=train_config.get("num_train_epochs", 5),
        weight_decay=train_config.get("weight_decay", 0.01),
        warmup_ratio=train_config.get("warmup_ratio", 0.06),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "linear"),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 1),
        
        # Hardware & Precision (L40S Optimizations)
        fp16=sys_config.get("fp16", False),
        bf16=sys_config.get("bf16", False), # If L40S supports bf16, use it!
        seed=sys_config.get("seed", 42),
        
        # Evaluation & Saving Strategies
        evaluation_strategy=train_config.get("evaluation_strategy", "epoch"),
        save_strategy=train_config.get("save_strategy", "epoch"),
        eval_steps=train_config.get("eval_steps", None),
        save_steps=train_config.get("save_steps", None),
        load_best_model_at_end=train_config.get("load_best_model_at_end", True),
        metric_for_best_model="eval_loss", # We track loss to prevent over-fitting
        greater_is_better=False,
        
        # Logging
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=train_config.get("logging_steps", 50),
        report_to=[sys_config.get("report_to", "none")],
        run_name=f"{task_name}-{method}-seed{sys_config.get('seed', 42)}",
        
        # DeepSpeed Integration (Will only activate if path is provided in YAML)
        deepspeed=train_config.get("deepspeed_config_path", None)
    )

    # 4. Instantiate the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        # Adding Early Stopping to prevent overfitting (especially critical in Few-Shot scenario)
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    return trainer