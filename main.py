import argparse
import yaml
import logging
import torch
import os
from transformers import AutoTokenizer, set_seed

# Import our custom modular components
from utils.logger import setup_logger
from data.dataset_builder import FlexibleTADADatasetBuilder
from data.data_utils import FlexibleTADADataProcessor
from models import get_model
from utils.metrics import build_compute_metrics_fn
from trainer.engine import build_trainer
from trainer.evaluator import run_evaluation

logger = logging.getLogger(__name__)

def parse_args():
    """Parses command-line arguments to override YAML configurations dynamically."""
    parser = argparse.ArgumentParser(description="Flexible TADA Evaluation Pipeline")
    
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the YAML configuration file (e.g., configs/roberta_glue.yaml)")
    parser.add_argument("--method", type=str, required=True, 
                        choices=["fft", "lora", "static_tada", "flex_tada"],
                        help="Tuning method to apply")
    parser.add_argument("--task", type=str, default=None, 
                        help="Specific task to run (overrides the YAML task list)")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed (overrides the YAML seed, crucial for few-shot runs)")
    parser.add_argument("--few_shot", type=int, default=None, 
                        help="Number of training samples for the low-data regime scenario")
    
    return parser.parse_args()

def main():
    # 1. Parse Arguments & Load Configuration
    args = parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 2. Dynamic Overrides from Command Line
    if args.task:
        config["dataset"]["tasks"] = [args.task.lower()]
    if args.seed is not None:
        config["system"]["seed"] = args.seed

    task_name = config["dataset"]["tasks"][0]
    seed = config["system"].get("seed", 42)
    
    # 3. Setup Global Logger
    # Creates a dedicated log file for this specific task and method
    log_dir = os.path.join("outputs", f"{task_name}_{args.method}_{seed}", "logs")
    setup_logger(output_dir=log_dir)
    logger.info("="*50)
    logger.info(f"🚀 Starting Flexible-TADA Pipeline")
    logger.info(f"Task: {task_name.upper()} | Method: {args.method.upper()} | Seed: {seed}")
    if args.few_shot:
        logger.info(f"Mode: FEW-SHOT ({args.few_shot} training samples)")
    logger.info("="*50)

    # 4. Set Reproducibility Seed
    set_seed(seed)
    
    # 5. Initialize Tokenizer
    model_path = config["model"]["model_name_or_path"]
    logger.info(f"Loading Tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"].get("tokenizer_name", model_path),
        trust_remote_code=config["model"].get("trust_remote_code", False)
    )

    # 6. Build and Tokenize Dataset
    dataset_builder = FlexibleTADADatasetBuilder(config)
    raw_dataset = dataset_builder.load_task(task_name, num_train_samples=args.few_shot)
    
    data_processor = FlexibleTADADataProcessor(
        tokenizer=tokenizer, 
        max_seq_length=config["model"].get("max_seq_length", 128)
    )
    tokenized_dataset = data_processor.prepare_dataset(raw_dataset, task_name)
    data_collator = data_processor.get_data_collator()

    # Determine number of labels for the classification head
    # (HuggingFace datasets usually have a 'label' feature we can inspect)
    # Determine number of labels for the classification head
    if task_name.lower() == "stsb":
        num_labels = 1  # Regression task
    elif "label" in raw_dataset["train"].features:
        # Get the number of classes directly from the dataset features for classification tasks
        num_labels = raw_dataset["train"].features["label"].num_classes
    else:
        num_labels = 2  # Fallback default

    # 7. Build Model via Factory (Applies the PEFT / TADA logic)
    model = get_model(config, args.method, num_labels, tokenizer)

    # 8. Setup Evaluation Metrics
    compute_metrics_fn = build_compute_metrics_fn(task_name)

    # 9. Instantiate the Trainer Engine
    trainer = build_trainer(
        model=model,
        config=config,
        method=args.method,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics_fn=compute_metrics_fn
    )

    # 10. Execute Training
    logger.info("Starting Training Phase...")
    train_result = trainer.train()
    
    # Save the final optimized model (and LoRA weights if applicable)
    trainer.save_model()
    logger.info(f"Training completed. Global Step: {train_result.global_step}")

    # گرفتن متریک‌های زمان آموزش
    train_metrics = train_result.metrics

    # 11. Execute Evaluation & Hardware Profiling
    logger.info("Starting Evaluation Phase...")
    output_dir = trainer.args.output_dir
    run_evaluation(
        trainer=trainer,
        eval_dataset=tokenized_dataset["validation"],
        task_name=task_name,
        method_name=args.method,
        output_dir=output_dir,
        train_metrics=train_metrics 
    )

    logger.info(f"🎉 Pipeline finished successfully for {task_name.upper()} using {args.method.upper()}.")

if __name__ == "__main__":
    main()