import logging
import os
import json
import time
import torch

logger = logging.getLogger(__name__)

def run_evaluation(trainer, eval_dataset, task_name: str, method_name: str, output_dir: str):
    """
    Runs the final evaluation loop, calculates critical metrics including 
    inference latency, and dumps the results into a clean JSON file 
    for easy aggregation into the paper's tables.

    Args:
        trainer: The instantiated HuggingFace Trainer.
        eval_dataset: The tokenized validation/test dataset.
        task_name (str): The name of the task (e.g., 'mnli', 'boolq').
        method_name (str): The method used (e.g., 'lora', 'flex_tada').
        output_dir (str): The directory to save the JSON results.

    Returns:
        dict: A dictionary containing all computed metrics.
    """
    logger.info(f"========== Starting Evaluation ==========")
    logger.info(f"Task: {task_name.upper()} | Method: {method_name.upper()}")

    # Step 1: GPU Warm-up (Crucial for accurate latency measurement)
    # CUDA requires a few forward passes to initialize its graphs. 
    # If we don't warm up, the first few batches will artificially inflate the latency time.
    logger.info("Warming up GPU for accurate latency measurement...")
    warmup_subset = eval_dataset.select(range(min(16, len(eval_dataset))))
    _ = trainer.predict(warmup_subset)

    # Step 2: Measure Inference Time and Compute Standard Metrics
    torch.cuda.synchronize() # Wait for all preceding CUDA commands to finish
    start_time = time.time()
    
    # trainer.evaluate() automatically calls our custom compute_metrics_fn 
    # (which we will define in utils/metrics.py)
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    
    torch.cuda.synchronize() # Wait for evaluation to finish completely
    end_time = time.time()

    # Step 3: Calculate Latency per Sample (Green AI Metric for Table 4)
    total_samples = len(eval_dataset)
    total_time_ms = (end_time - start_time) * 1000
    latency_per_sample_ms = total_time_ms / total_samples

    # Inject our custom hardware metrics into the HuggingFace metrics dictionary
    metrics["eval_latency_per_sample_ms"] = round(latency_per_sample_ms, 4)
    metrics["eval_total_samples"] = total_samples

    logger.info(f"Evaluation completed. Inference Latency: {latency_per_sample_ms:.2f} ms/sample")
    logger.info(f"Primary Metric (e.g., Accuracy): {metrics.get('eval_accuracy', metrics.get('eval_f1', 'N/A'))}")

    # Step 4: Clean up and Save Results for Paper Tables
    # We remove the 'eval_' prefix from the keys for a cleaner JSON structure
    cleaned_metrics = {k.replace("eval_", ""): v for k, v in metrics.items()}
    cleaned_metrics["task"] = task_name
    cleaned_metrics["method"] = method_name
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"results_{task_name}_{method_name}.json")
    
    with open(results_path, "w") as f:
        json.dump(cleaned_metrics, f, indent=4)
        
    logger.info(f"Results successfully saved to {results_path}")
    logger.info(f"=========================================")

    return cleaned_metrics