import os
import json
import glob
import torch
import yaml
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Import our custom modules
from models import get_model
from utils.cka_metrics import calculate_layerwise_cka
from data.dataset_builder import FlexibleTADADatasetBuilder
from data.data_utils import FlexibleTADADataProcessor

def find_latest_checkpoint(base_dir):
    """
    Finds the most recent HuggingFace Trainer checkpoint.

    This function searches the specified model directory for checkpoint
    folders, identifies the checkpoint with the highest training step,
    and returns its path. If no checkpoint directories are found, the
    base directory is returned instead.

    Args:
        base_dir (str): Directory containing the model checkpoints.

    Returns:
        str: Path to the latest checkpoint or the base directory if no
        checkpoints exist.
    """
    checkpoints = glob.glob(os.path.join(base_dir, "checkpoint-*"))
    if not checkpoints:
        return base_dir # Fallback to the base directory if no checkpoint folders exist
    # Sort by step number to get the latest one
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoints[-1]

def main():
    """
    Performs layer-wise representation analysis using Linear CKA.

    This script loads trained models produced by different fine-tuning
    methods, extracts their hidden representations on a validation dataset,
    computes layer-wise Linear CKA similarity against a Fully Fine-Tuned
    reference model, and saves the resulting representation similarity
    scores for quantitative analysis.

    Returns:
        None
    """
    print("🚀 Starting Representation Analysis (CKA)...")
    
    # Configuration Settings
    config_path = "configs/roberta_glue.yaml"
    task = "sst2"       # Using SST-2 or MNLI is highly recommended for reasoning tasks
    seed = 42
    num_samples = 256   # A batch of 256 is statistically robust for CKA calculation

    # 1. Load Config & Setup Device
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Using device: {device}")

    # 2. Prepare Dataset and Dataloader
    print(f"📚 Loading dataset for task: {task.upper()}...")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name_or_path"])
    builder = FlexibleTADADatasetBuilder(config)
    raw_dataset = builder.load_task(task, num_train_samples=num_samples) 
    
    processor = FlexibleTADADataProcessor(tokenizer, max_seq_length=128)
    tokenized_dataset = processor.prepare_dataset(raw_dataset, task)
    
    # use a single batch equal to num_samples for robust CKA matrix computation
    val_subset = tokenized_dataset["validation"].select(range(min(num_samples, len(tokenized_dataset["validation"]))))
    dataloader = DataLoader(
        val_subset, 
        batch_size=num_samples, 
        collate_fn=processor.get_data_collator()
    )

    # Helper function to initialize and load trained models
    def load_trained_model(method):
        """
        Loads a trained model for the specified fine-tuning method.

        This helper function initializes the appropriate model architecture,
        loads the latest available checkpoint, restores the trained weights,
        and prepares the model for evaluation.

        Args:
            method (str): Name of the fine-tuning method (e.g., "fft",
                "static_tada", or "flex_tada").

        Returns:
            torch.nn.Module: The trained model loaded onto the target device.
        """
        print(f"📦 Loading {method.upper()} model...")
        model_dir = f"outputs/{task}_{method}_{seed}"
        ckpt_dir = find_latest_checkpoint(model_dir)
        
        # Determine num_labels dynamically based on the task
        num_labels = 3 if task == "mnli" else (1 if task == "stsb" else 2)
        
        model = get_model(config, method, num_labels=num_labels, tokenizer=tokenizer)
        
        # Load the trained weights (handling both standard .bin and .safetensors formats)
        bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
        safe_path = os.path.join(ckpt_dir, "model.safetensors")
        
        if os.path.exists(bin_path):
            model.load_state_dict(torch.load(bin_path, map_location=device), strict=False)
        elif os.path.exists(safe_path):
            from safetensors.torch import load_file
            model.load_state_dict(load_file(safe_path, device=device), strict=False)
        else:
            print(f"⚠️ Warning: No trained weights found in {ckpt_dir}. Using base initialized weights.")
            
        model.to(device)
        return model

    # 3. Load Models
    model_fft = load_trained_model("fft")
    model_static = load_trained_model("static_tada")
    model_flex = load_trained_model("flex_tada")

    # 4. Calculate Layer-wise CKA
    print("\n🧠 Calculating Layer-wise CKA: FFT vs Static TADA (Proving Collapse)...")
    cka_static = calculate_layerwise_cka(model_fft, model_static, dataloader, device)

    print("🧠 Calculating Layer-wise CKA: FFT vs Flex TADA (Proving Recalibration)...")
    cka_flex = calculate_layerwise_cka(model_fft, model_flex, dataloader, device)

    # 5. Save Results
    results = {
        "task": task,
        "evaluated_samples": len(val_subset),
        "layer_indices": list(range(len(cka_static))),
        "cka_fft_vs_static": cka_static,
        "cka_fft_vs_flex": cka_flex
    }

    os.makedirs("outputs/representation_analysis", exist_ok=True)
    out_file = f"outputs/representation_analysis/cka_{task}.json"
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n✅ Representation Analysis Complete! Results successfully saved to: {out_file}")

if __name__ == "__main__":
    main()
