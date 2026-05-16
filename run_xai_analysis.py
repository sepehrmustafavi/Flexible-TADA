import argparse
import json
import os
import torch
from transformers import AutoTokenizer
from models import get_model
from trainer.xai_evaluator import XAIEvaluator
from data.dataset_builder import FlexibleTADADatasetBuilder
from data.data_utils import FlexibleTADADataProcessor
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    # 1. Load Config & Setup Device
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Load the Best Saved Model and Tokenizer
    print(f"Loading trained model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # Re-initialize the model architecture using our factory, but load the saved weights
    model = get_model(config, args.method, num_labels=2, tokenizer=tokenizer)
    
    # Load state dict (Assuming standard HF saving)
    state_dict_path = os.path.join(args.model_path, "pytorch_model.bin")
    if os.path.exists(state_dict_path):
        model.load_state_dict(torch.load(state_dict_path, map_location=device), strict=False)
    
    model.to(device)

    # 3. Load Only a Subset of Test Data (e.g., 500 samples for speed)
    dataset_builder = FlexibleTADADatasetBuilder(config)
    raw_dataset = dataset_builder.load_task(args.task, num_train_samples=10) # train doesn't matter here
    
    processor = FlexibleTADADataProcessor(tokenizer, max_seq_length=128)
    tokenized_dataset = processor.prepare_dataset(raw_dataset, args.task)
    
    # Take 500 samples from validation for XAI (to save time on L40S)
    xai_dataset = tokenized_dataset["validation"].select(range(min(500, len(tokenized_dataset["validation"]))))
    
    from torch.utils.data import DataLoader
    test_dataloader = DataLoader(xai_dataset, batch_size=16, collate_fn=processor.get_data_collator())

    # 4. Run Evaluator
    evaluator = XAIEvaluator(model, tokenizer, device=device)
    results = evaluator.run_analysis(test_dataloader, args.task, args.method, args.output_dir)

    # 5. Save Results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"xai_{args.task}_{args.method}.json")
    
    results["task"] = args.task
    results["method"] = args.method
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"✅ XAI Results saved to {output_file}")

if __name__ == "__main__":
    main()