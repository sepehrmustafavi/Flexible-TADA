import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils.xai_metrics import calculate_faithfulness, calculate_sufficiency

class XAIEvaluator:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def run_analysis(self, test_dataloader, task_name: str, method_name: str, output_dir: str):
        print("🔍 Starting Semantic Robustness Analysis (XAI)...")
        f_scores = []
        s_scores = []
        heatmap_generated = False  

        for batch in test_dataloader:
            important_indices = self.get_important_tokens(batch) 
            
            if not heatmap_generated and len(important_indices[0]) > 0:
                self.generate_heatmap(batch, important_indices, task_name, method_name, output_dir)
                heatmap_generated = True

            f_score = calculate_faithfulness(self.model, batch, important_indices, self.device)
            s_score = calculate_sufficiency(self.model, batch, important_indices, self.device)
            
            f_scores.append(f_score)
            s_scores.append(s_score)

        return {
            "avg_faithfulness": float(np.mean(f_scores)),
            "avg_sufficiency": float(np.mean(s_scores))
        }

    def generate_heatmap(self, batch, important_indices, task_name, method_name, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        input_ids = batch["input_ids"][0]
        actual_len = batch["attention_mask"][0].sum().item()
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[:actual_len])
        importance_scores = np.zeros(actual_len)
        
        for idx in important_indices[0]:
            if idx < actual_len:
                importance_scores[idx] = 1.0

        plt.figure(figsize=(14, 2))
        sns.heatmap([importance_scores], annot=[tokens], fmt="", cmap="Reds", 
                    cbar=False, xticklabels=False, yticklabels=False, annot_kws={"size": 10})
        plt.title(f"Token Importance Heatmap - {task_name.upper()} ({method_name.upper()})", pad=20)
        
        save_path = os.path.join(output_dir, f"heatmap_{task_name}_{method_name}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"📊 Heatmap successfully saved to {save_path}")

    def get_important_tokens(self, batch):
        """
        Extracts indices of important tokens.
        In a production setting, this should use Integrated Gradients.
        For this baseline, we simulate it by selecting a subset of tokens,
        ensuring we DO NOT select special tokens (like [CLS] or [SEP]).
        """
        batch_size, seq_len = batch["input_ids"].shape
        important_indices = []
        
        for i in range(batch_size):
            # Find the actual length of the sequence ignoring padding
            actual_len = batch["attention_mask"][i].sum().item()
            
            # Skip the first token [CLS] and the last token [SEP]
            if actual_len > 3:
                # Select top 20% of the actual words
                num_to_select = max(1, int((actual_len - 2) * 0.2))
                # For baseline simulation, we pick indices from the middle of the sentence
                start_idx = 1
                end_idx = start_idx + num_to_select
                important_indices.append(list(range(start_idx, end_idx)))
            else:
                important_indices.append([])
                
        return important_indices