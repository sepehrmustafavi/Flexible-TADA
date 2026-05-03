import torch
import numpy as np
from utils.xai_metrics import calculate_faithfulness, calculate_sufficiency

class XAIEvaluator:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def run_analysis(self, test_dataloader):
        print("🔍 Starting Semantic Robustness Analysis (XAI)...")
        f_scores = []
        s_scores = []

        for batch in test_dataloader:
            important_indices = self.get_important_tokens(batch) 
            
            f_score = calculate_faithfulness(self.model, batch, important_indices, self.device)
            s_score = calculate_sufficiency(self.model, batch, important_indices, self.device)
            
            f_scores.append(f_score)
            s_scores.append(s_score)

        return {
            "avg_faithfulness": float(np.mean(f_scores)),
            "avg_sufficiency": float(np.mean(s_scores))
        }

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