import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from captum.attr import LayerIntegratedGradients
from utils.xai_metrics import calculate_faithfulness, calculate_sufficiency

class XAIEvaluator:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # We will store the raw attributions of the last batch here 
        # to generate accurate, gradient-based heatmaps.
        self.last_raw_attributions = None

    def run_analysis(self, test_dataloader, task_name: str, method_name: str, output_dir: str):
        print("🔍 Starting Semantic Robustness Analysis (XAI) using Integrated Gradients...")
        f_scores = []
        s_scores = []
        heatmap_generated = False  

        for batch in test_dataloader:
            # 1. Get important indices AND compute real IG attributions
            important_indices = self.get_important_tokens(batch) 
            
            # 2. Generating the First batch heatmap using real attributions
            if not heatmap_generated and len(important_indices[0]) > 0:
                self.generate_heatmap(batch, task_name, method_name, output_dir)
                heatmap_generated = True

            # 3. Calculate Faithfulness and Sufficiency metrics
            # These now return a list of scores for every sample in the current batch
            f_batch_scores = calculate_faithfulness(self.model, batch, important_indices, self.device)
            s_batch_scores = calculate_sufficiency(self.model, batch, important_indices, self.device)
            
            # 4. Flatten the lists to collect scores for the entire dataset
            f_scores.extend(f_batch_scores)
            s_scores.extend(s_batch_scores)

        # Return both Mean and Standard Deviation (Std) across all validation samples
        return {
            "avg_faithfulness": float(np.mean(f_scores)),
            "std_faithfulness": float(np.std(f_scores)),
            "avg_sufficiency": float(np.mean(s_scores)),
            "std_sufficiency": float(np.std(s_scores))
        }

    def generate_heatmap(self, batch, task_name, method_name, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract the sequence length and tokens for the first sample in the batch
        input_ids = batch["input_ids"][0]
        actual_len = batch["attention_mask"][0].sum().item()
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[:actual_len])
        
        # Retrieve the raw attribution scores calculated by Captum
        importance_scores = self.last_raw_attributions[0][:actual_len]
        
        # Normalize scores to [-1, 1] for optimal visualization with diverging colormap
        max_abs = np.max(np.abs(importance_scores))
        if max_abs > 0:
            importance_scores = importance_scores / max_abs

        # Graphical Setting
        plt.figure(figsize=(14, 2))
        
        # Using 'coolwarm' cmap: Red for positive impact, Blue for negative impact, White for neutral
        sns.heatmap([importance_scores], annot=[tokens], fmt="", cmap="coolwarm", 
                    vmin=-1, vmax=1, center=0, cbar=False, 
                    xticklabels=False, yticklabels=False, annot_kws={"size": 10})
        
        plt.title(f"Token Attribution (IG) - {task_name.upper()} ({method_name.upper()})", pad=20)
        
        save_path = os.path.join(output_dir, f"heatmap_{task_name}_{method_name}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"📊 Heatmap successfully saved to {save_path}")

    def get_important_tokens(self, batch):
        """
        Extracts indices of important tokens using Captum's LayerIntegratedGradients.
        This provides a mathematically rigorous attribution map instead of random selection.
        """
        self.model.eval()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        batch_size, seq_len = input_ids.shape

        important_indices = []
        raw_attributions_list = []

        # Step 1: Forward pass to determine the model's predicted class
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_classes = outputs.logits.argmax(dim=-1)

        # Step 2: Setup Captum LayerIntegratedGradients
        # We wrap the model's forward pass to strictly take input_ids
        def custom_forward(inputs, mask):
            return self.model(input_ids=inputs, attention_mask=mask).logits

        # Attach the hook to the model's embedding layer
        embedding_layer = self.model.get_input_embeddings()
        lig = LayerIntegratedGradients(custom_forward, embedding_layer)

        for i in range(batch_size):
            single_input_ids = input_ids[i:i+1]
            single_mask = attention_mask[i:i+1]
            target_class = predicted_classes[i].item()

            actual_len = single_mask.sum().item()

            if actual_len <= 3:
                important_indices.append([])
                raw_attributions_list.append(np.zeros(seq_len))
                continue

            # Step 3: Define the Baseline (The uninformative state)
            # For NLP, a sequence of PAD tokens is the standard baseline
            baseline_ids = torch.full_like(single_input_ids, self.tokenizer.pad_token_id)

            # Step 4: Calculate Attributions
            # We enable gradients strictly for the IG computation
            with torch.enable_grad():
                attributions = lig.attribute(
                    inputs=single_input_ids,
                    baselines=baseline_ids,
                    target=target_class,
                    additional_forward_args=(single_mask,),
                    n_steps=20, # 20 steps provide a solid balance between accuracy and speed
                    return_convergence_delta=False
                )

            # Step 5: Summarize attributions across the embedding dimension
            # We sum across the hidden dim to get a single score per token
            token_attributions = attributions.sum(dim=-1).squeeze(0)
            
            # Step 6: Filter out special tokens from being selected in the rationale
            token_attributions_abs = token_attributions.abs().clone()
            token_attributions_abs[0] = 0.0 # Ignore [CLS] / <s>
            token_attributions_abs[actual_len - 1] = 0.0 # Ignore [SEP] / </s>
            if actual_len < seq_len:
                token_attributions_abs[actual_len:] = 0.0 # Ignore [PAD]

            # Step 7: Select top 20% most important tokens based on absolute attribution magnitude
            num_to_select = max(1, int((actual_len - 2) * 0.2))
            _, top_indices = torch.topk(token_attributions_abs, num_to_select)
            
            important_indices.append(top_indices.cpu().tolist())
            raw_attributions_list.append(token_attributions.detach().cpu().numpy())

        # Cache the raw attributions for potential heatmap generation
        self.last_raw_attributions = raw_attributions_list
        
        return important_indices