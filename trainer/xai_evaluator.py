import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from captum.attr import LayerIntegratedGradients
from utils.xai_metrics import calculate_faithfulness, calculate_sufficiency

class XAIEvaluator:
    """
    Performs Explainable AI (XAI) analysis on trained Transformer models.

    This class generates token-level explanations using Integrated Gradients,
    computes explanation quality metrics such as Faithfulness and Sufficiency,
    and visualizes token importance through attribution heatmaps for model
    interpretability and semantic robustness analysis.
    """
    def __init__(self, model, tokenizer, device="cuda"):
        """
        Initializes the XAI evaluator.

        The evaluator stores the trained model, tokenizer, and computation device
        required for generating token attributions and evaluating explanation
        quality metrics.

        Args:
            model (torch.nn.Module): Fine-tuned Transformer model.
            tokenizer: HuggingFace tokenizer corresponding to the model.
            device (str): Device used for computation (e.g., "cuda" or "cpu").

        Returns:
            None
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.last_raw_attributions = None

    def run_analysis(self, test_dataloader, task_name: str, method_name: str, output_dir: str):
        """
        Performs semantic robustness analysis on a test dataset.

        This method generates Integrated Gradients explanations for each batch,
        computes Faithfulness and Sufficiency metrics, produces a token
        attribution heatmap for the first processed sample, and returns the
        average and standard deviation of the explanation quality metrics.

        Args:
            test_dataloader: DataLoader containing the evaluation dataset.
            task_name (str): Name of the benchmark task being evaluated.
            method_name (str): Name of the fine-tuning method.
            output_dir (str): Directory for saving generated heatmaps.

        Returns:
            dict: Dictionary containing the mean and standard deviation of
            Faithfulness and Sufficiency scores.
        """
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
        """
        Generates a token attribution heatmap for a single input sample.

        The heatmap visualizes the normalized Integrated Gradients attribution
        scores assigned to each input token, providing an intuitive explanation
        of the model's prediction.

        Args:
            batch (dict): Batch containing tokenized input data.
            task_name (str): Name of the benchmark task.
            method_name (str): Name of the fine-tuning method.
            output_dir (str): Directory where the heatmap image is saved.

        Returns:
            None
        """
        os.makedirs(output_dir, exist_ok=True)
        
        input_ids = batch["input_ids"][0]
        actual_len = batch["attention_mask"][0].sum().item()
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[:actual_len])
        
        importance_scores = self.last_raw_attributions[0][:actual_len]
        
        max_abs = np.max(np.abs(importance_scores))
        if max_abs > 0:
            importance_scores = importance_scores / max_abs

        # Graphical Setting
        plt.figure(figsize=(14, 2))
        
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
        Identifies the most influential input tokens using Integrated Gradients.

        This method computes token-level attribution scores with Captum's
        LayerIntegratedGradients, filters out special tokens, and selects the
        top 20% most influential tokens based on attribution magnitude. The
        resulting token indices are used for evaluating explanation quality
        and generating attribution heatmaps.

        Args:
            batch (dict): Batch containing tokenized inputs and attention masks.

        Returns:
            list: A list containing the indices of the most important tokens
            for each sample in the batch.
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
        def custom_forward(inputs, mask):
            return self.model(input_ids=inputs, attention_mask=mask).logits

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
            token_attributions = attributions.sum(dim=-1).squeeze(0)
            
            # Step 6: Filter out special tokens from being selected in the rationale
            token_attributions_abs = token_attributions.abs().clone()
            token_attributions_abs[0] = 0.0
            token_attributions_abs[actual_len - 1] = 0.0
            if actual_len < seq_len:
                token_attributions_abs[actual_len:] = 0.0

            # Step 7: Select top 20% most important tokens based on absolute attribution magnitude
            num_to_select = max(1, int((actual_len - 2) * 0.2))
            _, top_indices = torch.topk(token_attributions_abs, num_to_select)
            
            important_indices.append(top_indices.cpu().tolist())
            raw_attributions_list.append(token_attributions.detach().cpu().numpy())

        self.last_raw_attributions = raw_attributions_list
        
        return important_indices
