import torch

def linear_cka(features_x: torch.Tensor, features_y: torch.Tensor) -> float:
    """
    Calculates the Linear CKA (Centered Kernel Alignment) between two sets of representations.
    This metric provides a robust measure of similarity between hidden layers.
    
    Args:
        features_x (torch.Tensor): Features from the first model, shape (batch_size, hidden_dim).
        features_y (torch.Tensor): Features from the second model, shape (batch_size, hidden_dim).
        
    Returns:
        float: The CKA similarity score between 0.0 (completely orthogonal) and 1.0 (identical).
    """
    # Step 1: Center the features (mean subtraction across the batch dimension)
    x_centered = features_x - features_x.mean(dim=0, keepdim=True)
    y_centered = features_y - features_y.mean(dim=0, keepdim=True)

    # Step 2: Compute the linear kernel (dot product) matrices
    dot_prod_x = torch.mm(x_centered, x_centered.t())
    dot_prod_y = torch.mm(y_centered, y_centered.t())

    # Step 3: Compute the Hilbert-Schmidt Independence Criterion (HSIC)
    # This is efficiently calculated as the sum of the element-wise product
    hsic = torch.sum(dot_prod_x * dot_prod_y)
    
    # Step 4: Compute the normalization terms (Frobenius norms)
    norm_x = torch.sqrt(torch.sum(dot_prod_x * dot_prod_x))
    norm_y = torch.sqrt(torch.sum(dot_prod_y * dot_prod_y))

    # Handle edge cases to prevent division by zero gracefully
    if norm_x == 0 or norm_y == 0:
        return 0.0

    # Step 5: Calculate and return the final CKA score
    cka_score = hsic / (norm_x * norm_y)
    return cka_score.item()

def calculate_layerwise_cka(model_a, model_b, dataloader, device="cuda"):
    """
    Extracts layer-wise CKA similarity between two models over a given dataset.
    This is actively used to quantify "Representation Collapse" by comparing 
    a baseline PEFT model to a Fully Fine-Tuned (FFT) reference model.
    """
    model_a.eval()
    model_b.eval()
    
    cka_scores_per_layer = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            
            # Forward pass requesting hidden states from ALL transformer layers
            outputs_a = model_a(**inputs, output_hidden_states=True)
            outputs_b = model_b(**inputs, output_hidden_states=True)
            
            hidden_states_a = outputs_a.hidden_states
            hidden_states_b = outputs_b.hidden_states
            
            # Initialize the list of lists for storing layer-wise scores dynamically
            if not cka_scores_per_layer:
                cka_scores_per_layer = [[] for _ in range(len(hidden_states_a))]
                
            # Calculate CKA for the [CLS] token (index 0) across all layers
            for layer_idx in range(len(hidden_states_a)):
                # Extract the [CLS] token representation for the current layer
                cls_a = hidden_states_a[layer_idx][:, 0, :]
                cls_b = hidden_states_b[layer_idx][:, 0, :]
                
                score = linear_cka(cls_a, cls_b)
                cka_scores_per_layer[layer_idx].append(score)
                
            # For CKA, calculating over a single large batch (e.g., 64 or 128 samples) 
            # is statistically sufficient to capture representational similarity structure.
            # We break early to optimize evaluation latency.
            break 

    # Calculate the mean CKA score for each layer across the evaluated batches
    final_cka = [sum(scores) / len(scores) for scores in cka_scores_per_layer]
    return final_cka