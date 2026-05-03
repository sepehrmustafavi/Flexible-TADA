import torch

def calculate_faithfulness(model, batch, important_indices, device):
    """
    Faithfulness (Infidelity): Measures if removing important words 
    leads to a significant drop in the correct class probability.
    """
    model.eval()
    with torch.no_grad():
        # 1. Original Probability
        outputs = model(**{k: v.to(device) for k, v in batch.items()})
        probs = torch.softmax(outputs.logits, dim=-1)
        original_prob = probs.max(dim=-1).values
        predicted_class = probs.argmax(dim=-1)

        # 2. Masked Probability
        input_ids = batch["input_ids"].clone()
        attention_mask = batch["attention_mask"].clone()
        
        for i, idx in enumerate(important_indices):
            if len(idx) > 0: # Ensure there are indices to mask
                # Replace important tokens with Pad token and update mask
                input_ids[i, idx] = model.config.pad_token_id
                attention_mask[i, idx] = 0
        
        masked_outputs = model(input_ids.to(device), attention_mask=attention_mask.to(device))
        masked_probs = torch.softmax(masked_outputs.logits, dim=-1)
        
        # Look at the probability of the *original* predicted class
        masked_prob_for_class = torch.gather(masked_probs, 1, predicted_class.unsqueeze(-1))

    # Faithfulness score: Higher is better
    return (original_prob - masked_prob_for_class.squeeze(-1)).mean().item()

def calculate_sufficiency(model, batch, important_indices, device):
    """
    Sufficiency: Measures if the important words ALONE are enough 
    for the model to make the correct prediction.
    """
    model.eval()
    with torch.no_grad():
        # 1. Get Original Predicted Class
        original_outputs = model(**{k: v.to(device) for k, v in batch.items()})
        original_probs = torch.softmax(original_outputs.logits, dim=-1)
        predicted_class = original_probs.argmax(dim=-1)

        # 2. Keep ONLY important words
        input_ids = torch.full_like(batch["input_ids"], model.config.pad_token_id)
        attention_mask = torch.zeros_like(batch["attention_mask"])
        
        for i, idx in enumerate(important_indices):
            if len(idx) > 0:
                input_ids[i, idx] = batch["input_ids"][i, idx]
                attention_mask[i, idx] = 1 # Only attend to these important words
            
        outputs = model(input_ids.to(device), attention_mask=attention_mask.to(device))
        probs = torch.softmax(outputs.logits, dim=-1)
        
        # We MUST evaluate the confidence on the ORIGINAL predicted class
        masked_prob_for_class = torch.gather(probs, 1, predicted_class.unsqueeze(-1))
        
        # High score means important words alone were sufficient
        return masked_prob_for_class.squeeze(-1).mean().item()