import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)

def build_compute_metrics_fn(task_name: str):
    """
    A factory function that returns the appropriate compute_metrics function 
    for the HuggingFace Trainer based on the specific GLUE/SuperGLUE task.

    Args:
        task_name (str): The name of the task (e.g., 'mnli', 'cola', 'cb').

    Returns:
        function: A compute_metrics function that takes an EvalPrediction object.
    """
    task_name = task_name.lower()
    logger.info(f"Building metrics calculator for task: {task_name.upper()}")

    def compute_metrics(eval_pred):
        """
        The actual evaluation logic called by Trainer at the end of each epoch/step.
        """
        logits, labels = eval_pred
        
        # 1. Handle Regression Task (STS-B)
        # STS-B predicts a continuous similarity score between 0 and 5, not a class.
        if task_name == "stsb":
            predictions = np.squeeze(logits)
            
            # Prevent division by zero or NaN issues in constant predictions
            try:
                pearson_corr = pearsonr(predictions, labels)[0]
                spearman_corr = spearmanr(predictions, labels)[0]
            except Exception:
                pearson_corr, spearman_corr = 0.0, 0.0
                
            return {
                "pearson": pearson_corr,
                "spearman": spearman_corr,
            }
        
        # 2. Handle Classification Tasks
        # For classification, we take the argmax of logits to get the predicted class index.
        predictions = np.argmax(logits, axis=-1)

        # 3. Task-Specific Metric Routing
        if task_name == "cola":
            # CoLA is highly imbalanced; Matthews Correlation Coefficient is the standard.
            return {"mcc": matthews_corrcoef(labels, predictions)}
        
        elif task_name in ["mrpc", "qqp"]:
            # These binary classification tasks report both Accuracy and F1-Score
            return {
                "accuracy": accuracy_score(labels, predictions),
                "f1": f1_score(labels, predictions)
            }
            
        elif task_name in ["cb"]:
            # CommitmentBank in SuperGLUE is multi-class and evaluated using F1-Macro
            return {
                "accuracy": accuracy_score(labels, predictions),
                "f1_macro": f1_score(labels, predictions, average="macro")
            }
            
        else:
            # Default fallback for balanced tasks like MNLI, SST-2, QNLI, RTE, BoolQ, WiC
            return {"accuracy": accuracy_score(labels, predictions)}

    return compute_metrics