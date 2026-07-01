import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)

def build_compute_metrics_fn(task_name: str):
    """
    Creates a task-specific evaluation function for the HuggingFace Trainer.

    This factory function returns a compute_metrics callback that automatically
    selects the appropriate evaluation metrics based on the specified GLUE task.
    It supports both classification and regression tasks, ensuring that each
    benchmark is evaluated using its standard performance metrics.

    Args:
        task_name (str): Name of the GLUE task for which evaluation metrics
            should be computed.

    Returns:
        function: A compute_metrics function compatible with the HuggingFace
        Trainer API.
    """
    task_name = task_name.lower()
    logger.info(f"Building metrics calculator for task: {task_name.upper()}")

    def compute_metrics(eval_pred):
        """
        Computes evaluation metrics for model predictions.

        This function is invoked automatically by the HuggingFace Trainer
        during evaluation. It processes the model predictions and ground-truth
        labels, applies task-specific metric calculations, and returns the
        resulting performance scores.

        Args:
            eval_pred: Tuple containing model predictions (logits) and
                corresponding ground-truth labels.

        Returns:
            dict: Dictionary containing the evaluation metrics appropriate
            for the selected GLUE task.
        """
        logits, labels = eval_pred
        
        # 1. Handle Regression Task (STS-B)
        # STS-B predicts a continuous similarity score between 0 and 5, not a class.
        if task_name == "stsb":
            predictions = np.squeeze(logits)
            
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
