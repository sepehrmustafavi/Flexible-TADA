import logging
from datasets import load_dataset, DatasetDict

# Set up logger
logger = logging.getLogger(__name__)

class FlexibleTADADatasetBuilder:
    """
    A unified dataset builder for loading GLUE and SuperGLUE benchmarks,
    with built-in support for Few-Shot sampling based on random seeds.
    """
    
    # Supported tasks mappings
    GLUE_TASKS = ["mnli", "sst2", "mrpc", "cola", "qnli", "qqp", "rte", "stsb"]
    SUPERGLUE_TASKS = ["boolq", "cb", "wic"]

    def __init__(self, config: dict):
        """
        Args:
            config (dict): A dictionary containing dataset configurations 
                           (typically loaded from yaml files).
        """
        self.benchmark = config.get("benchmark", "glue").lower()
        self.tasks = config.get("tasks", [])
        self.val_split_name = config.get("validation_split", "validation")
        self.seed = config.get("seed", 42) # Crucial for reproducible few-shot runs

    def load_task(self, task_name: str, num_train_samples: int = None) -> DatasetDict:
        """
        Loads a specific task from HuggingFace datasets and applies few-shot 
        sub-sampling if num_train_samples is provided.

        Args:
            task_name (str): Name of the task (e.g., 'mnli', 'boolq')
            num_train_samples (int, optional): Number of samples to keep for training.
            
        Returns:
            DatasetDict: A HuggingFace DatasetDict containing train and validation splits.
        """
        task_name = task_name.lower()
        
        # Determine the correct path based on the benchmark
        if task_name in self.GLUE_TASKS:
            path = "glue"
        elif task_name in self.SUPERGLUE_TASKS:
            path = "super_glue"
        else:
            raise ValueError(f"Task '{task_name}' is not supported in GLUE or SuperGLUE.")

        logger.info(f"Loading task '{task_name}' from benchmark '{path}'...")
        
        try:
            raw_datasets = load_dataset(path, task_name)
        except Exception as e:
            logger.error(f"Failed to load dataset {task_name}: {e}")
            raise

        # Handle MNLI special case (it has matched and mismatched validation sets)
        val_split = "validation_matched" if task_name == "mnli" else self.val_split_name

        # Extract only the necessary splits to save memory
        processed_dataset = DatasetDict({
            "train": raw_datasets["train"],
            "validation": raw_datasets[val_split]
        })

        # Apply Few-Shot logic if specified (Scenario 4 in our execution plan)
        if num_train_samples is not None:
            logger.info(f"Applying Few-Shot sampling: Reducing train set to {num_train_samples} samples (Seed: {self.seed}).")
            
            # Ensure we don't ask for more samples than available
            total_samples = len(processed_dataset["train"])
            if num_train_samples > total_samples:
                logger.warning(f"Requested {num_train_samples} samples, but only {total_samples} available. Using all.")
                num_train_samples = total_samples

            # Shuffle with a fixed seed and select the subset
            processed_dataset["train"] = processed_dataset["train"].shuffle(seed=self.seed).select(range(num_train_samples))
            logger.info(f"New train set size: {len(processed_dataset['train'])}")

        return processed_dataset

    def get_all_tasks(self, num_train_samples: int = None) -> dict:
        """
        Loads all tasks defined in the configuration.
        
        Returns:
            dict: A dictionary mapping task names to their respective DatasetDicts.
        """
        all_datasets = {}
        for task in self.tasks:
            all_datasets[task] = self.load_task(task, num_train_samples=num_train_samples)
        return all_datasets

# Example usage for debugging purposes
if __name__ == "__main__":
    # Mock config
    test_config = {
        "benchmark": "glue",
        "tasks": ["rte"],
        "seed": 2026 # One of our specific seeds for the few-shot robust test
    }
    
    builder = FlexibleTADADatasetBuilder(test_config)
    
    # Load RTE with only 100 samples (Few-Shot Scenario)
    few_shot_dataset = builder.load_task("rte", num_train_samples=100)
    print(few_shot_dataset)