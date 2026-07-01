import logging
from datasets import load_dataset, DatasetDict

logger = logging.getLogger(__name__)

class FlexibleTADADatasetBuilder:
    """
    Loads and prepares GLUE benchmark datasets for model training.

    This class provides a unified interface for loading individual GLUE
    benchmark tasks from the HuggingFace Datasets library. It also supports
    reproducible Few-Shot learning by randomly sampling a fixed number of
    training examples using a user-defined random seed.
    """
    
    # Supported tasks mappings
    GLUE_TASKS = ["mnli", "sst2", "mrpc", "cola", "qnli", "qqp", "rte", "stsb"]

    def __init__(self, config: dict):
        """
        Initializes the dataset builder using the experiment configuration.

        The configuration specifies the benchmark to load, the tasks to process,
        the validation split to use, and the random seed for reproducible
        Few-Shot sampling.

        Args:
            config (dict): Dictionary containing dataset and experiment
                configuration parameters.

        Returns:
            None
        """
        self.benchmark = config.get("benchmark", "glue").lower()
        self.tasks = config.get("tasks", [])
        self.val_split_name = config.get("validation_split", "validation")
        self.seed = config.get("system", {}).get("seed", 42) # Crucial for reproducible few-shot runs

    def load_task(self, task_name: str, num_train_samples: int = None) -> DatasetDict:
        """
        Loads a specified GLUE task and optionally applies Few-Shot sampling.

        The method downloads the requested dataset from the HuggingFace Datasets
        library, extracts the training and validation splits, handles special
        cases such as MNLI's matched validation set, and optionally reduces the
        training set to a fixed number of randomly sampled examples.

        Args:
            task_name (str): Name of the GLUE task to load.
            num_train_samples (int, optional): Number of training samples to
                retain for Few-Shot learning. If None, the full training set is
                used.

        Returns:
            DatasetDict: A HuggingFace DatasetDict containing the processed
            training and validation splits.
        """
        task_name = task_name.lower()
        
        # Determine the correct path based on the benchmark
        if task_name in self.GLUE_TASKS:
            path = "glue"
        else:
            raise ValueError(f"Task '{task_name}' is not supported in GLUE.")

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
        Loads all tasks specified in the experiment configuration.

        This method iterates over every configured task, loads its corresponding
        dataset, optionally applies Few-Shot sampling, and stores the processed
        datasets in a dictionary indexed by task name.

        Args:
            num_train_samples (int, optional): Number of training samples to
                retain for each task. If None, the complete training set is used.

        Returns:
            dict: A dictionary mapping each task name to its corresponding
            processed DatasetDict.
        """
        all_datasets = {}
        for task in self.tasks:
            all_datasets[task] = self.load_task(task, num_train_samples=num_train_samples)
        return all_datasets

if __name__ == "__main__":
    # Mock config
    test_config = {
        "benchmark": "glue",
        "tasks": ["rte"],
        "seed": 2026
    }
    
    builder = FlexibleTADADatasetBuilder(test_config)
    
    # Load RTE with only 100 samples (Few-Shot Scenario)
    few_shot_dataset = builder.load_task("rte", num_train_samples=100)
    print(few_shot_dataset)
