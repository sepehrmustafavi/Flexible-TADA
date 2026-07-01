import logging
from transformers import DataCollatorWithPadding

logger = logging.getLogger(__name__)

# Mapping each task to its respective text column names in the HuggingFace dataset
TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}

class FlexibleTADADataProcessor:
    """
    Performs dataset preprocessing for encoder-based Transformer models.

    This class is responsible for converting raw GLUE dataset examples into
    tokenized inputs that are compatible with HuggingFace Trainer. It handles
    task-specific text extraction, sequence tokenization, dynamic padding,
    truncation, and label formatting for downstream training and evaluation.
    """

    def __init__(self, tokenizer, max_seq_length: int = 128):
        """
        Initializes the data processor with the tokenizer and preprocessing settings.

        Args:
            tokenizer (PreTrainedTokenizer): HuggingFace tokenizer used to encode
                input text into token IDs.
            max_seq_length (int): Maximum sequence length after tokenization.
                Input sequences longer than this value are truncated.

        Returns:
            None
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def preprocess_function(self, examples, task_name: str):
        """
        Tokenizes a batch of dataset examples for a specified GLUE task.

        This method automatically selects the appropriate input text columns
        based on the task, performs tokenization with truncation, and renames
        the dataset labels to the "labels" field expected by the HuggingFace
        Trainer API.

        Args:
            examples (dict): A batch of dataset examples provided by the
                HuggingFace Dataset object.
            task_name (str): Name of the GLUE task used to determine the
                corresponding input text columns.

        Returns:
            dict: A dictionary containing tokenized inputs and labels
            formatted for model training.
        """
        task_name = task_name.lower()
        if task_name not in TASK_TO_KEYS:
            raise ValueError(f"Unrecognized task name: {task_name}")

        key1, key2 = TASK_TO_KEYS[task_name]

        # Extract text columns
        texts_1 = examples[key1]
        texts_2 = examples[key2] if key2 is not None else None

        args = (texts_1,) if texts_2 is None else (texts_1, texts_2)
        
        tokenized_inputs = self.tokenizer(
            *args,
            padding=False, # We will pad dynamically in the DataCollator
            truncation=True,
            max_length=self.max_seq_length,
        )

        # Ensure labels are correctly named for the HuggingFace Trainer
        if "label" in examples:
            tokenized_inputs["labels"] = examples["label"]

        return tokenized_inputs

    def prepare_dataset(self, dataset_dict, task_name: str):
        """
        Applies tokenization to all dataset splits for a specified GLUE task.

        This method maps the preprocessing function over every split in the
        dataset, removes the original text columns after tokenization, and
        returns a fully processed DatasetDict ready for model training.

        Args:
            dataset_dict (DatasetDict): HuggingFace DatasetDict containing
                the dataset splits.
            task_name (str): Name of the GLUE task being processed.

        Returns:
            DatasetDict: The tokenized dataset with original text columns removed.
        """
        logger.info(f"Tokenizing dataset for task: {task_name}...")
        
        column_names = dataset_dict["train"].column_names

        tokenized_datasets = dataset_dict.map(
            lambda examples: self.preprocess_function(examples, task_name),
            batched=True,
            remove_columns=column_names, 
            desc=f"Running tokenizer on {task_name} dataset",
        )

        return tokenized_datasets

    def get_data_collator(self):
        """
        Creates a dynamic padding data collator for batch preparation.

        The returned collator pads each batch to the maximum sequence length
        within that batch instead of padding the entire dataset, improving
        memory efficiency during training. If the tokenizer does not define
        a padding token, the EOS token is used as the padding token.

        Returns:
            DataCollatorWithPadding: A HuggingFace data collator configured
            for dynamic batch padding.
        """
        if self.tokenizer.pad_token is None:
            logger.warning("Pad token is not set. Using eos_token as pad_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        return DataCollatorWithPadding(tokenizer=self.tokenizer, pad_to_multiple_of=8)
