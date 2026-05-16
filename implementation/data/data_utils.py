import logging
from transformers import DataCollatorWithPadding

logger = logging.getLogger(__name__)

# Mapping each task to its respective text column names in the HuggingFace dataset
TASK_TO_KEYS = {
    # GLUE Tasks
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
    Handles tokenization, dynamic padding, and tensor formatting for 
    Encoder-based pipelines (BERT, RoBERTa, DeBERTa, ELECTRA).
    """

    def __init__(self, tokenizer, max_seq_length: int = 128):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def preprocess_function(self, examples, task_name: str):
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
        # Encoders typically have pad_token set by default, but keeping a fallback is safe practice.
        if self.tokenizer.pad_token is None:
            logger.warning("Pad token is not set. Using eos_token as pad_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        return DataCollatorWithPadding(tokenizer=self.tokenizer, pad_to_multiple_of=8)