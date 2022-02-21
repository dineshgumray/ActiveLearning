# Preparing the Data
import logging
import transformers
from transformers import AutoTokenizer
transformers.logging.get_verbosity = lambda: logging.NOTSET

class PreTrainedModel:
    def __init__(self):
        pass
    def set_tokenizer(self, transformer_model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(
            transformer_model_name
        )
    def get_set_tokenizer(self):
        return self.tokenizer