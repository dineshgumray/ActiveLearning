# Loading the Data
import datasets
import logging
import numpy as np


# disables the progress bar for notebooks: https://github.com/huggingface/datasets/issues/2651
datasets.logging.get_verbosity = lambda: logging.NOTSET

raw_dataset = datasets.load_dataset('rotten_tomatoes')
num_classes = np.unique(raw_dataset['train']['label']).shape[0]

print('First 2 testing samples:\n')
for i in range(2):
    print(raw_dataset['test']['label'][i], ' ', raw_dataset['test']['text'][i])