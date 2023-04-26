
# package imports
import pandas as pd
import pickle
from datasets import Audio, Dataset, DatasetDict, load_from_disk

# constants
root_dir = '/playpen/mlaney/multimodal/'
traits = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']


# load from disk
dataset = load_from_disk('FIV2')

train = dataset['train'].to_pandas()
train_dict = {}
for trait in traits:
    train_dict[trait] = len(train[train[trait] >= 0.5])/len(train)

print(train_dict)

