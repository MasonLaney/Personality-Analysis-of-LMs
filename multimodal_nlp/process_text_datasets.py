
# package imports
from datasets import Dataset
import pandas as pd
import pickle

# constants
root_dir = '/playpen/mlaney/multimodal/'

# create training dataset
with open(f'{root_dir}input_data/first_impressions_dataset/training/annotation_training.pkl', 'rb') as f:
    train_annotations = pickle.load(f, encoding='latin1')
with open(f'{root_dir}input_data/first_impressions_dataset/training/transcription_training.pkl', 'rb') as f:
    train_transcriptions = pickle.load(f, encoding='latin1')

train_df = pd.DataFrame(train_annotations)
train_df.insert(0, 'text', [train_transcriptions[filename] for filename in train_df.index])
train_dataset = Dataset.from_pandas(train_df, preserve_index=False)

# create validation dataset
with open(f'{root_dir}input_data/first_impressions_dataset/validation/annotation_validation.pkl', 'rb') as f:
    val_annotations = pickle.load(f, encoding='latin1')
with open(f'{root_dir}input_data/first_impressions_dataset/validation/transcription_validation.pkl', 'rb') as f:
    val_transcriptions = pickle.load(f, encoding='latin1')

val_df = pd.DataFrame(val_annotations)
val_df.insert(0, 'text', [val_transcriptions[filename] for filename in val_df.index])
val_dataset = Dataset.from_pandas(val_df, preserve_index=False)

# save datasets locally
train_dataset.save_to_disk('first_impressions_v2/text/train')
val_dataset.save_to_disk('first_impressions_v2/text/validate')

print(val_dataset)
