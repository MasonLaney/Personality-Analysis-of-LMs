
# package imports
import pandas as pd
import pickle

# Hugging Face Hub stuff
from huggingface_hub import HfApi
api = HfApi()
access_token = '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'

# process annotation data
'''
with open('/playpen/mlaney/multimodal/input_data/first_impressions_dataset/training/annotation_training.pkl', 'rb') as f:
    train_annotations = pickle.load(f, encoding='latin1')
train_df = pd.DataFrame(train_annotations)
train_df.insert(0, 'file_name', train_df.index)
train_df['file_name'] = 'data/train/' + train_df['file_name'].replace('.mp4', '.mp3', regex=True)
with open('/playpen/mlaney/multimodal/input_data/first_impressions_dataset/validation/annotation_validation.pkl', 'rb') as f:
    val_annotations = pickle.load(f, encoding='latin1')
val_df = pd.DataFrame(val_annotations)
val_df.insert(0, 'file_name', val_df.index)
val_df['file_name'] = 'data/validation/' + val_df['file_name'].replace('.mp4', '.mp3', regex=True)
full_df = pd.concat([train_df, val_df])
full_df.to_csv('/playpen/mlaney/multimodal/input_data/first_impressions_dataset/metadata.csv', index=False)

with open('/playpen/mlaney/multimodal/input_data/first_impressions_dataset/training/annotation_training.pkl', 'rb') as f:
    train_annotations = pickle.load(f, encoding='latin1')
train_df = pd.DataFrame(train_annotations)
train_df.insert(0, 'file_name', train_df.index)
train_df['file_name'] = train_df['file_name'].replace('.mp4', '.mp3', regex=True)
train_df.to_csv('/playpen/mlaney/multimodal/input_data/first_impressions_dataset/train_metadata.csv', index=False)

with open('/playpen/mlaney/multimodal/input_data/first_impressions_dataset/validation/annotation_validation.pkl', 'rb') as f:
    val_annotations = pickle.load(f, encoding='latin1')
val_df = pd.DataFrame(val_annotations)
val_df.insert(0, 'file_name', val_df.index)
val_df['file_name'] = val_df['file_name'].replace('.mp4', '.mp3', regex=True)
val_df.to_csv('/playpen/mlaney/multimodal/input_data/first_impressions_dataset/validate_metadata.csv', index=False)
'''

# upload audio data to hub
'''
api.upload_folder(
    folder_path='/playpen/mlaney/multimodal/input_data/first_impressions_dataset/audio',
    repo_id='mlaney/First_Impressions_V2',
    repo_type='dataset',
    token=access_token
)
'''

