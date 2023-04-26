
# package imports
import pandas as pd
import pickle

# constants
root_dir = '/playpen/mlaney/multimodal/'

# process annotation data
with open(f'{root_dir}input_data/first_impressions_dataset/training/annotation_training.pkl', 'rb') as f:
    train_annotations = pickle.load(f, encoding='latin1')
train_df = pd.DataFrame(train_annotations)
train_df.insert(0, 'file_name', train_df.index)
train_df['file_name'] = 'videos/train/' + train_df['file_name']
with open(f'{root_dir}input_data/first_impressions_dataset/validation/annotation_validation.pkl', 'rb') as f:
    val_annotations = pickle.load(f, encoding='latin1')
val_df = pd.DataFrame(val_annotations)
val_df.insert(0, 'file_name', val_df.index)
val_df['file_name'] = 'videos/validation/' + val_df['file_name']
full_df = pd.concat([train_df, val_df])
full_df.to_csv(f'{root_dir}input_data/videos/metadata.csv', index=False)

with open(f'{root_dir}input_data/first_impressions_dataset/training/annotation_training.pkl', 'rb') as f:
    train_annotations = pickle.load(f, encoding='latin1')
train_df = pd.DataFrame(train_annotations)
train_df.insert(0, 'file_name', train_df.index)
train_df['file_name'] = train_df['file_name']
train_df.to_csv(f'{root_dir}input_data/videos/train_metadata.csv', index=False)

with open(f'{root_dir}input_data/first_impressions_dataset/validation/annotation_validation.pkl', 'rb') as f:
    val_annotations = pickle.load(f, encoding='latin1')
val_df = pd.DataFrame(val_annotations)
val_df.insert(0, 'file_name', val_df.index)
val_df['file_name'] = val_df['file_name']
val_df.to_csv(f'{root_dir}input_data/videos/validate_metadata.csv', index=False)


