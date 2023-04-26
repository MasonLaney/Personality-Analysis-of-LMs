
# package imports
import pandas as pd
import pickle
from datasets import Audio, Dataset, DatasetDict

# constants
root_dir = '/playpen/mlaney/multimodal/'

# create datasets for each of the three splits
dataset_splits = {}
for split in ['train', 'validate', 'test']:

    # load the annotation df
    with open(f"{root_dir}input_data/FIV2/{split}/annotation_{split}.pkl", 'rb') as f:
        annotations = pickle.load(f, encoding='latin1')
        df = pd.DataFrame(annotations)

    # load the transcription dict
    with open(f"{root_dir}input_data/FIV2/{split}/transcription_{split}.pkl", 'rb') as f:
        transcriptions = pickle.load(f, encoding='latin1')
        df.insert(0, 'text', [transcriptions[filename] for filename in df.index])

    # filter out clips with fewer than 20 words spoken
    df = df[df['text'].str.len() >= 20]
    print(f"New {split} split size: {len(df)}")

    # add audio and video filepaths to dataset
    df.insert(0, 'video', df.index)
    df.insert(1, 'audio', df['video'].replace('.mp4', '.mp3', regex=True))
    df['video'] = df['video'].apply(lambda s: f"{root_dir}input_data/FIV2/{split}/video/{s}")
    df['audio'] = df['audio'].apply(lambda s: f"{root_dir}input_data/FIV2/{split}/audio/{s}")

    # covert to Dataset and store in dict
    dataset_split = Dataset.from_pandas(df, preserve_index=False)
    dataset_splits[split] = dataset_split

# combine different splits into one DatasetDict
dataset = DatasetDict(dataset_splits)

# save to disk
dataset.save_to_disk('FIV2')
