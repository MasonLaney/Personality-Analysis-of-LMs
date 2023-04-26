# package imports
import multiprocessing
import numpy as np
import os
import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from datasets import load_from_disk
from sklearn.metrics import accuracy_score
from transformers import (
    VideoMAEFeatureExtractor,
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from pytorchvideo.transforms import UniformTemporalSubsample

# constants / GPU setup
seed = 42
root_dir = '/playpen/mlaney/multimodal/'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
torch.cuda.device(2)
torch.cuda.empty_cache()
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
num_proc = multiprocessing.cpu_count()

# load dataset
dataset = load_from_disk('FIV2')
dataset = dataset.remove_columns(['audio', 'text', 'interview'])

# task setup
# labels = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']
labels = [label for label in dataset['train'].features.keys() if label != 'video']
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

# model parameters / setup
batch_size = 2
metric_name = 'averaged_accuracy'
#pretrained_model_name = 'MCG-NJU/videomae-base'
pretrained_model_name = 'MCG-NJU/videomae-small-finetuned-ssv2'
#pretrained_model_name = 'hf-tiny-model-private/tiny-random-VideoMAEForVideoClassification'
processor = VideoMAEImageProcessor.from_pretrained(pretrained_model_name)

# setup model
model = VideoMAEForVideoClassification.from_pretrained(
    pretrained_model_name,
    problem_type='multi_label_classification',
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# TODO: modify so it allows actual batching
def prepare_dataset(batch):

    video = EncodedVideo.from_path(batch['video'], decoder='decord')
    clip = video.get_clip(start_sec=0, end_sec=video.duration)['video']
    #clip = video
    subsampler = UniformTemporalSubsample(model.config.num_frames)
    subsampled_frames  = subsampler(clip)
    clip_np = subsampled_frames.numpy().transpose(1, 2, 3, 0)
    batch['pixel_values'] = processor.preprocess(list(clip_np)).pixel_values[0]
    #batch['input_length'] = len(batch['pixel_values'])
    # TODO: fix input_length

    labels_batch = {k: batch[k] for k in batch.keys() if k in labels}
    labels_matrix = np.zeros((1, len(labels)))

    for idx, label in enumerate(labels):
        # NOTE: for regression, remove this rounding
        labels_matrix[:, idx] = round(labels_batch[label])
    batch['labels'] = labels_matrix[0]
    # TODO: labels is currently a list inside a list, hence the [0]

    return batch

# preprocess/encode datasets
#dataset['train'] = dataset['train'].select(range(10))
#dataset['validate'] = dataset['validate'].select(range(10))
#dataset['test'] = dataset['test'].select(range(10))

encoded_dataset_train = dataset['validate'].select(range(2,4)).map(prepare_dataset, remove_columns=dataset['train'].column_names, load_from_cache_file=True, writer_batch_size=2000)
encoded_dataset_train.set_format('torch')
encoded_dataset_train.save_to_disk('FIV2_video_validate')
exit()