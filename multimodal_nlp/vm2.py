'''
# package imports
import multiprocessing
import numpy as np
import os
import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from datasets import load_from_disk, load_dataset
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

encoded_dataset_train = dataset['train'].select(range(2000,4000)).map(prepare_dataset, remove_columns=dataset['train'].column_names, load_from_cache_file=True, writer_batch_size=2005)
encoded_dataset_train.set_format('torch')
encoded_dataset_train.save_to_disk('FIV2_video_train2')
exit()
'''

# based on audio_model.py, takes inspiration from: https://huggingface.co/tasks/video-classification

# package imports
import multiprocessing
import numpy as np
import os
import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from sklearn.metrics import accuracy_score
from transformers import (
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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.cuda.device(1)
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
#pretrained_model_name = 'hf-internal-testing/tiny-random-VideoMAEForVideoClassification'
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
#encoded_dataset = dataset.map(prepare_dataset, remove_columns=dataset['train'].column_names, load_from_cache_file=True, writer_batch_size=2005)
#encoded_dataset.save_to_disk('FIV2_video')
train1 = load_from_disk('FIV2_video_train1')
train2 = load_from_disk('FIV2_video_train2')
train3 = load_from_disk('FIV2_video_train3')
train = concatenate_datasets([train1, train2, train3])
val = load_from_disk('FIV2_video_validate')
encoded_dataset = DatasetDict({'train': train, 'validate': val})
encoded_dataset.set_format('torch')

# setup training parameters
args = TrainingArguments(
    output_dir='first_impressions_v2_audio',
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir=root_dir+'logs/',
    gradient_accumulation_steps=1,
    fp16=True,
)

# modified from https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):

    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels

    # convert to tensors for column slicing
    y_pred_tensor = torch.Tensor(y_pred)
    y_true_tensor = torch.Tensor(y_true)

    # calculate the mean accuracy of each individual personality trait and the averaged mean accuracy
    extraversion_accuracy = accuracy_score(y_true_tensor[:, 0].tolist(), y_pred_tensor[:, 0].tolist())
    neuroticism_accuracy = accuracy_score(y_true_tensor[:, 1].tolist(), y_pred_tensor[:, 1].tolist())
    agreeableness_accuracy = accuracy_score(y_true_tensor[:, 2].tolist(), y_pred_tensor[:, 2].tolist())
    conscientiousness_accuracy = accuracy_score(y_true_tensor[:, 3].tolist(), y_pred_tensor[:, 3].tolist())
    openness_accuracy = accuracy_score(y_true_tensor[:, 4].tolist(), y_pred_tensor[:, 4].tolist())
    averaged_accuracy = np.mean([extraversion_accuracy, neuroticism_accuracy, agreeableness_accuracy, conscientiousness_accuracy, openness_accuracy])
    exact_accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        'averaged_accuracy': averaged_accuracy,
        'exact_accuracy': exact_accuracy,
        'extraversion_accuracy': extraversion_accuracy,
        'agreeableness_accuracy': agreeableness_accuracy,
        'conscientiousness_accuracy': conscientiousness_accuracy,
        'neuroticism_accuracy': neuroticism_accuracy,
        'openness_accuracy': openness_accuracy
    }
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result

# modified from https://discuss.huggingface.co/t/multilabel-sequence-classification-with-roberta-value-error-expected-input-batch-size-to-match-target-batch-size/1653
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(pixel_values=inputs['pixel_values'])
        loss = torch.nn.BCEWithLogitsLoss()(outputs['logits'], inputs['labels'])
        return (loss, outputs) if return_outputs else loss


trainer = CustomTrainer(
    model,
    args,
    #data_collator=data_collator,
    train_dataset=encoded_dataset['train'].select(range(250)),
    eval_dataset=encoded_dataset['validate'].select(range(500)),
    compute_metrics=compute_metrics
)

# training and evaluation
#trainer.train('first_impressions_v2_audio/checkpoint-125')
#trainer.save_model('video_model_250')
#print(trainer.evaluate())

model = VideoMAEForVideoClassification.from_pretrained('video_model_250')
print(trainer.predict(encoded_dataset['validate'].select(range(500))))

test = load_from_disk('FIV2_video_test')
print(trainer.predict(test.select(range(500))))
