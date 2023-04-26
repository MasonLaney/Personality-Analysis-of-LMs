
# partially based on https://huggingface.co/docs/transformers/tasks/video_classification

# package imports
import numpy as np
import os
import torch
import torch.multiprocessing
import pytorchvideo.data
from pytorchvideo.data import LabeledVideoDataset
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
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

# constants / GPU setup
seed = 42
root_dir = '/playpen/mlaney/multimodal/'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
torch.cuda.device(2)
torch.cuda.empty_cache()
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

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
sample_rate = 1
fps = 30
metric_name = 'averaged_accuracy'
#pretrained_model_name = 'MCG-NJU/videomae-base'
pretrained_model_name = 'MCG-NJU/videomae-small-finetuned-ssv2'
#pretrained_model_name = 'hf-tiny-model-private/tiny-random-VideoMAEForVideoClassification'

# setup model
model = VideoMAEForVideoClassification.from_pretrained(
    pretrained_model_name,
    problem_type='multi_label_classification',
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# image processor setup
processor = VideoMAEImageProcessor.from_pretrained(pretrained_model_name)
mean = processor.image_mean
std = processor.image_std
if 'shortest_edge' in processor.size:
    height = width = processor.size['shortest_edge']
else:
    height = processor.size['height']
    width = processor.size['width']
resize_to = (height, width)
num_frames_to_sample = model.config.num_frames
clip_duration = num_frames_to_sample * sample_rate / fps

# preprocess/encode datasets
dataset['train'] = dataset['train'].select(range(10))
dataset['validate'] = dataset['validate'].select(range(10))
dataset['test'] = dataset['test'].select(range(10))

# transform images in dataset
train_transform = Compose([ApplyTransformToKey(key='video', transform=Compose([
    UniformTemporalSubsample(num_frames_to_sample),
    Lambda(lambda x: x / 255.0),
    Normalize(mean, std),
    #RandomShortSideScale(min_size=256, max_size=320),
    #RandomCrop(resize_to),
    #RandomHorizontalFlip(p=0.5)
    Resize(resize_to)
]))])
train_dataset = LabeledVideoDataset(
    labeled_video_paths=[(item['video'], {l: item[l] for l in labels}) for item in dataset['train']],
    clip_sampler=pytorchvideo.data.make_clip_sampler('random', clip_duration),
    decode_audio=False,
    transform=train_transform,
    decoder='decord'
)
val_transform = Compose([ApplyTransformToKey(key='video',transform=Compose([
    UniformTemporalSubsample(num_frames_to_sample),
    Lambda(lambda x: x / 255.0),
    Normalize(mean, std),
    Resize(resize_to)
]))])
val_dataset = LabeledVideoDataset(
    labeled_video_paths=[(item['video'], {l: item[l] for l in labels}) for item in dataset['validate']],
    clip_sampler=pytorchvideo.data.make_clip_sampler('uniform', clip_duration),
    decode_audio=False,
    transform=val_transform,
    decoder='decord'
)
test_dataset = LabeledVideoDataset(
    labeled_video_paths=[(item['video'], {l: item[l] for l in labels}) for item in dataset['test']],
    clip_sampler=pytorchvideo.data.make_clip_sampler('uniform', clip_duration),
    decode_audio=False,
    transform=val_transform,
    decoder='decord'
)

# setup training parameters
num_epochs = 4
grad_acc_steps = 1
args = TrainingArguments(
    output_dir='first_impressions_v2_video',
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir=root_dir+'logs/',
    remove_unused_columns=False,
    max_steps=(train_dataset.num_videos // batch_size) * num_epochs//grad_acc_steps,
    gradient_accumulation_steps=grad_acc_steps,
    #prediction_loss_only=True,
    fp16=True
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

# TODO: batching?
def collate_fn(examples):

    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example['video'].permute(1, 0, 2, 3) for example in examples]
    )
    ex_labels = torch.stack([torch.tensor([round(v) for k, v in example.items() if k in labels]) for example in examples])
    # TODO ^ changed
    return {'pixel_values': pixel_values, 'labels': ex_labels}

# modified from https://discuss.huggingface.co/t/multilabel-sequence-classification-with-roberta-value-error-expected-input-batch-size-to-match-target-batch-size/1653
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(pixel_values=inputs['pixel_values'])
        loss = torch.nn.BCEWithLogitsLoss()(outputs['logits'], inputs['labels'].float())
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model,
    args,
    data_collator=collate_fn,
    train_dataset=train_dataset,
    tokenizer=processor,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# training and evaluation
trainer.train()
trainer.save_model('video_model')
print(trainer.evaluate())
print(trainer.predict(test_dataset))

