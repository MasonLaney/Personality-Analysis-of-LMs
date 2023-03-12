#cache_dir = root_dir + 'cache/'
#download_config = DownloadConfig(cache_dir=cache_dir, resume_download=True)
'''
# Hugging Face Hub stuff
from huggingface_hub import HfApi, login
api = HfApi()
access_token = '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'
login(token=access_token)
'''
#   {'audio': {'path': '/playpen/mlaney/multimodal/input_data/first_impressions_v2/data/train/--Ymqszjv54.003.mp3',
#       'array': array([0.        , 0.        , 0.        , ..., 0.00745735, 0.02285403, 0.03162834], dtype=float32),
#       'sampling_rate': 44100},
#   'extraversion': 0.3925233644859813, 'neuroticism': 0.4270833333333333, 'agreeableness': 0.5164835164835165, 'conscientiousness': 0.4757281553398058, 'interview': 0.3925233644859813, 'openness': 0.4666666666666667}


# package imports
import numpy as np
import os
import torch
from datasets import Audio, load_dataset, DatasetDict
from sklearn.metrics import accuracy_score
from transformers import AutoProcessor, AutoFeatureExtractor, HubertForSequenceClassification, \
    TrainingArguments, Trainer, EvalPrediction, PretrainedConfig
from DCCTC import DataCollatorCTCWithPadding

# constants / GPU setup
seed = 42
root_dir = '/playpen/mlaney/multimodal/'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
#torch.cuda.set_device(1)
torch.cuda.empty_cache()
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# TODO: figure out why unified train/val version didn't work, update HF Hub
# load datasets
train_dataset = load_dataset('audiofolder', data_dir=root_dir+'input_data/first_impressions_v2/data/train')['train']
val_dataset = load_dataset('audiofolder', data_dir=root_dir+'input_data/first_impressions_v2/data/validation')['train']
dataset = DatasetDict({'train': train_dataset, 'validation': val_dataset})
dataset = dataset.cast_column('audio', Audio(sampling_rate=16_000))

# task setup
# labels = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']
labels = [label for label in dataset['train'].features.keys() if label not in ['audio', 'interview']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

# model parameters / setup
sampling_rate = dataset['train'].features['audio'].sampling_rate
batch_size = 10
metric_name = 'averaged_accuracy'
pretrained_model_name = 'facebook/hubert-base-ls960'
# voidful/hubert-tiny-v2
processor = AutoProcessor.from_pretrained(pretrained_model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name)

# TODO: modify so it allows actual batching
def prepare_dataset(batch):

    audio = batch['audio']
    batch['input_values'] = processor(audio['array'], sampling_rate=audio['sampling_rate']).input_values[0]
    batch['input_length'] = len(batch['input_values'])

    labels_batch = {k: batch[k] for k in batch.keys() if k in labels}
    labels_matrix = np.zeros((1, len(labels)))

    for idx, label in enumerate(labels):
        # NOTE: for regression, remove this rounding
        labels_matrix[:, idx] = round(labels_batch[label])
    batch['labels'] = labels_matrix[0]
    # TODO: labels is currently a list inside a list
    return batch


# preprocess/encode datasets
encoded_dataset = dataset.map(prepare_dataset, remove_columns=dataset['train'].column_names)
encoded_dataset.set_format('torch')

# setup model
model = HubertForSequenceClassification.from_pretrained(pretrained_model_name,
                                                           problem_type='multi_label_classification',
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)

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
    gradient_accumulation_steps=1
)

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
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



data_collator = DataCollatorCTCWithPadding(
            processor=feature_extractor,
            padding=True
)

trainer = Trainer(
    model,
    args,
    data_collator=data_collator,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation'],
    compute_metrics=compute_metrics
)

# training and evaluation
trainer.train()
print(trainer.evaluate())
