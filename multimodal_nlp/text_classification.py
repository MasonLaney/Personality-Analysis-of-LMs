
# loosely based on https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb

# package imports
import numpy as np
import os
import torch
from datasets import load_from_disk, DatasetDict
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction

# constants / GPU setup
seed = 42
root_dir = '/playpen/mlaney/multimodal/'
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
torch.cuda.empty_cache()
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# load datasets
train_dataset = load_from_disk('first_impressions_v2/text/train')
val_dataset = load_from_disk('first_impressions_v2/text/validate')
dataset = DatasetDict({'train': train_dataset, 'validation': val_dataset})

# task setup
# labels = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']
labels = [label for label in dataset['train'].features.keys() if label not in ['text', 'interview']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

# model parameters / setup
max_seq_len = 128
batch_size = 8
metric_name = 'averaged_accuracy'
pretrained_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

# takes a batch of texts, encodes them, adds labels
# creates np array of shape (batch_size, num_labels)
def preprocess_data(examples):
    text = examples['text']
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=128)
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    labels_matrix = np.zeros((len(text), len(labels)))
    for idx, label in enumerate(labels):
        # NOTE: for regression, remove this rounding
        labels_matrix[:, idx] = [round(l) for l in labels_batch[label]]
    encoding['labels'] = labels_matrix.tolist()
    return encoding

# preprocess/encode datasets
encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
encoded_dataset.set_format('torch')

# setup model
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name,
                                                           problem_type='multi_label_classification',
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)

args = TrainingArguments(
    output_dir='first_impressions_v2_text',
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=8,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir=root_dir+'logs/'
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

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# training and evaluation
trainer.train()
print(trainer.evaluate())
