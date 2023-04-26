
# takes inspiration from https://towardsdatascience.com/fine-tuning-hubert-for-emotion-recognition-in-custom-audio-data-using-huggingface-c2d516b41cd8

# package imports
import numpy as np
import os
import torch
from datasets import Audio, load_from_disk
from sklearn.metrics import accuracy_score
from transformers import AutoProcessor, AutoFeatureExtractor, HubertForSequenceClassification, TrainingArguments, Trainer, EvalPrediction, Wav2Vec2Processor
from DCCTC import DataCollatorCTCWithPadding

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
dataset = dataset.remove_columns(['video', 'text', 'interview'])
dataset = dataset.cast_column('audio', Audio(sampling_rate=16_000))

# task setup
# labels = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']
labels = [label for label in dataset['train'].features.keys() if label != 'audio']
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

# model parameters / setup
sampling_rate = dataset['train'].features['audio'].sampling_rate
batch_size = 8
metric_name = 'averaged_accuracy'
pretrained_model_name = 'ntu-spml/distilhubert'
processor = AutoProcessor.from_pretrained(pretrained_model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name)
#processor = Wav2Vec2Processor

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
    # TODO: labels is currently a list inside a list, hence the [0]

    return batch

# preprocess/encode datasets
encoded_dataset = dataset.map(prepare_dataset, remove_columns=dataset['train'].column_names)
encoded_dataset.set_format('torch')

# setup model
model = HubertForSequenceClassification.from_pretrained(
    pretrained_model_name,
    problem_type='multi_label_classification',
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# setup training parameters
args = TrainingArguments(
    output_dir='first_impressions_v2_audio',
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=8,
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

# TODO: change from class to standalone function
data_collator = DataCollatorCTCWithPadding(
    processor=feature_extractor,
    padding=True
)

# modified from https://discuss.huggingface.co/t/multilabel-sequence-classification-with-roberta-value-error-expected-input-batch-size-to-match-target-batch-size/1653
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(input_values=inputs['input_values'])
        loss = torch.nn.BCEWithLogitsLoss()(outputs['logits'], inputs['labels'])
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model,
    args,
    data_collator=data_collator,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validate'],
    compute_metrics=compute_metrics
)

# training and evaluation
trainer.train()
trainer.save_model('audio_model')
print(trainer.evaluate())
print(trainer.predict(encoded_dataset['test']))
