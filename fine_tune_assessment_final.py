
# performance settings
import os
import gc
import torch
print('Number of unreachable objects collected by GC:', gc.collect())
torch.cuda.device(1)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.cuda.empty_cache()
#torch.use_deterministic_algorithms(True)

# package imports
import multiprocessing
from datasets import DownloadConfig, load_dataset
from itertools import chain
from transformers import TrainingArguments, Trainer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, \
    AutoTokenizer

# useful constants
DEFAULT_SEED = 42
DATA_DIR = '/ssd-playpen/mlaney/'
CACHE_DIR = DATA_DIR+'cache/'
CHUNK_SIZE = 256  # 512
NUM_PROC = multiprocessing.cpu_count()
DOWNLOAD_CONFIG = DownloadConfig(cache_dir=CACHE_DIR, resume_download=True)
#TRAIN_SIZE = 20_000 # number of text chunks, corresponds to ~100k sentences
#TEST_SIZE = int(0.1*TRAIN_SIZE)
# ^ NOTE: chunks are now like 6x larger

#SEEDS = [int(seed) for seed in np.random.randint(0, 10000, 10)]
SEEDS = [5904, 557, 1113, 4819, 6624, 269, 6429, 3929, 9085, 7906]


# helper function for tokenization
def tokenize_fn(examples, tokenizer):
    tokenized_inputs = tokenizer(
       examples['text'], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length
    )
    if tokenizer.is_fast:
        tokenized_inputs['word_ids'] = [tokenized_inputs.word_ids(i) for i in range(len(tokenized_inputs['input_ids']))]
    return tokenized_inputs


# main data processing function that will concatenate all texts from our dataset and generate chunks of model_max_length.
def group_texts(examples, tokenizer):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= tokenizer.model_max_length:
        total_length = (total_length // tokenizer.model_max_length) * tokenizer.model_max_length
    result = {
        k: [t[i : i + tokenizer.model_max_length] for i in range(0, total_length, tokenizer.model_max_length)]
        for k, t in concatenated_examples.items()
    }
    return result


# fine tune a base model on a particular dataset and save to disk
def fine_tune_single(base_model, dataset, model_name, tokenizer, num_epochs=4, seed=42):
    print(f'* Fine tuning with seed={seed}')
    output_dir = f'{DATA_DIR}final_models/bert/{model_name}_{seed}'

    # tokenize the data
    tokenized_dataset = dataset.map(tokenize_fn, fn_kwargs={'tokenizer': tokenizer}, batched=True, remove_columns=['text'], num_proc=NUM_PROC)
    tokenized_dataset = tokenized_dataset.map(group_texts, fn_kwargs={'tokenizer': tokenizer}, batched=True, num_proc=NUM_PROC)
    tokenized_dataset = tokenized_dataset.train_test_split(train_size=0.85, test_size=0.15, seed=seed)
    num_total_batches = tokenized_dataset['train'].num_rows
    print(f'** Split dataset into {num_total_batches} training chunks, each of size {tokenizer.model_max_length}')

    # set up training parameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=4,  # this * 4 = batch size
        learning_rate=5e-5,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        warmup_ratio=0.1,
        num_train_epochs=num_epochs,
        seed=seed,
        data_seed=seed,
        per_device_eval_batch_size=4,  # this * 4 = batch size
        evaluation_strategy='epoch',
        save_total_limit=1,
        save_strategy='no',
        load_best_model_at_end=False
    )


    # create trainer
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)
    )

    # train and save model
    trainer.train()
    trainer.save_model(output_dir)
    print(f'* Model saved to {output_dir}')


# perform fine_tune_single for multiple seeds
def fine_tune_multiple(base_model_name, dataset, model_name, tokenizer, num_epochs=4, seeds=SEEDS):
    print(f'Fine tuning {base_model_name} model on {model_name} dataset over {num_epochs} epochs')
    for seed in seeds:
        torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        base_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
        fine_tune_single(base_model, dataset, model_name, tokenizer, num_epochs, seed)
    print(f'Finished fine tuning {base_model_name} model on {model_name} dataset')


# datasets we'll be using
DATASET_NAMES = {
    'main': ['ted_talks', 'arxiv_abstracts', 'friends_scripts', 'childrens_lit', 'reuters_news'],
    'essays': ['extroversion_positive_assessment', 'extroversion_negative_assessment',
                'agreeableness_positive_assessment', 'agreeableness_negative_assessment',
                'conscientiousness_positive_assessment', 'conscientiousness_negative_assessment',
                'emotional_stability_positive_assessment', 'emotional_stability_negative_assessment',
                'openness_to_experience_positive_assessment', 'openness_to_experience_negative_assessment'],
    'model': ['bert_data']
}

'''
# fine tune on various datasets
for dataset_name in DATASET_NAMES['main']:
    base_model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.model_max_length=CHUNK_SIZE
    dataset = load_dataset('text', data_files=f'{DATA_DIR}input_data/cleaned_data/{dataset_name}_data.txt', download_config=DOWNLOAD_CONFIG)['train']
    fine_tune_multiple(base_model_name, dataset, dataset_name, tokenizer, num_epochs=4, seeds=SEEDS)
'''

for dataset_name in DATASET_NAMES['essays']:
    base_model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.model_max_length=CHUNK_SIZE
    dataset = load_dataset('text', data_files=f'{DATA_DIR}input_data/cleaned_data/{dataset_name}.txt', download_config=DOWNLOAD_CONFIG)['train']
    fine_tune_multiple(base_model_name, dataset, dataset_name, tokenizer, num_epochs=4, seeds=SEEDS)



