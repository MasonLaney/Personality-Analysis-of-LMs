# adapted from this tutorial: https://huggingface.co/blog/pretraining-bert#1-prepare-the-dataset

# package imports
import datasets
import gc
import multiprocessing
import os
import torch
from datasets import load_dataset
from itertools import chain
from tokenizers import Tokenizer
#from tokenizers import BertWordPieceTokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, \
    Trainer, BertTokenizerFast
from transformers import BertConfig, BertTokenizerFast
from tqdm import tqdm

# clear the CUDA cache to avoid OOM errors
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
gc.collect()
torch.cuda.empty_cache()

# useful constants
SEED = 42
DATA_DIR = '/ssd-playpen/mlaney/'
CACHE_DIR = DATA_DIR+'cache/'
DOWNLOAD_CONFIG = datasets.DownloadConfig(cache_dir=CACHE_DIR, resume_download=True)

# seeding
torch.manual_seed(42)
os.environ['PYTHONHASHSEED'] = str(SEED)

# retrieve datasets and combine into one
wikipedia_dataset = load_dataset('wikipedia', '20220301.en', cache_dir=CACHE_DIR, download_config=DOWNLOAD_CONFIG, split='train')
wikipedia_dataset = wikipedia_dataset.remove_columns([col for col in wikipedia_dataset.column_names if col != 'text'])
bookcorpus_dataset = load_dataset('bookcorpus', cache_dir=CACHE_DIR, download_config=DOWNLOAD_CONFIG, split='train')
assert bookcorpus_dataset.features.type == wikipedia_dataset.features.type
raw_datasets = datasets.concatenate_datasets([bookcorpus_dataset, wikipedia_dataset])

# repositor id for saving the tokenizer
tokenizer_id = 'bert_base_uncased_recreation'

# create a python generator to dynamically load the data
def batch_iterator(batch_size=10000):
    for i in tqdm(range(0, len(raw_datasets), batch_size)):
        yield raw_datasets[i : i + batch_size]['text']

# create a tokenizer from existing one to re-use special tokens
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# train the tokenizer
bert_tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=32_000)
bert_tokenizer.save_pretrained('bert_base_uncased_recreation_tokenizer')

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert_base_uncased_recreation_tokenizer')
num_proc = multiprocessing.cpu_count()
print(f'The max length for the tokenizer is: {tokenizer.model_max_length}')

# first grouping function
def first_group_texts(examples):
    tokenized_inputs = tokenizer(
       examples['text'], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length
    )
    return tokenized_inputs

# preprocess dataset
tokenized_datasets = raw_datasets.map(first_group_texts, batched=True, remove_columns=['text'], num_proc=num_proc)

# main data processing function that will concatenate all texts from our dataset and generate chunks of max_seq_length.
def group_texts(examples):
    # TODO: this is expecting seperate documents, currently just getting one lone doc?
    # concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # we drop the small remainder, we could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    if total_length >= tokenizer.model_max_length:
        total_length = (total_length // tokenizer.model_max_length) * tokenizer.model_max_length
    # split by chunks of max_len.

    result = {
        k: [t[i : i + tokenizer.model_max_length] for i in range(0, total_length, tokenizer.model_max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc)
tokenized_datasets = tokenized_datasets.train_test_split(train_size=0.009, test_size=0.001, seed=SEED)
print(f'the dataset contains in total {len(tokenized_datasets)*tokenizer.model_max_length} tokens')
# TODO: change sizes back

# create untrained model
config = BertConfig()
config.max_position_embeddings = 512  # TODO: change back for last 100k
# ^ tried running 128, wouldn't work bc of tokenization
model = AutoModelForMaskedLM.from_config(config=config)

# set up training parameters
training_args = TrainingArguments(
    output_dir=DATA_DIR + 'bert_base_uncased_recreation_small',
    overwrite_output_dir=True,
    do_train=True,
    learning_rate=1e-4,
    adam_beta1=0.9,
    adam_beta2=0.999,
    weight_decay=0.01,
    warmup_steps=10000,
    per_device_train_batch_size=256,
    max_steps=1000000,
    logging_strategy='steps',
    logging_steps=500,
    log_level='debug'
)

# seq len of 128 for 90% of steps
# remaining 10% of steps on 512

DATA_COLLATOR = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    data_collator=DATA_COLLATOR
)

# train and save model
trainer.train()
#trainer.save_model('bert_base_uncased_recreation_final')
print('Training from scratch completed!')

