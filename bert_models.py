# class to create and fetch default and fine-tuned BERT models

# TODO: notes
# For GPT, build a BPE tokenizer instead of a WordPiece tokenizer


# package imports
import datasets
import gc
import os
import torch
from datasets import load_dataset
from tokenizers import Tokenizer
#from tokenizers import BertWordPieceTokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from transformers import DistilBertConfig, DistilBertTokenizerFast

# clear the CUDA cache to avoid OOM errors
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
gc.collect()
torch.cuda.empty_cache()

# useful constants
SEED = 42
DATA_DIR = '/ssd-playpen/mlaney/'
CACHE_DIR = DATA_DIR+'cache/'
CHUNK_SIZE = 128
BATCH_SIZE = 8
TRAIN_SIZE = 20_000 # number of text chunks, corresponds to ~100k sentences
TEST_SIZE = int(0.1*TRAIN_SIZE)
RERUN_MODELS = False
DOWNLOAD_CONFIG = datasets.DownloadConfig(cache_dir=CACHE_DIR, resume_download=True)
# NOTE: ^ this didn't actually work for some reason, only changing the environment variables in PyCharm did

# set environment variables
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR+'transformers'
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR+'datasets'
# NOTE: ^ this also didn't actually work

# seeding
torch.manual_seed(42)
os.environ['PYTHONHASHSEED'] = "42"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)

# create default (non-fine-tuned) model and save it
DEFAULT_MODEL_CHECKPOINT = 'distilbert-base-uncased'
DEFAULT_MODEL = AutoModelForMaskedLM.from_pretrained(DEFAULT_MODEL_CHECKPOINT)
DEFAULT_MODEL.save_pretrained(DATA_DIR + 'bert_models/default')
TOKENIZER = AutoTokenizer.from_pretrained(DEFAULT_MODEL_CHECKPOINT)
DATA_COLLATOR = DataCollatorForLanguageModeling(tokenizer=TOKENIZER, mlm_probability=0.15)


# helper function for tokenization
def tokenize_fn(examples, tokenizer):
    result = TOKENIZER(examples['text'])
    if tokenizer.is_fast:
        result['word_ids'] = [result.word_ids(i) for i in range(len(result['input_ids']))]
    return result


# sentence chunking (drops the last chunk if it's smaller than the chunk size)
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // CHUNK_SIZE) * CHUNK_SIZE
    result = {
        k: [t[i : i + CHUNK_SIZE] for i in range(0, total_length, CHUNK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    result['labels'] = result['input_ids'].copy()
    return result


# function for converting data (in the form of txt files) into a Dataset
def create_dataset(filepath, tokenizer=TOKENIZER, name='text'):
    dataset = load_dataset('text', data_files=DATA_DIR+filepath, download_config=DOWNLOAD_CONFIG)
    # TODO: change so that it reads from this new dataset folder that we've just created?
    dataset.save_to_disk(DATA_DIR+'input_data/datasets/'+name)
    tokenized_dataset = dataset.map(tokenize_fn, fn_kwargs={'tokenizer': tokenizer}, batched=True, remove_columns=['text'])
    lm_dataset = tokenized_dataset.map(group_texts, batched=True)
    num_total_batches = lm_dataset['train'].num_rows
    if num_total_batches < TRAIN_SIZE:
        print('Warning: dataset at ' + filepath + ' contains only ' + str(num_total_batches) + ' batches.')
        downsampled_dataset = lm_dataset['train'].train_test_split(train_size=0.9, test_size=0.1, seed=SEED)
    else:
        downsampled_dataset = lm_dataset['train'].train_test_split(train_size=TRAIN_SIZE, test_size=TEST_SIZE, seed=SEED)
    downsampled_dataset = downsampled_dataset.remove_columns(['word_ids'])
    return downsampled_dataset


# combines the datasets used to train DistilBERT and samples from them
def get_distilbert_dataset(tokenizer=TOKENIZER):

    # retrieve datasets and combine into one
    wikipedia_dataset = load_dataset('wikipedia', '20220301.en', cache_dir=CACHE_DIR, download_config=DOWNLOAD_CONFIG, split='train')
    wikipedia_dataset = wikipedia_dataset.remove_columns([col for col in wikipedia_dataset.column_names if col != 'text'])
    bookcorpus_dataset = load_dataset('bookcorpus', cache_dir=CACHE_DIR, download_config=DOWNLOAD_CONFIG, split='train')
    assert bookcorpus_dataset.features.type == wikipedia_dataset.features.type
    dataset = datasets.concatenate_datasets([bookcorpus_dataset, wikipedia_dataset])

    tokenized_dataset = dataset.map(tokenize_fn, fn_kwargs={'tokenizer': tokenizer}, batched=True, remove_columns=['text'])
    lm_dataset = tokenized_dataset.map(group_texts, batched=True)
    num_total_batches = lm_dataset.num_rows
    print('Number of chunks: ' + str(num_total_batches))

    downsampled_dataset = lm_dataset.train_test_split(train_size=0.009, test_size=0.001, seed=SEED)
    downsampled_dataset = downsampled_dataset.remove_columns(['word_ids'])
    print(downsampled_dataset)
    return downsampled_dataset


# fine-tunes a model on the given dataset and saves it to output_dir
def create_fine_tuned_model(input_filepath, output_dir, name='text'):

    # retrieve dataset and set up pre-trained model
    dataset = create_dataset(input_filepath, TOKENIZER, name=name)
    model = AutoModelForMaskedLM.from_pretrained(DEFAULT_MODEL_CHECKPOINT)

    # set up training parameters
    logging_steps = len(dataset['train']) // BATCH_SIZE
    training_args = TrainingArguments(
        output_dir=DATA_DIR+output_dir,
        overwrite_output_dir=True,
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=True,
        logging_steps=logging_steps)

    # create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=DATA_COLLATOR
    )

    # train and save model
    trainer.train()
    trainer.save_model(DATA_DIR+output_dir)
    print('Fine-tuning completed!')


# trains a model from scratch on the given dataset and saves it to output_dir
def create_from_scratch_model(input_filepath, output_dir, name='text'):

    # TODO: save tokenizer to file? will save a few seconds
    # train custom tokenizer
    tokenizer = Tokenizer(WordPiece(unk_token='[UNK]'))
    trainer = WordPieceTrainer(special_tokens=['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]'])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train([DATA_DIR+input_filepath], trainer)

    # add post-processing to tokenizer
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    # retrieve dataset and create untrained model
    dataset = create_dataset(input_filepath, name=name)
    config = DistilBertConfig() # can change these hyperparameters if desired
    model = AutoModelForMaskedLM.from_config(config=config)

    # set up training parameters
    training_args = TrainingArguments(
        output_dir=DATA_DIR + output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True
    )

    # create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=DATA_COLLATOR
    )

    # train and save model
    trainer.train()
    trainer.save_model(DATA_DIR+output_dir)
    print('Training from scratch completed!')


# trains a model from scratch on the dataset used to train DistilBERT
# TODO: combine into create_from_scratch_model()?
def recreate_distilbert_from_scratch():

    output_dir = 'bert_models/from_scratch/distilbert_recreation'

    # retrieve dataset and create untrained model
    dataset = get_distilbert_dataset()
    config = DistilBertConfig() # can change these hyperparameters if desired
    model = AutoModelForMaskedLM.from_config(config=config)

    # set up training parameters
    training_args = TrainingArguments(
        output_dir=DATA_DIR + output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True
    )

    # create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=DATA_COLLATOR
    )

    # train and save model
    trainer.train()
    trainer.save_model(DATA_DIR+output_dir)
    print('Training from scratch completed!')


# fine-tunes DistilBERT on the data it was trained on originally
# TODO: combine into create_fine_tuned_model()?
def fine_tune_distilbert_further():

    output_dir = 'bert_models/fine_tuned/distilbert_further'

    # retrieve dataset and set up pre-trained model
    dataset = get_distilbert_dataset()
    model = AutoModelForMaskedLM.from_pretrained(DEFAULT_MODEL_CHECKPOINT)

    # set up training parameters
    logging_steps = len(dataset['train']) // BATCH_SIZE
    training_args = TrainingArguments(
        output_dir=DATA_DIR + output_dir,
        overwrite_output_dir=True,
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=True,
        logging_steps=logging_steps)

    # create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=DATA_COLLATOR
    )

    # train and save model
    trainer.train()
    trainer.save_model(DATA_DIR + output_dir)
    print('Fine-tuning completed!')


recreate_distilbert_from_scratch()
#fine_tune_distilbert_further()


# train the models
if RERUN_MODELS:

    # create main models
    for model_name in ['ted_talks', 'arxiv_abstracts', 'friends_scripts', 'childrens_lit', 'reuters_news']:
        create_fine_tuned_model('input_data/cleaned_data/' + model_name + '_data.txt', 'bert_models/fine_tuned/' + model_name, model_name)
        create_from_scratch_model('input_data/cleaned_data/' + model_name + '_data.txt', 'bert_models/from_scratch/' + model_name, model_name)

    # create reddit models
    for trait in ['extroversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness_to_experience']:
        create_fine_tuned_model('input_data/cleaned_data/' + trait + '_positive_reddit.txt', 'bert_models/fine_tuned/' + trait + '_positive_reddit', trait + '_positive_reddit')
        create_fine_tuned_model('input_data/cleaned_data/' + trait + '_negative_reddit.txt', 'bert_models/fine_tuned/' + trait + '_negative_reddit', trait + '_negative_reddit')


# class for retrieving BERT models
# TODO: refine this
class BertModels:

    def __init__(self):
        self.supported_models = ['default', 'ted_talks', 'arxiv_abstracts', 'friends_scripts', 'childrens_lit', 'reuters_news']
        self.reddit_models = ['extroversion_positive_reddit', 'extroversion_negative_reddit',
                              'agreeableness_positive_reddit', 'agreeableness_negative_reddit',
                              'conscientiousness_positive_reddit', 'conscientiousness_negative_reddit',
                              'neuroticism_positive_reddit', 'neuroticism_negative_reddit',
                              'openness_to_experience_positive_reddit', 'openness_to_experience_negative_reddit']
        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_CHECKPOINT)

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self, name, from_scratch=False):
        assert name in self.supported_models or name in self.reddit_models
        if name == 'default':
            model = AutoModelForMaskedLM.from_pretrained(DATA_DIR + 'bert_models/' + name)
        # NOTE: due to small data size, reddit models only support fine-tuning, regardless of the value of from_scratch
        elif name in self.reddit_models:
            model = AutoModelForMaskedLM.from_pretrained(DATA_DIR + 'bert_models/fine_tuned/' + name)
        elif from_scratch:
            model = AutoModelForMaskedLM.from_pretrained(DATA_DIR + 'bert_models/from_scratch/' + name)
        else:
            model = AutoModelForMaskedLM.from_pretrained(DATA_DIR + 'bert_models/fine_tuned/' + name)
        return model

    def get_models(self):
        models = {}
        for model_name in self.supported_models:
            if model_name == 'default':
                models['default'] = self.get_model('default')
            else:
                models[model_name + '_fine_tuned'] = self.get_model(model_name, from_scratch=False)
                models[model_name + '_from_scratch'] = self.get_model(model_name, from_scratch=True)
        return models

    def get_reddit_models(self):
        models = {}
        for model_name in self.reddit_models:
            models[model_name] = self.get_model(model_name)
        return models

    def calculate_priors(self, model):
        tokenizer = self.get_tokenizer()
        sentence = ' [MASK] '
        input_idx = tokenizer.encode_plus(sentence, return_tensors="pt")
        logits = model(**input_idx).logits.squeeze()
        a = tokenizer.encode('always')[1]
        o = tokenizer.encode('often')[1]
        s = tokenizer.encode('sometimes')[1]
        r = tokenizer.encode('rarely')[1]
        n = tokenizer.encode('never')[1]
        softmax = F.softmax(logits, dim=-1)
        mask_idx = torch.where(input_idx["input_ids"][0] == tokenizer.mask_token_id)[0].item()
        output = [softmax[mask_idx][a].item(), softmax[mask_idx][o].item(), softmax[mask_idx][s].item(), softmax[mask_idx][r].item(), softmax[mask_idx][n].item()]
        return {'scores': [i/sum(output) for i in output]}
