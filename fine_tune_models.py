##!/ssd-playpen/mlaney/.virtualenvs/senior_honors_thesis/bin/python

import os
import gc
n = gc.collect()
print('Number of unreachable objects collected by GC:', n)

# clear the CUDA cache to avoid OOM errors
#os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# ^ stopped working? replaced with torch.cuda.device

import torch
torch.cuda.device(2)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
torch.cuda.empty_cache()


# package imports
import json
import math
import multiprocessing
import datasets
import numpy as np
from datasets import DownloadConfig, load_dataset
from itertools import chain
from torch.nn import functional as F
from transformers import TrainingArguments, Trainer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, \
    AutoTokenizer, BertTokenizerFast


# useful constants
SEED = 42
DATA_DIR = '/ssd-playpen/mlaney/'
CACHE_DIR = DATA_DIR+'cache/'
CHUNK_SIZE = 512 # 512
NUM_PROC = multiprocessing.cpu_count()
#TRAIN_SIZE = 20_000 # number of text chunks, corresponds to ~100k sentences
#TEST_SIZE = int(0.1*TRAIN_SIZE)
# ^ NOTE: chunks are now like 6x larger
DOWNLOAD_CONFIG = DownloadConfig(cache_dir=CACHE_DIR, resume_download=True)


# seeding
#SEED = np.random.randint(low=0, high=100000)
#torch.manual_seed(SEED)
#os.environ['PYTHONHASHSEED'] = str(SEED)
#print(os.environ)


# NOTE: first_group_texts from recreate_bert
def tokenize_fn(examples, tokenizer):
    tokenized_inputs = tokenizer(
       examples['text'], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length
    )
    if tokenizer.is_fast:
        tokenized_inputs['word_ids'] = [tokenized_inputs.word_ids(i) for i in range(len(tokenized_inputs['input_ids']))]
    return tokenized_inputs


# main data processing function that will concatenate all texts from our dataset and generate chunks of max_seq_length.
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


# TODO: docstring
def fine_tune(base_model, dataset, tokenizer, output_dir, seed=SEED):

    # calculate the output directory
    output_dir = DATA_DIR + 'bert_models/new/' + output_dir

    # tokenize the data
    tokenized_dataset = dataset.map(tokenize_fn, fn_kwargs={'tokenizer': tokenizer}, batched=True, remove_columns=['text'], num_proc=NUM_PROC)
    tokenized_dataset = tokenized_dataset.map(group_texts, fn_kwargs={'tokenizer': tokenizer}, batched=True, num_proc=NUM_PROC)
    tokenized_dataset = tokenized_dataset.train_test_split(train_size=0.9, test_size=0.1, seed=seed)
    num_total_batches = tokenized_dataset['train'].num_rows
    print(f'Number of training chunks: {num_total_batches}, each of size {tokenizer.model_max_length}')

    '''
    # NOTE: here we are reducing all datasets to the same max size for consistency
    # ^ we may want to rethink this
    if num_total_batches < TRAIN_SIZE:
        print(f'Warning: dataset \'{dataset.builder_name}\' contains only {num_total_batches} training batches.')
        downsampled_dataset = tokenized_dataset['train'].train_test_split(train_size=0.9, test_size=0.1, seed=seed)
    else:
        downsampled_dataset = tokenized_dataset['train'].train_test_split(train_size=TRAIN_SIZE, test_size=TEST_SIZE, seed=seed)
    downsampled_dataset = downsampled_dataset.remove_columns(['word_ids'])
    '''
    downsampled_dataset = tokenized_dataset.remove_columns(['word_ids'])

    # set up training parameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=4,  # this * 4 = batch size
        learning_rate=5e-5,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        num_train_epochs=1,
        warmup_ratio=0.1,
        seed=seed,
        data_seed=seed,
        logging_dir=output_dir+'/logs'
    )

    # create trainer
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=downsampled_dataset['train'],
        eval_dataset=downsampled_dataset['test'],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)
    )

    # train and save model
    trainer.train()
    trainer.save_model(output_dir)


# NOTE: taken from Graham's work
def get_probabilities_for_sentence(sentence, model, tokenizer):
    input_idx = tokenizer.encode_plus(sentence, return_tensors='pt')
    logits = model(**input_idx).logits.squeeze()
    a = tokenizer.encode('always')[1]
    o = tokenizer.encode('often')[1]
    s = tokenizer.encode('sometimes')[1]
    r = tokenizer.encode('rarely')[1]
    n = tokenizer.encode('never')[1]
    softmax = F.softmax(logits, dim = -1)
    mask_idx = torch.where(input_idx['input_ids'][0] == tokenizer.mask_token_id)[0].item()
    return [softmax[mask_idx][a].item(), softmax[mask_idx][o].item(), softmax[mask_idx][s].item(), softmax[mask_idx][r].item(), softmax[mask_idx][n].item()]
    # ^ The values returned are the perplexity of the single word.
    #   Since rest of sentence is same, don't need perplexity of whole sentence


# TODO: docstring
def take_big_five_assessment(model, tokenizer, assessment_file):

    model.eval()
    output_sentences = {}
    answers_idx = {0: 'always', 1: 'often', 2: 'sometimes', 3: 'rarely', 4: 'never'}

    with open(f'{DATA_DIR}input_data/big_five_assessment/{assessment_file}', 'r') as f:
        input_sentences = json.load(f)

    for key, sentence in input_sentences.items():

        output = get_probabilities_for_sentence(sentence, model, tokenizer)
        output_prob = [i / sum(output) for i in output]
        output_lp = [-math.log(i) for i in output_prob]

        output_sentences.update({key: {'sentence': input_sentences[key], 'prob': output_prob, 'nlog_prob': output_lp}})
        output_sentences[key].update({'min_score': min(output_sentences[key]['nlog_prob']),
                                      'min_score_idx': "%d" % np.argmin(np.asarray(output_sentences[key]['nlog_prob'])),
                                      'min_score_word': answers_idx[np.argmin(np.asarray(output_sentences[key]['nlog_prob']))]})

    return output_sentences


# TODO: docstring
def score_big_five_assessment(model, tokenizer):

    results = take_big_five_assessment(model, tokenizer, 'bert_assessment_items.json')

    max_answer_scores = {'extroversion': 20, 'agreeableness': 14, 'conscientiousness': 14, 'emotional stability': 38, 'openness to experience': 8}
    avg_answer_scores = {'extroversion': 20, 'agreeableness': 14, 'conscientiousness': 14, 'emotional stability': 38, 'openness to experience': 8}
    contributing_questions = {'extroversion': {'positive': [1, 11, 21, 31, 41], 'negative': [6, 16, 26, 36, 46]},
        'agreeableness': {'positive': [7, 17, 27, 37, 42, 47], 'negative': [2, 12, 22, 32]},
        'conscientiousness': {'positive': [3, 13, 23, 33, 43, 48], 'negative': [8, 18, 28, 38]},
        'emotional stability': {'positive': [9, 19], 'negative': [4, 14, 24, 29, 34, 39, 44, 49]},
        'openness to experience': {'positive': [5, 15, 25, 35, 40, 45, 50], 'negative': [10, 20, 30]}}
    answer_points = {'always': 5, 'often': 4, 'sometimes': 3, 'rarely': 2, 'never': 1}

    for trait, question_numbers in contributing_questions.items():
        for num in question_numbers['positive']:
            max_answer_scores[trait] += answer_points[results[str(num)]['min_score_word']]
            avg_answer_scores[trait] += np.average(np.multiply(results[str(num)]['prob'], list(answer_points.values())))
        for num in question_numbers['negative']:
            max_answer_scores[trait] -= answer_points[results[str(num)]['min_score_word']]
            avg_answer_scores[trait] -= np.average(np.multiply(results[str(num)]['prob'], list(answer_points.values())))

    return {'max_answer_scores': max_answer_scores, 'avg_answer_scores': avg_answer_scores, 'full_data': results}


# TODO: docstring
def score_big_five_assessment_modified(model, tokenizer):

    results = take_big_five_assessment(model, tokenizer, 'bert_assessment_items_modified.json')

    max_answer_scores = {'extroversion': 14, 'agreeableness': -4, 'conscientiousness': 14, 'emotional stability': 38, 'openness to experience': -4}
    avg_answer_scores = {'extroversion': 14, 'agreeableness': -4, 'conscientiousness': 14, 'emotional stability': 38, 'openness to experience': -4}
    contributing_questions = {'extroversion': {'positive': [1, 6, 11, 21, 31, 36], 'negative': [16, 26, 41, 46]},
        'agreeableness': {'positive': [2, 7, 17, 22, 27, 32, 37, 42, 47], 'negative': [12]},
        'conscientiousness': {'positive': [3, 13, 23, 33, 43, 48], 'negative': [8, 18, 28, 38]},
        'emotional stability': {'positive': [9, 19], 'negative': [4, 14, 24, 29, 34, 39, 44, 49]},
        'openness to experience': {'positive': [5, 15, 20, 25, 30, 35, 40, 45, 50], 'negative': [10]}}
    answer_points = {'always': 5, 'often': 4, 'sometimes': 3, 'rarely': 2, 'never': 1}

    for trait, question_numbers in contributing_questions.items():
        for num in question_numbers['positive']:
            max_answer_scores[trait] += answer_points[results[str(num)]['min_score_word']]
            avg_answer_scores[trait] += np.average(np.multiply(results[str(num)]['prob'], list(answer_points.values())))
        for num in question_numbers['negative']:
            max_answer_scores[trait] -= answer_points[results[str(num)]['min_score_word']]
            avg_answer_scores[trait] -= np.average(np.multiply(results[str(num)]['prob'], list(answer_points.values())))

    return {'max_answer_scores': max_answer_scores, 'avg_answer_scores': avg_answer_scores, 'full_data': results}



# TODO: docstring
def fine_tune_and_assess(base_model, dataset, tokenizer, output_dir, num_epochs, seed=SEED):

    full_data = {}
    base_results = score_big_five_assessment(base_model, tokenizer)
    full_data[0] = base_results['full_data']
    with open(f'{DATA_DIR}assessment_results/{output_dir}.txt', 'a') as f:
        f.write(f'Base: {base_results["max_answer_scores"]}\n')
    print(f'Base: {base_results["max_answer_scores"]}\n')


    model = base_model
    for i in range(num_epochs):
        fine_tune(model, dataset, tokenizer, output_dir, seed)
        model = AutoModelForMaskedLM.from_pretrained(DATA_DIR + 'bert_models/new/' + output_dir)
        results = score_big_five_assessment(model, tokenizer)
        full_data[i+1] = results['full_data']
        print(f'Epoch {i+1}: {results["max_answer_scores"]}\n')
        with open(f'{DATA_DIR}assessment_results/{output_dir}.txt', 'a') as f:
            f.write(f'Epoch {i+1}: {results["max_answer_scores"]}\n')

    with open(f'{DATA_DIR}assessment_results/{output_dir}_full_data.json', 'w') as f:
        json.dump(full_data, f)

    return full_data


# TODO: docstring
def fine_tune_and_assess_modified(base_model, dataset, tokenizer, output_dir, num_epochs, seed=SEED):

    full_data = {}
    base_results = score_big_five_assessment_modified(base_model, tokenizer)
    full_data[0] = base_results['full_data']
    with open(f'{DATA_DIR}modified_assessment_results/{output_dir}.txt', 'a') as f:
        f.write(f'Base: {base_results["max_answer_scores"]}\n')
    print(f'Base: {base_results["max_answer_scores"]}\n')

    model = base_model
    for i in range(num_epochs):
        fine_tune(model, dataset, tokenizer, output_dir, seed)
        model = AutoModelForMaskedLM.from_pretrained(DATA_DIR + 'bert_models/new/' + output_dir)
        results = score_big_five_assessment(model, tokenizer)
        full_data[i+1] = results['full_data']
        print(f'Epoch {i+1}: {results["max_answer_scores"]}\n')
        with open(f'{DATA_DIR}modified_assessment_results/{output_dir}.txt', 'a') as f:
            f.write(f'Epoch {i+1}: {results["max_answer_scores"]}\n')

    with open(f'{DATA_DIR}modified_assessment_results/{output_dir}_full_data.json', 'w') as f:
        json.dump(full_data, f)

    return full_data


def multiple_assessment_modified(base_model, dataset, tokenizer, output_dir, num_epochs, seeds):

    total_data = {}
    for seed in seeds:

        torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        print(os.environ)

        total_data[seed] = fine_tune_and_assess_modified(AutoModelForMaskedLM.from_pretrained('bert-base-uncased'), dataset, AutoTokenizer.from_pretrained('bert-base-uncased'), 'multiple/'+output_dir, num_epochs, seed)
        with open(f'{DATA_DIR}modified_assessment_results/multiple/{output_dir}.txt', 'a') as f:
            f.write(f'ASSESSMENT RESULTS WITH SEED={seed}:\n')
        # ^ note: prints this after the results accidentally

    with open(f'{DATA_DIR}modified_assessment_results/multiple/{output_dir}_total_data.json', 'w') as f:
        json.dump(total_data, f)


'''
OLD
# fine-tune on subset of BERT dataset
print('Fine-tuning on 1% of BERT dataset')
wikipedia_dataset = load_dataset('wikipedia', '20220301.en', cache_dir=CACHE_DIR, download_config=DOWNLOAD_CONFIG, split='train[:1%]')
wikipedia_dataset = wikipedia_dataset.remove_columns([col for col in wikipedia_dataset.column_names if col != 'text'])
bookcorpus_dataset = load_dataset('bookcorpus', cache_dir=CACHE_DIR, download_config=DOWNLOAD_CONFIG, split='train[:1%]')
assert bookcorpus_dataset.features.type == wikipedia_dataset.features.type
bert_dataset = datasets.concatenate_datasets([bookcorpus_dataset, wikipedia_dataset])
fine_tune_and_assess(AutoModelForMaskedLM.from_pretrained('bert-base-uncased'), bert_dataset, AutoTokenizer.from_pretrained('bert-base-uncased'), 'bert_data', 10)
'''

print('Running fine-tuning script')
#seeds = [int(seed) for seed in np.random.randint(0, 10000, 10)]
SEEDS = [5904, 557, 1113, 4819, 6624, 269, 6429, 3929, 9085, 7906]


# fine-tune on normal datasets
for data_name in ['ted_talks', 'arxiv_abstracts', 'friends_scripts', 'childrens_lit', 'reuters_news']:

    print(f'Fine-tuning on {data_name} dataset')
    dataset = load_dataset('text', data_files=f'{DATA_DIR}input_data/cleaned_data/{data_name}_data.txt', download_config=DOWNLOAD_CONFIG)['train']

    multiple_assessment_modified(AutoModelForMaskedLM.from_pretrained('bert-base-uncased'), dataset, AutoTokenizer.from_pretrained('bert-base-uncased'), data_name, 10, seeds)

'''
# fine-tune on assessment datasets
for data_name in ['extroversion_positive_assessment', 'extroversion_negative_assessment',
                  'agreeableness_positive_assessment', 'agreeableness_negative_assessment',
                  'conscientiousness_positive_assessment', 'conscientiousness_negative_assessment',
                  'emotional_stability_positive_assessment', 'emotional_stability_negative_assessment',
                  'openness_to_experience_positive_assessment', 'openness_to_experience_negative_assessment']:
    print(f'Fine-tuning on {data_name} dataset')
    dataset = load_dataset('text', data_files=f'{DATA_DIR}input_data/cleaned_data/{data_name}_data.txt', download_config=DOWNLOAD_CONFIG)['train']
    multiple_assessment_modified(AutoModelForMaskedLM.from_pretrained('bert-base-uncased'), dataset, AutoTokenizer.from_pretrained('bert-base-uncased'), data_name, 10, seeds)
'''

'''
# fine-tune on subset of BERT dataset
print('Fine-tuning on 1% of BERT dataset')
wikipedia_dataset = load_dataset('wikipedia', '20220301.en', cache_dir=CACHE_DIR, download_config=DOWNLOAD_CONFIG, split='train')
wikipedia_dataset = wikipedia_dataset.remove_columns([col for col in wikipedia_dataset.column_names if col != 'text'])
bookcorpus_dataset = load_dataset('bookcorpus', cache_dir=CACHE_DIR, download_config=DOWNLOAD_CONFIG, split='train')
assert bookcorpus_dataset.features.type == wikipedia_dataset.features.type
bert_dataset = datasets.concatenate_datasets([bookcorpus_dataset, wikipedia_dataset])
subset_length = int(0.01*len(bert_dataset))
bert_dataset = bert_dataset.shuffle(seed=SEED).select(range(subset_length))
fine_tune_and_assess_modified(AutoModelForMaskedLM.from_pretrained('bert-base-uncased'), bert_dataset, AutoTokenizer.from_pretrained('bert-base-uncased'), 'bert_data', 10)
'''

'''
data_name = 'ted_talks'
dataset = load_dataset('text', data_files=f'{DATA_DIR}input_data/cleaned_data/{data_name}_data.txt', download_config=DOWNLOAD_CONFIG)['train']
fine_tune_and_assess(AutoModelForMaskedLM.from_pretrained('bert-base-uncased'), dataset, AutoTokenizer.from_pretrained('bert-base-uncased'), data_name, 10)
'''

print('Finished running fine-tuning script')
