
# performance settings
import os
import gc
import torch
print('Number of unreachable objects collected by GC:', gc.collect())
torch.cuda.device(2)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
torch.cuda.empty_cache()
#torch.use_deterministic_algorithms(True)

# package imports
import csv
import json
import math
import multiprocessing
import numpy as np
from datasets import DownloadConfig
from torch.nn import functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

# useful constants
SEED = 42
DATA_DIR = '/ssd-playpen/mlaney/'
CACHE_DIR = DATA_DIR+'cache/'
CHUNK_SIZE = 512 # 512
NUM_PROC = multiprocessing.cpu_count()
DOWNLOAD_CONFIG = DownloadConfig(cache_dir=CACHE_DIR, resume_download=True)
SEEDS = [5904, 557, 1113, 4819, 6624, 269, 6429, 3929, 9085, 7906]


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
    print([softmax[mask_idx][a].item(), softmax[mask_idx][o].item(), softmax[mask_idx][s].item(), softmax[mask_idx][r].item(), softmax[mask_idx][n].item()])
    exit()
    return [softmax[mask_idx][a].item(), softmax[mask_idx][o].item(), softmax[mask_idx][s].item(), softmax[mask_idx][r].item(), softmax[mask_idx][n].item()]
    # ^ The values returned are the perplexity of the single word.
    #   Since rest of sentence is same, don't need perplexity of whole sentence


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


# datasets we'll be using
DATASET_NAMES = {
    'main': ['ted_talks', 'arxiv_abstracts', 'friends_scripts', 'childrens_lit', 'reuters_news'],
    'essays': ['ext_pos_essays', 'ext_neg_essays',
               'agr_pos_essays', 'agr_neg_essays',
               'con_pos_essays', 'con_neg_essays',
               'emo_pos_essays', 'emo_neg_essays',
               'opn_pos_essays', 'opn_neg_essays'],
    'assessment': ['extroversion_positive_assessment', 'extroversion_negative_assessment',
                'agreeableness_positive_assessment', 'agreeableness_negative_assessment',
                'conscientiousness_positive_assessment', 'conscientiousness_negative_assessment',
                'emotional_stability_positive_assessment', 'emotional_stability_negative_assessment',
                'openness_to_experience_positive_assessment', 'openness_to_experience_negative_assessment']
}


# load models and perform assessment of them
def assess_models(base_model_name, model_names, seeds=SEEDS):

    headers = ['seed', 'extroversion', 'agreeableness', 'conscientiousness', 'emotional stability', 'openness to experience']
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForMaskedLM.from_pretrained(base_model_name)

    for model_name in model_names:

        models = {'base': base_model}
        for seed in seeds:
            print(f"{model_name} {seed}")
            models[str(seed)] = AutoModelForMaskedLM.from_pretrained(f'{DATA_DIR}final_models/bert/{model_name}_{seed}')
            print(models[str(seed)])

        results = []
        for seed, model in models.items():
            seed_results = score_big_five_assessment_modified(model, tokenizer)
            row = [seed] + [val for key, val in seed_results['max_answer_scores'].items()]
            results.append(row)

        means = []
        stdevs = []
        for i in range(1, 6):
            means.append(np.mean([row[i] for row in results[1:]]))
            stdevs.append(np.std([row[i] for row in results[1:]]))
        statistics = [['mean'] + means, ['stdev'] + stdevs]

        with open(f'{DATA_DIR}final_assessment_results/bert/{model_name}.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(headers)
            writer.writerows(results)
            writer.writerows(statistics)


assess_models('bert-base-uncased', ['ted_talks'], SEEDS)

