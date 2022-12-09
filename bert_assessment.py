# for BERT, we are using MLM to "take" the personality assessments

# package imports
import json
import math
import numpy as np
import os
import torch
from torch.nn import functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

# local imports
from bert_models import BertModels

# string that points to data directory
DATA_DIR = '/ssd-playpen/mlaney/'

# seeding
torch.manual_seed(42)
os.environ["PYTHONHASHSEED"] = "42"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)

# fetch the model and tokenizer
bertModelLoader = BertModels()
tokenizer = bertModelLoader.get_tokenizer()
default_model = bertModelLoader.get_model('default')
ted_talks_fine_tuned_model = bertModelLoader.get_model('ted_talks', from_scratch=False)
ted_talks_from_scratch_model = bertModelLoader.get_model('ted_talks', from_scratch=True)
arxiv_abstracts_fine_tuned_model = bertModelLoader.get_model('arxiv_abstracts', from_scratch=False)
arxiv_abstracts_from_scratch_model = bertModelLoader.get_model('arxiv_abstracts', from_scratch=True)
friends_scripts_fine_tuned_model = bertModelLoader.get_model('friends_scripts', from_scratch=False)
friends_scripts_from_scratch_model = bertModelLoader.get_model('friends_scripts', from_scratch=True)
childrens_lit_fine_tuned_model = bertModelLoader.get_model('childrens_lit', from_scratch=False)
childrens_lit_from_scratch_model = bertModelLoader.get_model('childrens_lit', from_scratch=True)
reuters_news_fine_tuned_model = bertModelLoader.get_model('reuters_news', from_scratch=False)
reuters_news_from_scratch_model = bertModelLoader.get_model('reuters_news', from_scratch=True)


# NOTE: directly lifted from Graham's work
def get_probabilities_for_sentence(sentence, model):
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
def evaluate_big_five_assessment(model_name, from_scratch=True):

    model = bertModelLoader.get_model(model_name, from_scratch)
    model.eval()
    output_sentences = {}
    answers_idx = {0: 'always', 1: 'often', 2: 'sometimes', 3: 'rarely', 4: 'never'}

    with open(DATA_DIR + 'input_data/big_five_assessment/bert_assessment_items.json', 'r') as f:
        input_sentences = json.load(f)

    prior_scores = bertModelLoader.calculate_priors(model)

    for key, sentence in input_sentences.items():

        output = get_probabilities_for_sentence(sentence, model)
        output_prob = [i / sum(output) for i in output]
        output_lp = [-math.log(i) for i in output_prob]

        std_output_prob = [i/j for i, j in zip(output_prob, prior_scores['scores'])]
        std_output_prob = [i/sum(std_output_prob) for i in std_output_prob]
        std_output_lp = [-math.log(i) for i in std_output_prob]

        output_sentences.update({key: {'sentence': input_sentences[key], 'prob': output_prob,
                                       'std_prob': std_output_prob, 'nlog_prob': output_lp,
                                       'std_nlog_prob': std_output_lp}})
        output_sentences[key].update({'min_score': min(output_sentences[key]['nlog_prob']),
                                      'std_min_score': min(output_sentences[key]['std_nlog_prob']),
                                      'min_score_idx': "%d" % np.argmin(np.asarray(output_sentences[key]['nlog_prob'])),
                                      'std_min_score_idx': "%d" % np.argmin(np.asarray(output_sentences[key]['std_nlog_prob'])),
                                      'min_score_word': answers_idx[np.argmin(np.asarray(output_sentences[key]['nlog_prob']))],
                                      'std_min_score_word': answers_idx[np.argmin(np.asarray(output_sentences[key]['std_nlog_prob']))]})

    return output_sentences


# TODO: docstring
def score_big_five_assessment(model_name, from_scratch=True):

    results = evaluate_big_five_assessment(model_name, from_scratch)

    traits = {'extroversion': {'score': 20, 'std_score': 20, 'description': 'Extroversion (E) is the personality trait of seeking fulfillment from sources outside the self or in community. High scorers tend to be very social while low scorers prefer to work on their projects alone.'},
         'agreeableness': {'score': 14, 'std_score': 14, 'description': 'Agreeableness (A) reflects how much individuals adjust their behavior to suit others. High scorers are typically polite and like people. Low scorers tend to tell it like it is.'},
         'conscientiousness': {'score': 14, 'std_score': 14, 'description': 'Conscientiousness (C) is the personality trait of being honest and hardworking. High scorers tend to follow rules and prefer clean homes. Low scorers may be messy and cheat others.'},
         'neuroticism': {'score': 38, 'std_score': 38, 'description': 'Neuroticism (N) is the personality trait of being emotional. High scorers tend to have high emotional reactions to stress. They may perceive situations as threatening and be more likely to feel moody, depressed, angry, anxious, and experience mood swing. Low scorers tend to be more emotionally stable and less reactive to stress.'},
         'openness to experience': {'score': 8, 'std_score':8, 'description': 'Openness to Experience (O) is the personality trait of seeking new experiences and intellectual pursuits. High scores may day dream a lot (enjoy thinking about new and different things). Low scorers tend to be very down to earth (more of a ‘hear and now’ thinker). Consequently, it is thought that people with higher scores might be more creative, flexible, curious, and adventurous, whereas people with lower score might tend to enjoy routines, predictability, and structure.'}}

    contributing_questions = {'extroversion': {'positive': [1, 11, 21, 31, 41], 'negative': [6, 16, 26, 36, 46]},
        'agreeableness': {'positive': [7, 17, 27, 37, 42, 47], 'negative': [2, 12, 22, 32]},
        'conscientiousness': {'positive': [3, 13, 23, 33, 43, 48], 'negative': [8, 18, 28, 38]},
        'neuroticism': {'positive': [9, 19], 'negative': [4, 14, 24, 29, 34, 39, 44, 49]},
        'openness to experience': {'positive': [5, 15, 25, 35, 40, 45, 50], 'negative': [10, 20, 30]}}

    # NOTE: even though it's labeled as 'neuroticism' here, the source the descriptions were taken from is incorrect
    # and the score reflects emotional stability (i.e. higher scores on this "neuroticism" trait actually mean
    # higher emotional stability, contrary from what seems intuitive.

    answer_points = {'always': 5, 'often': 4, 'sometimes': 3, 'rarely': 2, 'never': 1}

    for question_type, question_numbers in contributing_questions.items():

        for num in question_numbers['positive']:
            traits[question_type]['score'] += answer_points[results[str(num)]['min_score_word']]
            traits[question_type]['std_score'] += answer_points[results[str(num)]['std_min_score_word']]

        for num in question_numbers['negative']:
            traits[question_type]['score'] -= answer_points[results[str(num)]['min_score_word']]
            traits[question_type]['std_score'] -= answer_points[results[str(num)]['std_min_score_word']]

    return traits

'''
test_evaluation_def = score_big_five_assessment('default')
print(bertModelLoader.calculate_priors(default_model))
print(json.dumps(test_evaluation_def, indent=4))

test_evaluation_ft = score_big_five_assessment('reuters_news', False)
print(bertModelLoader.calculate_priors(reuters_news_fine_tuned_model))
print(json.dumps(test_evaluation_ft, indent=4))

test_evaluation_fs = score_big_five_assessment('reuters_news', True)
print(bertModelLoader.calculate_priors(reuters_news_from_scratch_model))
print(json.dumps(test_evaluation_fs, indent=4))

test_evaluation_pos = score_big_five_assessment('openness_to_experience_positive_reddit')
print(json.dumps(test_evaluation_pos, indent=4))

test_evaluation_neg = score_big_five_assessment('openness_to_experience_negative_reddit')
print(json.dumps(test_evaluation_neg, indent=4))

print(json.dumps(score_big_five_assessment('extroversion_positive_assessment'), indent=4))
print(json.dumps(score_big_five_assessment('extroversion_negative_assessment'), indent=4))
'''

