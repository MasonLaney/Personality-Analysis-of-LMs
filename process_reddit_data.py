
# package imports
import json

# useful constants
SEED = 42
DATA_DIR = '/ssd-playpen/mlaney/'
TRAITS = ['extroversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness to experience']

# load Reddit context data from files
with open(DATA_DIR+'input_data/human_data/reddit_context.json') as f:
    reddit_text = json.load(f)

with open(DATA_DIR+'input_data/human_data/bert_traits_reddit.json') as f:
    reddit_traits = json.load(f)

# combine Reddit text and corresponding Big 5 scores into one dict
reddit_data = {}
for key, text in reddit_text['context'].items():
    reddit_data[key] = {'text': text}

for question_type, question_data in reddit_traits['question type'].items():
    for key, scores in question_data['context_keys'].items():
        reddit_data[key][question_type] = scores

# create dicts to store output data
positive_texts = {}
negative_texts = {}
for trait in TRAITS:
    positive_texts[trait] = []
    negative_texts[trait] = []

# categorize text based on positive or negative association with each trait
# NOTE: high neuroticism here actaully does mean neuroticism, not emotional stability
for key, info in reddit_data.items():
    for trait in TRAITS:
        # NOTE: currently basing off of unstandardized scores
        if info[trait]['diff'] > 0:
            positive_texts[trait].append(info['text'])
        elif info[trait]['diff'] < 0:
            negative_texts[trait].append(info['text'])

# write to files, one response per line
for trait in TRAITS:
    with open(DATA_DIR+'input_data/cleaned_data/'+trait.replace(' ', '_')+'_positive_reddit.txt', 'w') as f:
        for response in positive_texts[trait]:
            f.write(f"{response}\n")
        print('Number of sentences in ' + trait + ' positive: ' + str(len(positive_texts[trait])))
    with open(DATA_DIR+'input_data/cleaned_data/'+trait.replace(' ', '_')+'_negative_reddit.txt', 'w') as f:
        for response in negative_texts[trait]:
            f.write(f"{response}\n")
        print('Number of sentences in ' + trait + ' negative: ' + str(len(negative_texts[trait])))
