
# package imports
import json

# useful constants
SEED = 42
DATA_DIR = '/ssd-playpen/mlaney/'
TRAITS = ['extroversion', 'agreeableness', 'conscientiousness', 'emotional stability', 'openness to experience']
MASK = '[MASK]'

# load Big 5 assessment data
with open(DATA_DIR + 'input_data/big_five_assessment/bert_assessment_items.json', 'r') as f:
    assessment_statements = json.load(f)

# corresponding traits for each question
contributing_questions = {'extroversion': {'positive': [1, 11, 21, 31, 41], 'negative': [6, 16, 26, 36, 46]},
                          'agreeableness': {'positive': [7, 17, 27, 37, 42, 47], 'negative': [2, 12, 22, 32]},
                          'conscientiousness': {'positive': [3, 13, 23, 33, 43, 48], 'negative': [8, 18, 28, 38]},
                          'emotional stability': {'positive': [9, 19], 'negative': [4, 14, 24, 29, 34, 39, 44, 49]},
                          'openness to experience': {'positive': [5, 15, 25, 35, 40, 45, 50], 'negative': [10, 20, 30]}}

# NOTE: even though it's labeled as 'neuroticism' here, the source the descriptions were taken from is incorrect
# and the score reflects emotional stability (i.e. higher scores on this "neuroticism" trait actually mean
# higher emotional stability, contrary from what seems intuitive.

# create dicts to store output data
positive_statements = {}
negative_statements = {}
for trait in TRAITS:
    positive_statements[trait] = []
    negative_statements[trait] = []

# categorize statements based on positive or negative association with each trait
for trait, trait_data in contributing_questions.items():
    print(trait)
    for num in trait_data['positive']:
        masked_statement = assessment_statements[str(num)]
        positive_statements[trait].append(masked_statement.replace(MASK, 'always'))
        positive_statements[trait].append(masked_statement.replace(MASK, 'often'))
        negative_statements[trait].append(masked_statement.replace(MASK, 'rarely'))
        negative_statements[trait].append(masked_statement.replace(MASK, 'never'))
    for num in trait_data['negative']:
        masked_statement = assessment_statements[str(num)]
        positive_statements[trait].append(masked_statement.replace(MASK, 'never'))
        positive_statements[trait].append(masked_statement.replace(MASK, 'rarely'))
        negative_statements[trait].append(masked_statement.replace(MASK, 'often'))
        negative_statements[trait].append(masked_statement.replace(MASK, 'always'))

# write to files, one response per line
# TODO: change to less hacky method of dealing with extremely small datasets
for trait in TRAITS:
    with open(DATA_DIR+'input_data/cleaned_data/'+trait.replace(' ', '_')+'_positive_assessment.txt', 'w') as f:
        for statement in positive_statements[trait]:
            for i in range(100):
                f.write(f"{statement}\n")
        print('Number of sentences in ' + trait + ' positive: ' + str(len(positive_statements[trait])))
    with open(DATA_DIR+'input_data/cleaned_data/'+trait.replace(' ', '_')+'_negative_assessment.txt', 'w') as f:
        for statement in negative_statements[trait]:
            for i in range(100):
                f.write(f"{statement}\n")
        print('Number of sentences in ' + trait + ' negative: ' + str(len(negative_statements[trait])))
