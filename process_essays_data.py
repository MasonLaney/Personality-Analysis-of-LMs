
# package imports
import json
import pandas as pd
import spacy

# useful constants
SEED = 42
DATA_DIR = '/ssd-playpen/mlaney/'
TRAITS = ['extroversion', 'agreeableness', 'conscientiousness', 'emotional stability', 'openness to experience']

# set up spaCy for sentence segmentation
# spacy.cli.download('en_core_web_sm') # only needs to be run once
nlp = spacy.load('en_core_web_sm')

# load input data
essays_df = pd.read_csv(DATA_DIR + 'input_data/raw_data/essays.csv', encoding='cp1252')

# function for segmenting essays into sentences and writing to a file
def process_essays(essay_list, output_dir):
    with open(output_dir, 'w') as f:
        for essay in essay_list:
            doc = nlp(essay)
            for sent in doc.sents:
                f.write(f'{sent}\n')



process_essays(essays_df['TEXT'].tolist(), DATA_DIR+'input_data/cleaned_data/essays.txt')

# # create datasets for positive/negative extroversion scores
# ext_pos = essays_df.loc[essays_df['cEXT'] == 'y']['TEXT'].tolist()
# ext_neg = essays_df.loc[essays_df['cEXT'] == 'n']['TEXT'].tolist()
# process_essays(ext_pos, DATA_DIR+'input_data/cleaned_data/ext_pos_essays.txt')
# process_essays(ext_neg, DATA_DIR+'input_data/cleaned_data/ext_neg_essays.txt')
#
# # create datasets for positive/negative agreeableness scores
# agr_pos = essays_df.loc[essays_df['cAGR'] == 'y']['TEXT'].tolist()
# agr_neg = essays_df.loc[essays_df['cAGR'] == 'n']['TEXT'].tolist()
# process_essays(agr_pos, DATA_DIR+'input_data/cleaned_data/agr_pos_essays.txt')
# process_essays(agr_neg, DATA_DIR+'input_data/cleaned_data/agr_neg_essays.txt')
#
# # create datasets for positive/negative conscientiousness scores
# con_pos = essays_df.loc[essays_df['cCON'] == 'y']['TEXT'].tolist()
# con_neg = essays_df.loc[essays_df['cCON'] == 'n']['TEXT'].tolist()
# process_essays(con_pos, DATA_DIR+'input_data/cleaned_data/con_pos_essays.txt')
# process_essays(con_neg, DATA_DIR+'input_data/cleaned_data/con_neg_essays.txt')
#
# # create datasets for positive/negative emotional stability scores
# emo_pos = essays_df.loc[essays_df['cNEU'] == 'n']['TEXT'].tolist()
# emo_neg = essays_df.loc[essays_df['cNEU'] == 'y']['TEXT'].tolist()
# process_essays(emo_pos, DATA_DIR+'input_data/cleaned_data/emo_pos_essays.txt')
# process_essays(emo_neg, DATA_DIR+'input_data/cleaned_data/emo_neg_essays.txt')
#
# # create datasets for positive/negative openness to experience scores
# opn_pos = essays_df.loc[essays_df['cOPN'] == 'y']['TEXT'].tolist()
# opn_neg = essays_df.loc[essays_df['cOPN'] == 'n']['TEXT'].tolist()
# process_essays(opn_pos, DATA_DIR+'input_data/cleaned_data/opn_pos_essays.txt')
# process_essays(opn_neg, DATA_DIR+'input_data/cleaned_data/opn_neg_essays.txt')

