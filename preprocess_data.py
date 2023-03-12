
# package imports
import os
import pandas as pd
import random
import re
import spacy
from datetime import datetime

# useful constants
SEED = 42
DATA_DIR = '/ssd-playpen/mlaney/'
#DATA_DIR = './local_files/'

# seeding
os.environ['PYTHONHASHSEED'] = '42'

# set up spaCy for sentence segmentation
#spacy.cli.download('en_core_web_sm')
nlp = spacy.load('en_core_web_sm')

'''
# preprocess the TED Talks data (roughly 450,000k sentences)
ted_talk_df = pd.read_csv(DATA_DIR+'input_data/raw_data/talks_info.csv').sample(frac=1.0, random_state=SEED)
ted_talk_list = ted_talk_df[ted_talk_df['transcript'].notna()]['transcript'].tolist()
ted_talk_data = []
for doc in ted_talk_list:
    ted_talk_doc = nlp(doc)
    assert ted_talk_doc.has_annotation('SENT_START')
    ted_talk_data += [sent.text for sent in ted_talk_doc.sents]
with open(DATA_DIR+'input_data/cleaned_data/ted_talks_data.txt', 'w') as f:
    for sent in ted_talk_data:
        f.write(f"{sent}\n")

# preprocess the arXiv abstract data (roughly ____ sentences)
print('Before processing arXiv: ' + str(datetime.now()))
arxiv_df = pd.read_json(DATA_DIR+'input_data/raw_data/arxiv-metadata-oai-snapshot.json', lines=True).sample(frac=0.012, random_state=SEED)
arxiv_list = arxiv_df[arxiv_df['abstract'].notna()]['abstract'].tolist()
arxiv_data = []
print('Number of abstracts in arXiv: ' + str(len(arxiv_list)))
for doc in arxiv_list:
    arxiv_doc = nlp(doc)
    assert arxiv_doc.has_annotation('SENT_START')
    arxiv_data += [sent.text for sent in arxiv_doc.sents]
with open(DATA_DIR+'input_data/cleaned_data/arxiv_abstracts_data.txt', 'w') as f:
    for sent in arxiv_data:
        f.write(f"{sent}\n")
print('Number of sentences in arXiv: ' + str(len(arxiv_data)))
print('After processing arXiv: ' + str(datetime.now()))
'''

# preprocess the Friends scripts data (roughly ____ sentences)
print('Before processing FRIENDS: ' + str(datetime.now()))
friends_scripts_list = []
for filename in os.listdir(DATA_DIR+'input_data/raw_data/friends_scripts'):
    with open(DATA_DIR+'input_data/raw_data/friends_scripts/'+filename, 'r') as f:
        # removes which character is delivering in lines and stage directions
        friends_scripts_list.append('\n'.join([re.sub(r"([\(\[]).*?([\)\]])", '', ' '.join(line.split()[1:])) for line in f.readlines() if line.strip()]))
random.Random(SEED).shuffle(friends_scripts_list)
with open(DATA_DIR+'input_data/cleaned_data/friends_scripts_data.txt', 'w') as f:
    for script in friends_scripts_list:
        f.write(f"{script}\n")
print('Number of sentences in FRIENDS: ' + str(sum([len(script.split('\n')) for script in friends_scripts_list])))
print('After processing FRIENDS: ' + str(datetime.now()))

# preprocess the Children's literature data (roughly ____ sentences)
print('Before processing Children\'s Lit: ' + str(datetime.now()))
childrens_lit_list = []
for filename in os.listdir(DATA_DIR+'input_data/raw_data/childrens_literature'):
    with open(DATA_DIR+'input_data/raw_data/childrens_literature/'+filename, 'r') as f:
        f.readline()
        f.readline()
        childrens_lit_list.append('\n'.join([line for line in f.readlines() if line.strip()]))
random.Random(SEED).shuffle(childrens_lit_list)
with open(DATA_DIR+'input_data/cleaned_data/childrens_lit_data.txt', 'w') as f:
    for book in childrens_lit_list:
        f.write(f"{book}\n")
print('Number of sentences in Children\'s lit: ' + str(sum([len(book.split('\n')) for book in childrens_lit_list])))
print('After processing Children\'s Lit: ' + str(datetime.now()))

# preprocess the Reuters news data (roughly ____ sentences)
print('Before processing Reuter\'s: ' + str(datetime.now()))
reuters_news_list = []
for filename in os.listdir(DATA_DIR+'input_data/raw_data/reuters_news'):
    with open(DATA_DIR+'input_data/raw_data/reuters_news/'+filename, 'r') as f:
        reuters_news_list.append(' '.join([line for line in f.readlines() if line.strip()]))
random.Random(SEED).shuffle(reuters_news_list)
with open(DATA_DIR+'input_data/cleaned_data/reuters_news_data.txt', 'w') as f:
    for article in reuters_news_list:
        f.write(f"{article}\n")
print('Number of *articles* in Reuters news: ' + str(len(reuters_news_list)))
print('After processing Reuter\'s: ' + str(datetime.now()))
