
# package imports
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

# local imports
#import constants as C
from datasets import DownloadConfig
SEED = 42
DATA_DIR = '/ssd-playpen/mlaney/'
CACHE_DIR = DATA_DIR+'cache/'
DOWNLOAD_CONFIG = DownloadConfig(cache_dir=CACHE_DIR, resume_download=True)

from fine_tune_models import fine_tune_and_assess, fine_tune_and_assess_modified

# load datasets
essays_names = ['ext_pos_essays', 'ext_neg_essays', 'agr_pos_essays', 'agr_neg_essays', 'con_pos_essays', 'con_neg_essays',
                'emo_pos_essays', 'emo_neg_essays', 'opn_pos_essays', 'opn_neg_essays']
essays_datasets = {}
for name in essays_names:
    essays_datasets[name] = load_dataset('text', data_files=f'{DATA_DIR}input_data/cleaned_data/{name}.txt', download_config=DOWNLOAD_CONFIG)['train']

# fine-tune and assess on essays datasets
for name, dataset in essays_datasets.items():
    fine_tune_and_assess_modified(AutoModelForMaskedLM.from_pretrained('bert-base-uncased'), dataset, AutoTokenizer.from_pretrained('bert-base-uncased'), name, 10)
