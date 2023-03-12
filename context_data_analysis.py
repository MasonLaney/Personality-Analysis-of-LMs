# package imports
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

# local imports
import constants as C
from fine_tune_models import fine_tune_and_assess

# load datasets
context_names = ['context_ext_pos', 'context_ext_neg', 'context_agr_pos', 'context_agr_neg', 'context_con_pos', 'context_con_neg',
                'context_emo_pos', 'context_emo_neg', 'context_opn_pos', 'context_opn_neg']
context_datasets = {}
for name in context_names:
    context_datasets[name] = load_dataset('text', data_files=f'{C.DATA_DIR}input_data/cleaned_data/context_datasets/{name}.txt', download_config=C.DOWNLOAD_CONFIG)['train']

# fine-tune and assess on essays datasets
for name, dataset in context_datasets.items():
    fine_tune_and_assess(AutoModelForMaskedLM.from_pretrained('bert-base-uncased'), dataset, AutoTokenizer.from_pretrained('bert-base-uncased'), name, 10)

