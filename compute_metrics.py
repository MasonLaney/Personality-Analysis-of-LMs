
# package imports

import os
import torch
from evaluate import load

# local imports
from bert_models import BertModels

# useful constants
DATA_DIR = '/ssd-playpen/mlaney/'
METRICS = ['copa', 'rte', 'wic', 'wsc', 'wsc.fixed', 'boolq', 'axg']

# seeding
torch.manual_seed(42)
os.environ["PYTHONHASHSEED"] = "42"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)

# fetch the models and tokenizer
bertModelLoader = BertModels()
tokenizer = bertModelLoader.get_tokenizer()
models = bertModelLoader.get_models()

print(models)
test_model = models['default']



