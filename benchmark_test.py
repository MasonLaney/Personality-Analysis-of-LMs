# package imports
import jiant.utils.python.io as py_io
import jiant.proj.simple.runscript as simple_run
import jiant.scripts.download_data.runscript as downloader
from jiant.tasks.constants import SUPERGLUE_TASKS
import os

# local imports
from bert_models import BertModels

# constants
EXP_DIR = '/ssd-playpen/mlaney/benchmarks/superglue'
DATA_DIR = '/ssd-playpen/mlaney/benchmarks/superglue/tasks'

# seeding
os.environ["PYTHONHASHSEED"] = "42"

# get the currently implemented models
bertModelLoader = BertModels()
models = bertModelLoader.get_models()

# download the SuperGLUE data
downloader.download_data(SUPERGLUE_TASKS, DATA_DIR)

args = simple_run.RunConfiguration(
    run_name='default_superglue',
    exp_dir=EXP_DIR,
    data_dir=DATA_DIR,
    hf_pretrained_model_name_or_path='/ssd-playpen/mlaney/bert_models/default',
    test_tasks=SUPERGLUE_TASKS,
)
simple_run.run_simple(args)
