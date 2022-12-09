
import os
import sys

file = '/tmp/pycharm_project_620/fine_tune_models.py'
output = '/ssd-playpen/mlaney/nohup/fine_tune_output.txt'
#cmd = f"nohup sh -c '{sys.executable} -u {file} > {output}' &"
#cmd = f"nohup sh -c '{sys.executable} {file}' &"
#cmd = f"nohup {sys.executable} {file} > {output} &"
#cmd = f"{sys.executable} {file} ... {output} & disown -h"
#cmd = f"nohup {file} > {output} &"
cmd = f"nohup sh -c '{sys.executable} {file} > {output}'"

print(cmd)
os.system(cmd)
