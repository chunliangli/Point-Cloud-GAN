import os
import numpy as np
from subprocess import check_output

# 1. get the list of gpu and status
gpu_query_columns = ('index', 'memory.used', 'memory.total')
gpu_list = []

smi_output = check_output(
    r'nvidia-smi --query-gpu={query_cols} --format=csv,noheader,nounits'.format(
        query_cols=','.join(gpu_query_columns),
    ), shell=True).decode().strip()

for line in smi_output.split('\n'):
    if not line: continue
    query_results = line.split(',')
    g = {col_name: col_value.strip() for (col_name, col_value) in zip(gpu_query_columns, query_results)}
    gpu_list.append(g)
free_mem = {int(g['index']):int(g['memory.total'])-int(g['memory.used']) for g in gpu_list}

np_free_mem = np.zeros(len(gpu_list))
for i in range(len(gpu_list)):
    np_free_mem[i] = free_mem[i]
min_idx = np.argmax(np_free_mem)

if np_free_mem[min_idx]< 1000:
    print('WARNING: No GPU entirely free!')

print('Using GPU {0}'.format(min_idx))

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(min_idx)
