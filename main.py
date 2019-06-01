#import gpu_select
import argparse
import numpy as np
import os
import random
import torch
import sys
import pdb
import yaml
import pprofile
from model import * #D, G, weights_init, Res_G, D_noro, skipG
from sandwich_trainer import SandwichTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='path to config file')

try:
    args = parser.parse_args()
except IOError as msg:
    parser.error(str(msg))

with open(args.config, 'r') as f:
    parsed = yaml.load(f)


parsed['out_dir'] = './results/aug_points_{0}_opt_{1}_lr_{2}_{3}_c_{4}_z1_{5}_z2_{6}_d_{7}_clip_{8}__pool_{9}_type_{10}_gradpen_{11}_Arc_{12}_invact_{16}_N_{13}_odim_{14}_obj_{15}'.format(
        parsed['num_points_per_object'],
        parsed['optimizer'],
        parsed['d_lr'],
        parsed['g_lr'],
        parsed['critic_steps'], 
        parsed['z1_dim'],
        parsed['z2_dim'],
        parsed['d_dim'],
        parsed['weight_clip'],
        parsed['pool'],
        parsed['type'],
        parsed['lambda_grad_pen'],
        parsed['arc'],
        parsed['n_obj'],
        parsed['o_dim'],
        parsed['obj'],
        parsed['invact']
        )
if not os.path.exists(parsed['out_dir']):
    os.makedirs(parsed['out_dir']) 

# Parse number of objects
if parsed['obj'] == 'multi10':
    parsed['obj'] = [1, 2, 8, 12, 14, 22, 23, 30, 33, 35]
elif parsed['obj'] == 'multi30':
    parsed['obj'] = list(range(30))
elif parsed['obj'] == 'multi40':
    parsed['obj'] = list(range(40))
else:
    parsed['obj'] = int(parsed['obj'])

# Print 
maxLen = max([len(ii) for ii in parsed.keys()])
fmtString = '\t%' + str(maxLen) + 's : %s'
with open(os.path.join(parsed['out_dir'],'log.txt'), 'w') as f:
    f.write(' '.join(sys.argv) + '\n\n')
    print('Arguments:')
    f.write('Arguments:\n')
    for keyPair in sorted(parsed.items()):
        print(fmtString % keyPair)
        f.write(fmtString % keyPair + '\n')

    d = skipD(x_dim=parsed['x_dim'], d_dim=parsed['d_dim'], z1_dim=parsed['z1_dim'], o_dim = 1)
    if parsed['arc'] == 'MLP':
      	g = G(z1_dim=parsed['z1_dim'], z2_dim=parsed['z2_dim'], x_dim=parsed['x_dim'])

    if parsed['invact'] == 'tanh':
    	g_inv = G_inv_Tanh(x_dim=parsed['x_dim'], d_dim=parsed['d_dim'], z1_dim=parsed['z1_dim'], pool=parsed['pool'])
    
    # write structure to log file
    f.write(d.__str__() + '\n')
    f.write(g.__str__() + '\n')
    f.write(g_inv.__str__() + '\n')


trainer = SandwichTrainer(g, d, g_inv, parsed)

# Start training

#prof = pprofile.Profile()
#with prof():
trainer.train()
#prof.print_stats()
