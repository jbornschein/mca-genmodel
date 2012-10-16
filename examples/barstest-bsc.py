#!/usr/bin/env python
#
#  Author:   Marc Henniges
#        and Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: Academic Free License (AFL) v3.0
#

from __future__ import division

import sys
sys.path.insert(0, '..')

import numpy as np
from mpi4py import MPI

from random import sample

# Import 
from pulp.utils import create_output_path, check_basis
from pulp.utils.parallel import pprint

from pulp.utils.datalog import dlog, StoreToH5, TextPrinter, StoreToTxt
from pulp.visualize.gui import GUI, RFViewer, YTPlotter

from pulp.em import EM
from pulp.em.annealing import LinearAnnealing
from pulp.em.camodels.bsc_et import BSC_ET

# Datalogger presettings:
output_path = create_output_path()

# -------------------------------- Parameters ---------------------------------

# Number of datapoints:
N = 1000

# Number of pixels (will be squared for complete image size):
D2 = 5
D = D2**2

# Number of hidden causes (set automatically to 2*D2 for bars test):
H = 2*D2

# ET approximation parameters
Hprime = 5
gamma = 3

# np.random.normal(scale=W_noise_intensity, size=(H, D)) is added after each run
W_noise_intensity = 0.01
pi_noise_intensity = 0
sigma_noise_intensity = 0

# Values for bars, if performing bars test:
bar_value = 10.

# Model parameters:
pi_gt = 0.2
sigma_gt = 1.0
pi_learn = 0.2
sigma_learn = 5.0
#W_learn = np.random.normal(4,4./3,(H,D))
#W_learn = np.zeros((H,D))
#W_learn = 'ground_truth'
W_learn = 'mean'
mu_gt = np.zeros(D)
mu_learn = np.zeros(D)
#mu_learn = mu_gt

# Which parameters should be learned:
#to_learn = ['W', 'pi', 'sigma', 'mu']
to_learn = ['W', 'pi', 'sigma']
#to_learn = ['W']
#to_learn = ['W', 'pi']
#to_learn = ['W', 'sigma']
#to_learn = ['pi', 'sigma']
neg_bars = 0.5

# Annealing:
#start_temp = 13.0
start_temp = 1.0
end_temp = 1.0
anneal_steps = 50
anneal_prior = False

# Configure data logger below

# ----------------------------- Parameters end --------------------------------

# Main
if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    pprint("="*40)
    pprint(" Running %d parallel processes" % comm.size) 
    pprint("="*40)
    
    # -------------------------------- Controls -----------------------------------
    run_algorithm = True
    pprint("\n============================= Warnings =============================")
    if (not 'sigma' in to_learn) and (sigma_gt != sigma_learn):
        pprint("Generating Sigma not euqal to learned parameter. Aborting.")
        run_algorithm = False
    if (not 'pi' in to_learn) and (pi_gt != pi_learn):
        pprint("Generating Pi not euqal to learned parameter. Aborting.")
        run_algorithm = False
    if ('sigma' in to_learn) and (start_temp > 1.0):
        pprint("Starting temperature not 1 with activated Sigma learning. Aborting.")
        run_algorithm = False
    if ('W' in to_learn) and (W_noise_intensity == 0):
        pprint("W learned, but not noisified.")
    if ('pi' in to_learn) and (pi_noise_intensity == 0):
        pprint("Pi learned, but not noisified.")
    if ('sigma' in to_learn) and (sigma_noise_intensity == 0):
        pprint("Sigma learned, but not noisified.")
    if (not gamma <= Hprime) or (not Hprime <= H):
        pprint("gamma < Hprime < H has to be fullfilled. Aborting.")
        run_algorithm = False
    pprint("============================= ........ =============================\n")

    #if not run_algorithm:
    #    exit(1) 
    # ------------------------------ Controls end ---------------------------------
    
    # Configure DataLogger
    dlog.start_gui(GUI)
    print_list = ('Done', 'T', 'Q', 'pi', 'sigma', 'N', 'MAE')
    dlog.set_handler(print_list, TextPrinter)
    #dlog.set_handler(print_list, StoreToTxt, output_path +'terminal.txt')
    #dlog.set_handler('Q', YTPlotter)
    dlog.set_handler('W', RFViewer, rf_shape=(D2, D2))
    dlog.set_handler(('W', 'pi', 'sigma', 'mu', 'y', 'MAE', 'N'), StoreToH5, output_path +'result.h5')
    #dlog.set_handler(('W', 'pi', 'sigma', 'mu', 'y', 'MAE', 'N'), StoreToH5, output_path +'test_set.h5')
    dlog.set_handler(['pi'], YTPlotter)
    dlog.set_handler(['sigma'], YTPlotter)
    dlog.set_handler('y', RFViewer, rf_shape=(D2, D2))

    # Prepare ground-truth GFs (bars)
    W_gt = np.zeros( (H, D2, D2) )
    for i in xrange(D2):
        W_gt[   i, i, :] = bar_value
        W_gt[D2+i, :, i] = bar_value
    if neg_bars > 0.0:
        W_gt[sample(range(H), np.int(H*neg_bars))] *= -1
    W_gt = W_gt.reshape( (H, D) )
    W_gt += np.random.normal(size=(H, D), scale=0.5)
    
    # Prepare model...
    model = BSC_ET(D, H, Hprime, gamma, to_learn)
    mparams = {
        'W'     : W_gt,
        'pi'    : pi_gt,
        'sigma' : sigma_gt,
        'mu'    : mu_gt
    }
    mparams = comm.bcast(mparams)
    
    pprint("Generating Model Parameters:")
    pprint("pi = " + np.str(mparams['pi']) + "; sigma = " + np.str(mparams['sigma']))
    
    # Generate trainig data
    my_N = N // comm.size
    my_data = model.generate_data(mparams, my_N)
    dlog.append('y',my_data['y'][0:20])
    
    # Choose annealing schedule
    anneal = LinearAnnealing(anneal_steps)
    anneal['T'] = [(15, start_temp), (-10, end_temp)]
    anneal['Ncut_factor'] = [(0,0.),(2./3,1.)]
    anneal['anneal_prior'] = anneal_prior
    anneal['W_noise'] = [(0., W_noise_intensity), (0.9, W_noise_intensity), (1., 0.)]
    anneal['pi_noise'] = [(0., pi_noise_intensity), (0.9, pi_noise_intensity), (1., 0.)]
    anneal['sigma_noise'] = [(0., sigma_noise_intensity), (0.9, sigma_noise_intensity), (1., 0.)]
    
    mean_W = np.zeros((H,D))
    pics_per_H = my_N//H
    for indH in xrange(H):
        mean_W_tmp = np.sum(comm.allreduce(my_data['y'][indH*pics_per_H:(indH+1)*pics_per_H,:]),axis=0)/ (my_N // H * comm.size)
        mean_W[indH, :] = mean_W_tmp
    pies = 1./H

    # Initialize model parameters (to be learned)
    if W_learn == 'mean':
        lparams = {
            'W'        : mean_W,
            'pi'       : pi_learn,
            'sigma'    : sigma_learn,
            'mu'       : mu_learn
        }
    elif W_learn == 'ground_truth':
        lparams = {
            'W'        : W_gt,
            'pi'       : pi_gt,
            'sigma'    : sigma_gt,
            'mu'       : mu_gt
            }
    else:
        lparams = {
            'W'        : W_learn,
            'pi'       : pi_learn,
            'sigma'    : sigma_learn,
            'mu'       : mu_learn
            }
    lparams = model.noisify_params(lparams, anneal)    
    
    # Create and start EM annealing
    em = EM(model=model, anneal=anneal)
    em.data = my_data
    em.lparams = lparams
    em.run()
    pprint("Done with ET, Calculating MAE...")
    MAE = check_basis(em.lparams['W'], W_gt)
    if MAE == None:
        MAE = np.inf
    dlog.append('MAE', MAE[0])
    dlog.close(True)
    pprint("Done")
