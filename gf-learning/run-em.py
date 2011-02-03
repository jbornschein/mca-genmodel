#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

# Make sure our pulp library is in the path
import sys
sys.path.insert(0, '../')

from shutil import copyfile

import numpy as np
from mpi4py import MPI
from tables import openFile

from pulp.em import EM
from pulp.em.annealing import LinearAnnealing

from pulp.utils import create_output_path
from pulp.utils.parallel import pprint, sca_assim_load
from pulp.utils.datalog import dlog, StoreToH5, StoreToTxt
from pulp.utils.tracing import set_tracefile, tracepoint
from pulp.utils.autotable import AutoTable

#from pulp.utils.processing import deconv
#from gabor_fitting import pgabor_fit
#from dog_fitting import pdog_fit

comm = MPI.COMM_WORLD

#===============================================================================
# Main


    
# Some default parameter values
enable_tracing = False
anneal_prior = True
partial_a = 1.  
partial_b = 1.
partial_c = 1.

if len(sys.argv) < 2:
    pprint();
    pprint("Usage %s <paramfile> [<previous-output-dir-to-recover>]" % sys.argv[0])
    pprint();
    exit(1)

# Read paramfile
paramfile = sys.argv[1]
execfile(paramfile)

#=================== Create output path and files ==============================
output_path = create_output_path(paramfile)
result_path = output_path + 'result.h5'    
if comm.rank == 0: # We keep a copy of the parameter file
    copyfile(paramfile, output_path+"paramfile.py")

dlog.progress("Running %d parallel processes" % comm.size) 
dlog.progress("Reading paramfile %s" % paramfile)
dlog.progress("Output directory: %s" % output_path)

#============================= Load data =====================================
data_file = openFile(datafile, 'r')
N_data = data_file.root.patches.shape[0]

if N_data < N:
    dlog.progress("WARNING: N=%d chosen but only %d data points available. " % (N, N_data))
    N = N_data

# Partition the workload
my_N = N // comm.size
first_y = my_N * comm.rank
last_y = my_N * (comm.rank+1)
my_y = data_file.root.patches[first_y:last_y]

# Take care of not channel splitted case
if not channel_splitted:
    size = data_file.root.patches.shape[2]
    my_y = my_y[:, 0:size, :] - my_y[:, size:, :]

# We flatten everything in the D dimension
my_y = my_y.reshape(my_N, -1)
data_file.close()
_, D = my_y.shape

#========================= Configure DataLogger  ===============================
print_list = ('T', 'pi', 'sigma', 'N', 'N_use')
store_list = ('W', 'pi', 'sigma', 'N', 'N_use', 'Q', 'RF', 
     'gabor_params', 'gabor_errors', 
     'dog_params', 'dog_errors', 'dog_sigmas'
)
dlog.set_handler(print_list, StoreToTxt, output_path+'/terminal.txt')
dlog.set_handler(store_list, StoreToH5, result_path)

#========================= Enable tracing? =====================================
if enable_tracing:
    dlog.progress("Tracing enabled")
    set_tracefile(output_path+"/trace-%04d.txt")

#=========== Create annealing schedule if not done in param file ===============
if 'anneal' not in dir():
    anneal = LinearAnnealing(anneal_steps) 
    anneal['T'] = [(anneal_start, temp_start), (anneal_end, temp_end)] 
    anneal['Ncut_factor'] = [(cut_start, 0.),(cut_end, 1.)] 
    anneal['anneal_prior'] = anneal_prior 
    anneal['W_noise'] = [(noise_decrease, W_noise), (noise_end, 0.)] 
    anneal['pi_noise'] = [(noise_decrease, pi_noise), (noise_end, 0.)] 
    anneal['sigma_noise'] = [(noise_decrease, sigma_noise), (noise_end, 0.)] 
    anneal['partial'] = [(cut_start, partial_a), (cut_end, partial_b), 
                            (-11, partial_b), (-10, partial_c)] 

#========================= Instantiate model ============================== 
model = model_class(D, H, Hprime, gamma, to_learn)

if len(sys.argv) == 3:
    #=============== Pick up and continue previous computation ============
    dlog.progress("Picking up computation from %s" % sys.argv[2])
    tracepoint("Copy old reults")

    old_fname = sys.argv[2] + "/result.h5"
    old_h5 = openFile(old_fname, 'r')

    if comm.rank == 0:
        # Copy old results
        for node in old_h5.listNodes("/"):
            name = node.name
            rows = node.shape[0]
            for r in xrange(rows):
                dlog.append(name, node[r])

    # Extract current parameters
    steps_done = old_h5.root.W.shape[0]
    lparams = {
        'mu'    : np.zeros( (D,) ),
        'W'     : old_h5.root.W[-1],
        'pi'    : old_h5.root.pi[-1],
        'sigma' : old_h5.root.sigma[-1],
    }
    old_h5.close()

    # Advance annealing schedule
    dlog.progress("Skipping %d EM iterations" % steps_done)
    for i in xrange(steps_done-1):
        anneal.next()
else:
    #======================= Initial model parameters =======================
    tracepoint("Estimate params")
    dlog.progress("Estimating initial parameters")
    lparams = model.standard_init({'y' : my_y})
    lparams = model.noisify_params(lparams, anneal=anneal)

    if W_init != "estimate": lparams['W'] = W_init
    if pi_init != "estimate": lparams['pi'] = pi_init
    if sigma_init != "estimate": lparams['sigma'] = sigma_init

    #As a control, we also append the initial values
    dlog.append_all( {
        'W' : lparams['W'], 
        'pi': lparams['pi'],
        'sigma': lparams['sigma'],
    } )

comm.Barrier()

#==================== Create and run EM =====================================
dlog.progress("Starting EM")

em = EM(model=model, anneal=anneal)
em.data =  {'y': my_y}
em.lparams = lparams
em.run(verbose=True)
lparams = em.lparams

#dlog.close()
comm.Barrier()

#============================================================================
# Extract final generative fields (W)
GF = lparams["W"]
H, D  = GF.shape

# Undo channel splitting
if channel_splitted:
    size = np.int(np.sqrt(D//2))
    GF = GF.reshape(H, 2*size, size)
    GF = GF[:, 0:size, :] - GF[:, size:2*size, :]
else:
    size = np.int(np.sqrt(D))
    GF = GF.reshape(H, size, size)

#============================================================================
# Do postprocessing: GFs -> RFs
tracepoint("Postprocessing")

if processing == 'reverse-corr':
    dlog.progress("Postprocessing: reverse-corr")
    #Process with the original D
    #the output is single channeld (H, size, size)
    raise
elif processing == 'deconvolve':
    #Really fast operation, scatering the results would take more time 
    #than computing the same on all the processors.
    dlog.progress("Postprocessing: deconvolve")
    RF = deconv(s_p, s_m, GF, 6*s_p)
else:
    dlog.progress("Postprocessing: NONE")
    RF = GF

dlog.append('RF', RF)

#=================== Gabor & DoG fit =========================================
if fit:
    dlog.progress("Fitting gabor kernels to RFs")

    gabor_params, gabor_errors = pgabor_fit(RF)
    
    dog_params, dog_errors, dog_sigmas = pdog_fit(RF)

    # Store results...
    if comm.rank == 0:
        num_gabors = gabor_params.shape[0]
        for h in xrange(num_gabors):
            dlog.append_all( {
                'gabor_params'  : gabor_params[h],
                'gabor_errors'  : gabor_errors[h],
                'dog_params'    : dog_params[h],
                'dog_errors'    : dog_errors[h],
                'dog_sigmas'    : dog_sigmas[h]
            } )

tracepoint("Finished")
