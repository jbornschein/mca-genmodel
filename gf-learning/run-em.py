#!/usr/bin/env python
#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: Academic Free License (AFL) v3.0
#

from __future__ import division

#TODO: All our libraries paths are set in a source file
import sys
sys.path.insert(0, '../lib')
sys.path.insert(0, '../../../pylib')

try:
    import numexpr as ne
    ne.set_num_threads(1)
except:
    pass

import sys
from shutil import copyfile

import numpy as np
from mpi4py import MPI
from tables import openFile

from pulp.em import EM
from pulp.em.annealing import LinearAnnealing

from pulp.utils import create_output_path
from pulp.utils.parallel import pprint
from pulp.utils.datalog import dlog, StoreToH5, StoreToTxt
from pulp.utils.autotable import AutoTable
from pulp.utils.processing import deconv
from pulp.utils import accel

import pulp.utils.tracing as tracing

from gabor_fitting import pgabor_fit
from dog_fitting import pdog_fit

comm = MPI.COMM_WORLD


#===============================================================================
# Main

paramfile = sys.argv[1]

dlog.progress("%s" % ' '.join(sys.argv) )
dlog.progress("Running %d parallel processes" % comm.size) 
dlog.progress("Using accelerted functions: %s" % accel.backend)
dlog.progress("Reading paramfile %s" % paramfile)
    
# Some default parameter values
#TODO: Agree upon default values... 
enable_tracing = True
partial_a = 1.  
partial_b = 1.
partial_c = 1.

# Read paramfile
execfile(paramfile)

#=================== Create output path and files ==============================
output_path = create_output_path(paramfile)
result_path = output_path + 'result.h5'    
if comm.rank == 0: #We keep a copy of this file mainly because of the ET params
    copyfile(paramfile, output_path+"paramfile.py")

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
data_file.close()

if locals().has_key('rev_corr') and rev_corr:
    data_file = openFile(rev_corr_datafile, 'r')
    #my_y_rc = data_file.root.oversized[first_y:last_y]
    my_y_rc = data_file.root.patches[first_y:last_y]
    
    orig_shape = my_y_rc.shape
    my_y_rc = my_y_rc.reshape(my_N, -1)

    # Shift into a mean free regime & normalize
    if rev_corr_mean_free:
        my_y_rc -= my_y_rc.mean(axis=-1)[:, None]
    if rev_corr_normalize:
        my_y_rc /= np.maximum( my_y_rc.max(axis=-1), -my_y_rc.min(axis=-1) )[:, None]
        my_y_rc *= 10

    my_y_rc = my_y_rc.reshape( orig_shape )
    
    data_file.close()
else:
    my_y_rc = my_y
    rev_corr = False
    

#============================ Check format ==================================
if my_y.shape[1] == my_y.shape[2]:
    # New format (4 Aug 2012) here:
    # patches are simply stored as D2 x D2 sized fields
    if 'data_factor' in dir():
        my_y *= data_factor
    
    if channel_splitted:
        D2 = my_y.shape[1]
        new_my_y = np.empty( (my_N,2*D2,D2))
        new_my_y[:,:D2,:] = my_y
        new_my_y[:,D2:,:] = -my_y
        new_my_y[new_my_y<0.0] = 0.0

        my_y = new_my_y
        del new_my_y
else:
    # Old format: 
    # patches are stored channel splitted
    if not channel_splitted:
        # Take care of not channel splitted case
        size = data_file.root.patches.shape[1] / 2
        my_y = my_y[:, 0:size, :] - my_y[:, size:, :]

# We flatten everything here in the D dimension
my_y    = my_y.reshape(my_N, -1)
my_y_rc = my_y_rc.reshape(my_N, -1)

_, D = my_y.shape


    

#========================= Configure DataLogger  ===============================
print_list = ('T', 'pi', 'sigma', 'N', 'N_use')
store_list = ('*')

dlog.set_handler(print_list, StoreToTxt, output_path+'/terminal.txt')
dlog.set_handler(store_list, StoreToH5, result_path)

#========================= Enable tracing? =====================================
if enable_tracing:
    dlog.progress("Tracing enabled")
    tracing.set_tracefile(output_path+"/trace-%04d.txt")

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
                            (noise_end, partial_c)] 

#========================= Instantiate model ============================== 
model = model_class(D, H, Hprime, gamma, to_learn)
model.rev_corr = rev_corr

if len(sys.argv) == 3:
    #=============== Pick up and continue previous computation ============
    dlog.progress("Picking up computation from %s" % sys.argv[2])
    tracing.tracepoint("Copy old reults")

    old_fname = sys.argv[2] + "/result.h5"
    old_h5 = openFile(old_fname, 'r')

    if comm.rank == 0:
        # Copy old results
        for node in old_h5.listNodes("/"):
            name = node.name
            rows = node.shape[0]
            if name in ['RF', 'gabor_params', 'gabor_errors', 'dog_params', 'dog_errors', 'dog_sigmas']:
                continue
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
    tracing.tracepoint("Estimate params")
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
em.data =  {'y': my_y, 'y_rc': my_y_rc}
em.lparams = lparams
em.run(verbose=True)
lparams = em.lparams

#dlog.close()
comm.Barrier()

#============================================================================
# Extract final generative fields (W)

if lparams.has_key("rev_corr") and lparams['rev_corr'].max() > 1e-7:
    dlog.progress("Using actual reverse correlated fields")
    RF = lparams["rev_corr"]

    H, D  = RF.shape

   # Undo channel splitting
    if channel_splitted:
        size = np.int(np.sqrt(D//2))
        RF = RF.reshape(H, 2*size, size)
        RF = RF[:, 0:size, :] - RF[:, size:2*size, :]
    else:
        size = np.int(np.sqrt(D))
        RF = RF.reshape(H, size, size)

else:
    dlog.progress("Using W to estimate reverse fields")
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

    # Do postprocessing: GFs -> RFs
    dlog.progress("Postprocessing: deconvolve")
    RF = deconv(s_p, s_m, GF, 10, mode='full')

dlog.append('RF', RF)

#=================== Gabor & DoG fit =========================================
if fit:
    dlog.progress("Fitting gabor kernels to RFs")
    gabor_params, gabor_errors = pgabor_fit(RF)
    
    old_seterr = np.seterr(all='ignore')
    dlog.progress("Fitting DoG kernels to RFs")
    dog_params, dog_errors, dog_sigmas = pdog_fit(RF)
    np.seterr(**old_seterr)

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

#=============================================================================
dlog.progress("Closing and archiving tracefiles")
tracing.close()
