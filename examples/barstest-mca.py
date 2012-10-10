#!/usr/bin/env python
#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: Academic Free License (AFL) v3.0
#

from __future__ import division

import sys
sys.path.insert(0, '..')

import numpy as np
from mpi4py import MPI
from matplotlib import use

from pulp.utils.parallel import pprint

from pulp.utils.datalog import dlog, StoreToH5, TextPrinter
from pulp.visualize.gui import GUI, RFViewer, YTPlotter

from pulp.em import EM
from pulp.em.annealing import LinearAnnealing
from pulp.em.camodels.mca_et import MCA_ET

# Parameters
N = 1000
D2 = 5
D = D2**2
H = 2*D2

Hprime = 4
gamma = 3

Tsteps = 40
Tstart = 3.
Tend = 1.0


#============================================================================
# Main

if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    pprint("="*78)
    pprint(" Running %d parallel processes" % comm.size) 
    pprint("="*78)

    #Configure DataLogger
    #use('GTKAgg')
    dlog.start_gui(GUI)

    #dlog.set_handler('freeEnergy', YTPlotter)
    dlog.set_handler(('T', 'Qmean', 'pi', 'sigma', 'Wmin', 'Wmean', 'Wmax'), TextPrinter)
    dlog.set_handler('W', RFViewer, rf_shape=(D2, D2), symmetric=0, global_maximum=0)
    #dlog.set_handler('y', RFViewer, rf_shape=(D2, D2))

    # Choose annealing schedule
    anneal = LinearAnnealing(Tsteps)
    anneal['T']           = [(10, Tstart) , (-10, Tend)]
    anneal['Ncut_factor'] = [(2/3, 0.), (-10, 1.)]
    anneal['W_noise']     = [(-10, 0.01), (-1, 0.)]

    # Prepare ground-truth GFs (bars)
    W_gt = np.zeros( (H, D2, D2) )
    for i in xrange(D2):
        W_gt[   i, i, :] = 10.
        W_gt[D2+i, :, i] = 10.
    W_gt = W_gt.reshape( (H, D) )
    
    # Prepare model...
    model = MCA_ET(D, H, Hprime, gamma)
    gt_params = {
        'W'     : W_gt,
        'pi'    : 2./H,
        'sigma' : 1.00
    }
    
    # Generate trainig data
    my_N = N // comm.size
    my_data = model.generate_data(gt_params, my_N)
    dlog.append('y',my_data['y'][0:25,:])
    
    # Initialize model parameters (to be learned)
    params = {
    #    'W'     : W_gt,
        'W'     : np.abs(5 + np.random.normal(size=W_gt.shape)),
        'pi'    : 2/H, 
        'sigma' : 5.00
    }
    #params = model.noisify_params(params, anneal)
    params = comm.bcast(params)

   # Create and start EM annealing
    em = EM(model=model, anneal=anneal)
    em.data = my_data
    em.lparams = params
    em.run()
    
    dlog.close()

    #print(em.lparams['W'])
