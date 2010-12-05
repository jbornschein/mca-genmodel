from __future__ import division

"""LinCA_ET subclass.

   selection function is overwrittten by MCA selection function
    

"""


from math import pi
from mpi4py import MPI

from scipy import comb
import numpy as np

import pulp.em as em
import pulp.utils.parallel as parallel
import pulp.utils.tracing as tracing
from pulp.utils.datalog import dlog

from pulp.em.linca_et import LinCA_ET


class LinCA_ET_MSEL(LinCA_ET):


    def select_Hprimes(self, model_params, data):
        """
        """
        comm      = self.comm
        my_y      = data['y']
        my_N, _   = my_y.shape
        H, Hprime = self.H, self.Hprime
        W         = model_params['W']

        # Allocate return structure
        candidates = np.zeros( (my_N, Hprime), dtype=np.int )
        
        #TODO: When using different pies this should be changed!
        for n in xrange(my_N):
            W_interm = np.maximum(W, my_y[n])
            sim = np.abs(W_interm-my_y[n]).sum(axis=1)
            candidates[n] = np.argsort(sim)[0:Hprime]

        data['candidates'] = candidates

        return data
