#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#        and Marc Henniges
#  Lincense: GPL V3.0
#

"""
  Base class for Expectation Truncation based multiple cause models
"""

from __future__ import division

import numpy as np
from mpi4py import MPI

try:
    from itertools import combinations
except ImportError, e:
    from pulp.utils.py25_compatibility import _py25_combinations as combinations

try:
    from abc import ABCMeta, abstractmethod
except ImportError, e:
    from pulp.utils.py25_compatibility import _py25_ABCMeta as ABCMeta
    from pulp.utils.py25_compatibility import _py25_abstractmethod as abstractmethod

import pulp.utils.tracing as tracing
import pulp.utils.parallel as parallel

from pulp.utils.datalog import dlog
from pulp.em import Model

#=============================================================================#
# Abstract base class for component analysis models

class CAModel(Model):
    __metaclass__ = ABCMeta

    def __init__(self, D, H, Hprime, gamma, to_learn=['W', 'pi', 'sigma'], comm=MPI.COMM_WORLD):
        Model.__init__(self, comm)
        self.to_learn = to_learn

        # Model meta-parameters
        self.D = D
        self.H = H
        self.Hprime = Hprime
        self.gamma = gamma

        # Noise Policy
        tol = 1e-5
        self.noise_policy = {
            'W'    : (-np.inf, +np.inf, False ),
            'pi'   : (    tol,  1.-tol, False ),
            'sigma': (     0., +np.inf, False )
        }

        # Generate state-space list
        sl = []
        for g in xrange(2,gamma+1):
            for s in combinations(range(Hprime), g):
                sl.append( np.array(s, dtype=np.int8) )
        self.state_list = sl

        no_states = len(sl)
        self.no_states = no_states
        
        # Generate state-matrix
        sm = np.zeros((no_states, Hprime), dtype=np.uint8)
        for i in xrange(no_states):
            s = sl[i]
            sm[i, s] = 1
        self.state_matrix = sm
        self.state_abs = sm.sum(axis=1)
        
    @tracing.traced
    def check_params(self, model_params):
        """ Sanity check.

        Sanity-check the given model parameters. Raises an exception if 
        something is severely wrong.
        """
        W     = model_params['W']
        pies  = model_params['pi']
        sigma = model_params['sigma']

        assert np.isfinite(W).all()      # check W

        assert np.isfinite(pies).all()   # check pies
        assert pies >= 0.
        assert pies <= 1.

        assert np.isfinite(sigma).all()  # check sigma
        assert sigma >= 0.

        return model_params

    @tracing.traced
    def select_partial_data(self, anneal, my_data):
        """ Select a partial data-set from my_data and return it.

        The fraction of datapoints selected is determined by anneal['partial'].
        If anneal['partial'] is equal to either 1 or 0 the whole dataset will 
        be returned.
        """
        partial = anneal['partial']

        if partial == 0 or partial == 1:              # partial == full data
            return my_data

        my_N, D = my_data['y'].shape            
        my_pN = int(np.ceil(my_N * partial))

        if my_N == my_pN:                            # partial == full data
            return my_data

        # Choose subset...
        sel = np.random.permutation(my_N)[:my_pN]
        sel.sort()

        # Construct partial my_pdata...
        my_pdata = {}
        for key, val in my_data.items():
            my_pdata[key] = val[sel]

        return my_pdata

    @tracing.traced
    def step(self, anneal, model_params, my_data):
        """ Perform an EM-step """

        # Noisify model parameters
        model_params = self.noisify_params(model_params, anneal)

        # Sanity check model parameters
        model_params = self.check_params(model_params)

        # For partial EM-step: select batch
        my_pdata = self.select_partial_data(anneal, my_data)

        # Annotate partial dataset with hidden-state candidates
        my_pdata = self.select_Hprimes(model_params, my_pdata)

        # Do E-step and calculate joint-probabilities
        my_joint_prob = self.E_step(anneal, model_params, my_pdata)

        # Use joint-probabilities to derive new parameter set
        new_model_params = self.M_step(anneal, model_params, my_joint_prob, my_pdata)

        # Log iboth model parameters and annealing parameters
        dlog.append_all(new_model_params)
        dlog.append_all(anneal.as_dict())

        return new_model_params

    @tracing.traced
    def standard_init(self, data):
        """ Standard Initial Estimation for *W* and *sigma*.

        each *W* raw is set to the average over the data plus WGN of mean zero
        and var *sigma*/4. *sigma* is set to the variance of the data around 
        the computed mean. *pi* is set to 1./H . Returns a dict with the 
        estimated parameter set with entries "W", "pi" and "sigma".
        """
        comm = self.comm
        H = self.H
        my_y = data['y']
        my_N, D = my_y.shape

        assert D == self.D

        # Calculate averarge W
        W_mean = parallel.allmean(my_y, axis=0, comm=comm)               # shape: (D, )

        # Calculate data variance
        sigma_sq = parallel.allmean((my_y-W_mean)**2, axis=0, comm=comm) # shape: (D, )
        sigma_init = np.sqrt(sigma_sq).sum() / D                         # scalar

        # Initial W
        noise = sigma_init/4.
        W_init = W_mean + np.random.normal(scale=noise, size=[H, D])    # shape: (H, D)

        #Create and set Model Parameters, W columns have the same average!
        model_params = {
            'W'     : W_init, 
            'pi'    : 1./H,
            'sigma' : sigma_init
        }

        return model_params


