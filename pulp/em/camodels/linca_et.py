# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from math import pi
from scipy import comb
from mpi4py import MPI

import pulp.em as em
import pulp.utils.parallel as parallel
import pulp.utils.tracing as tracing

from pulp.utils.datalog import dlog
from pulp.em.camodels import CAModel

import linca_et_cython
reload(linca_et_cython)

class LinCA_ET(CAModel):
    def __init__(self, D, H, Hprime, gamma, to_learn=['W', 'pi', 'sigma'], comm=MPI.COMM_WORLD):
        CAModel.__init__(self, D, H, Hprime, gamma, to_learn, comm)

    @tracing.traced
    def generate_data(self, model_params, my_N):
        D = self.D
        H = self.H
        pies = model_params['pi']
        W  = model_params['W']
        sigma = model_params['sigma']

        # Create output arrays, y is data, s is ground-truth
        y = np.zeros( (my_N, D) )
        s = np.zeros( (my_N, H), dtype=np.bool )

        for n in xrange(my_N):
            p = np.random.random(H)        # create latent vector
            s[n] = p < pies                # translate into boolean latent vector
            for h in xrange(H):            # Combine according to sum-rule
                if s[n,h]:
                    y[n] += W[h]

        # Add noise according to the model parameters
        y += np.random.normal( scale=sigma, size=(my_N, D) )

        # Build return structure
        return { 'y': y, 's': s }
        
    @tracing.traced
    def select_Hprimes(self, model_params, data):
        """
        Return a new data-dictionary which has been annotated with
        a data['candidates'] dataset. A set of self.Hprime candidates
        will be selected.
        """
        #W  = model_params['W']
        #my_y = data['y']
        #my_N, D   = data['y'].shape
        #H, Hprime = self.H, self.Hprime
        
        #candidates = np.zeros( (my_N, Hprime), dtype=np.int )
        #select = np.zeros( (my_N, H) )
        
        #for n in xrange(my_N):
            #sim = np.inner(W,my_y[n])/ np.sqrt(np.diag(np.inner(W,W)))/ np.sqrt(np.inner(my_y[n],my_y[n]))
            #select[n] = sim
            #candidates[n] = np.argsort(sim)[-Hprime:]
        #data['candidates'] = candidates
        #dlog.append('select', select)
        
        data['candidates'] = linca_et_cython.select_Hprimes(self.H, self.Hprime, model_params['W'], data['y'], data['y'].shape[0])
        return data

    @tracing.traced
    def E_step(self, anneal, model_params, my_data):
        """ LinCA E_step

        my_data variables used:
            
            my_data['y']           Datapoints
            my_data['can']         Candidate H's according to selection func.

        Annealing variables used:

            anneal['T']            Temperature for det. annealing
            anneal['N_cut_factor'] 0.: no truncation; 1. trunc. according to model

        """
        comm      = self.comm
        my_y      = my_data['y'].copy()
        my_cand   = my_data['candidates']
        my_N, D   = my_data['y'].shape
        H = self.H

        SM = self.state_matrix        # shape: (no_states, Hprime)
        state_abs = self.state_abs           # shape: (no_states,)

        W         = model_params['W']
        pies      = model_params['pi']
        sigma     = model_params['sigma']

        # Precompute 
        beta     = 1./anneal['T']
        pre1     = -1./2./sigma/sigma
        pil_bar  = np.log( pies/(1.-pies) )

        # Allocate return structures
        F = np.empty( [my_N, 1+H+self.no_states] )
        pre_F = np.empty( [my_N, 1+H+ self.no_states] )
        denoms = np.zeros(my_N)
        
        # Joerg's data noise idea
        data_noise_scale = anneal['data_noise']
        if data_noise_scale > 0.:
            my_data['data_noise'] = np.random.normal(scale=data_noise_scale, size=my_y.shape)
            my_y += my_data['data_noise']
        
        #posterior = np.zeros_like(F)

        # Iterate over all datapoints
        for n in xrange(my_N):
            tracing.tracepoint("E_step:iterating")
            y    = my_data['y'][n,:]
            cand = my_data['candidates'][n,:]

            # Zero active hidden causes
            log_prod_joint = pre1 * (y**2).sum()
            F[n,0] = log_prod_joint
            pre_F[n,0] = 0.

            # Hidden states with one active cause
            log_prod_joint = pre1 * ((W-y)**2).sum(axis=1)
            F[n,1:H+1] = log_prod_joint
            pre_F[n,1:H+1] = pil_bar

            # Handle hidden states with more than 1 active cause
            pre_F[n,1+H:] = pil_bar * state_abs   # is (no_states,)
            W_ = W[cand]                          # is (Hprime x D)

            Wbar = np.dot(SM,W_)
            log_prod_joint = pre1 * ((Wbar-y)**2).sum(axis=1)
            F[n,1+H:] = log_prod_joint

            corr = (pre_F[n,:]+F[n,:]).max()
            denom = (np.exp(pre_F[n,:] + F[n,:] - corr).sum())
            denoms[n] = corr + np.log(denom)
            
            #posterior[n,:] = np.exp(pre_F[n,:] + F[n,:])/ np.exp(pre_F[n,:] + F[n,:]).sum()

        if anneal['anneal_prior']:
            F = beta * (pre_F + F)
        else:
            F = pre_F + beta * F
            
        #dlog.append('posterior', posterior)

        return { 'logpj': F, 'denoms': denoms}

    @tracing.traced
    def M_step(self, anneal, model_params, my_suff_stat, my_data):
        """ LinCA M_step

        my_data variables used:
            
            my_data['y']           Datapoints
            my_data['candidates']         Candidate H's according to selection func.

        Annealing variables used:

            anneal['T']            Temperature for det. annealing
            anneal['N_cut_factor'] 0.: no truncation; 1. trunc. according to model

        """

        comm      = self.comm
        H, Hprime = self.H, self.Hprime
        gamma     = self.gamma
        W         = model_params['W']
        pies      = model_params['pi']
        sigma     = model_params['sigma']
        
        # Read in data:
        my_y       = my_data['y'].copy()
        candidates = my_data['candidates']
        logpj_all  = my_suff_stat['logpj']
        my_N, D    = my_y.shape
        N          = comm.allreduce(my_N)
        
        # Joerg's data noise idea
        data_noise_scale = anneal['data_noise']
        if data_noise_scale > 0:
            my_y += my_data['data_noise']

        SM = self.state_matrix        # shape: (no_states, Hprime)

        # To compute et_loglike:
        my_ldenom_sum = 0.0
        ldenom_sum = 0.0

        # Precompute factor for pi update
        A_pi_gamma = 0
        B_pi_gamma = 0
        for gamma_p in range(gamma+1):
            A_pi_gamma += comb(H,gamma_p) * (pies**gamma_p) * ((1-pies)**(H-gamma_p))
            B_pi_gamma += gamma_p * comb(H,gamma_p) * (pies**gamma_p) * ((1-pies)**(H-gamma_p))
        E_pi_gamma = pies * H * A_pi_gamma / B_pi_gamma
        
        # Truncate data
        if anneal['Ncut_factor'] > 0.0:
            tracing.tracepoint("M_step:truncating")
            #alpha = 0.9 # alpha from ET paper
            #N_use = int(alpha * (N * (1 - (1 - A_pi_gamma) * anneal['Ncut_factor'])))
            N_use = int(N * (1 - (1 - A_pi_gamma) * anneal['Ncut_factor']))
            cut_denom = parallel.allsort(my_suff_stat['denoms'])[-N_use]
            which   = np.array(my_suff_stat['denoms'] >= cut_denom)

            candidates = candidates[which]
            logpj_all = logpj_all[which]
            my_y    = my_y[which]
            my_N, D = my_y.shape
            N_use = comm.allreduce(my_N)
        else:
            N_use = N
        dlog.append('N_use', N_use)

        # Precompute
        pil_bar   = np.log( pies/(1.-pies) )
        corr_all  = logpj_all.max(axis=1)                 # shape: (my_N,)
        pjb_all   = np.exp(logpj_all - corr_all[:, None]) # shape: (my_N, no_states)

        # Allocate 
        my_Wp = np.zeros_like(W)   # shape (H, D)
        my_Wq = np.zeros((H,H))    # shape (H, H)
        my_pi = 0.0                #
        my_sigma = 0.0             #

        # Iterate over all datapoints
        for n in xrange(my_N):
            tracing.tracepoint("M_step:iterating")
            y     = my_y[n,:]                  # length D
            cand  = candidates[n,:] # length Hprime
            logpj = logpj_all[n,:] # length no_states
            corr  = corr_all[n]        # scalar
            pjb = pjb_all[n, :]

            this_Wp = np.zeros_like(my_Wp)    # numerator for current datapoint   (H, D)
            this_Wq = np.zeros_like(my_Wq)    # denominator for current datapoint (H, H)
            this_pi = 0.0                     # numerator for pi update (current datapoint)

            # Zero active hidden cause (do nothing for the W and pi case) 
            # this_Wp += 0.     # nothing to do
            # this_Wq += 0.     # nothing to do
            # this_pi += 0.     # nothing to do

            # One active hidden cause
            this_Wp = np.outer(pjb[1:(H+1)],y)
            this_Wq = pjb[1:(H+1)] * np.identity(H)
            this_pi = pjb[1:(H+1)].sum()
            
            # Handle hidden states with more than 1 active cause
            this_Wp[cand]      += np.dot(np.outer(y,pjb[(1+H):]),SM).T
            this_Wq_tmp         = np.zeros_like(my_Wq[cand])
            this_Wq_tmp[:,cand] = np.dot(pjb[(1+H):] * SM.T,SM)
            this_Wq[cand]      += this_Wq_tmp
            this_pi            += np.inner(pjb[(1+H):], SM.sum(axis=1))

            denom = pjb.sum()
            my_Wp += this_Wp / denom
            my_Wq += this_Wq / denom
            my_pi += this_pi / denom
            
        # Calculate updated W
        if 'W' in self.to_learn:
            tracing.tracepoint("M_step:update W")
            Wp = np.empty_like(my_Wp)
            Wq = np.empty_like(my_Wq)
            comm.Allreduce( [my_Wp, MPI.DOUBLE], [Wp, MPI.DOUBLE] )
            comm.Allreduce( [my_Wq, MPI.DOUBLE], [Wq, MPI.DOUBLE] )
            W_new  = np.dot(np.linalg.inv(Wq), Wp)
            # W_new  = np.linalg.lstsq(Wq, Wp)
        else:
            W_new = W

        # Calculate updated pi
        if 'pi' in self.to_learn:
            tracing.tracepoint("M_step:update pi")
            pi_new = E_pi_gamma * comm.allreduce(my_pi) / H / N_use
        else:
            pi_new = pies

        # Calculate updated sigma
        if 'sigma' in self.to_learn:
            tracing.tracepoint("M_step:update sigma")
            # Loop for sigma update:
            for n in xrange(my_N):
                tracing.tracepoint("M_step:update sigma iteration")
                y     = my_y[n,:]           # length D
                cand  = candidates[n,:]     # length Hprime
                logpj = logpj_all[n,:]      # length no_states
                corr  = logpj.max()         # scalar
                pjb   = np.exp(logpj - corr)

                # Zero active hidden causes
                this_sigma = pjb[0] * (y**2).sum()

                # Hidden states with one active cause
                this_sigma += (pjb[1:(H+1)] * ((W-y)**2).sum(axis=1)).sum()

                # Handle hidden states with more than 1 active cause
                SM = self.state_matrix                 # is (no_states, Hprime)
                W_ = W[cand]                           # is (Hprime x D)
                Wbar = np.dot(SM,W_)
                this_sigma += (pjb[(H+1):] * ((Wbar-y)**2).sum(axis=1)).sum()

                denom = pjb.sum()
                my_sigma += this_sigma/ denom

            sigma_new = np.sqrt(comm.allreduce(my_sigma) / D / N_use)
        else:
            sigma_new = sigma
        
        for param in anneal.crit_params:
            exec('this_param = ' + param)
            anneal.dyn_param(param, this_param)
        
        dlog.append('N_use', N_use)

        return { 'W': W_new, 'pi': pi_new, 'sigma': sigma_new, 'Q': 0.}

    def calculate_respons(self, anneal, model_params, data):
        data['candidates'].sort(axis=1) #(we do this to set the order back=outside)
        F_JB = self.E_step(anneal, model_params, data)['logpj']
        #Transform into responsabilities
        corr = np.max(F_JB, axis=1)       
        exp_F_JB_corr = np.exp(F_JB - corr[:, None])
        respons = exp_F_JB_corr/(np.sum(exp_F_JB_corr, axis=1).reshape(-1, 1))
        return respons

    def free_energy(self, model_params, my_data):
        return 0.0

    def gain(self, old_parameters, new_parameters):
        return 0.
