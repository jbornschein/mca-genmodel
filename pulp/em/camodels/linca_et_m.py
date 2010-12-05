# -*- coding: utf-8 -*-

from __future__ import division

from math import pi
from mpi4py import MPI

from itertools import combinations
from scipy import comb
import numpy as np

import pulp.em as em
import pulp.utils.parallel as parallel
import pulp.utils.tracing as tracing
from pulp.utils.datalog import dlog

import linca_et_cython
reload(linca_et_cython)

class LinCA_ET_M(em.CAModel):
    def __init__(self, D, H, Hprime, gamma, to_learn=['W', 'pi', 'sigma'], comm=MPI.COMM_WORLD):
        em.CAModel.__init__(self, D, H, Hprime, gamma, to_learn, comm)

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
            for h in xrange(H):
                if s[n,h]:
                    y[n] += W[h]

        # Add noise according to the model parameters
        y += np.random.normal( scale=sigma, size=(my_N, D) )

        # Build return structure
        return { 'y': y, 's': s }

    def select_Hprimes(self, model_params, data):
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

    #def E_step(self, T, model_params, my_data):
        #my_N, D   = my_data['y'].shape
        #return linca_et_cython.E_step(T, self.comm, my_N, D, self.H, self.Hprime, model_params['W'], model_params['pi'], model_params['sigma'], my_data, self.state_list, self.state_matrix, self.my_suff_stat)
        
    #def M_step(self, T, position, model_params, my_data):
        #result, my_N, percent_done = linca_et_cython.M_step(T, self.comm, position, self.H, self.Hprime, self.gamma, model_params['W'], model_params['pi'], model_params['sigma'], self.my_suff_stat['Q_sum2'], my_data, self.state_matrix, self.my_suff_stat)
        #dlog.append('N', my_N)
        #dlog.append('Done', percent_done[1:]+'%')
        #return result

#    def select_partial_data(self, my_data):
#        my_N = my_data.shape[0]
#        N_cut = my_N
#
#        idx = np.random.perutation(my_N)[:N_cut]
#    
#        my_pdata = {}
#        for key, val in my_data.items():
#            my_pdata[key] = val[idx]
#            
#        return my_pdata

    #def select_Hprimes(self, model_params, data):
        #H        = self.H
        #Hprime   = self.Hprime
        #W        = model_params['W']
        #my_y     = data['y']
        #my_N, _  = my_y.shape
        
        #candidates = np.zeros( (my_N, Hprime), dtype=np.int )
        
        #for n in xrange(my_N):
            #sim = np.inner(W,my_y[n])/ np.sqrt(np.diag(np.inner(W,W)))/ np.sqrt(np.inner(my_y[n],my_y[n]))
            #candidates[n] = np.argsort(sim)[-Hprime:]

        #data['candidates'] = candidates

        #return data

    def E_step(self, anneal, model_params, my_data):
        tracing.tracepoint("E_step:begin")

        comm      = self.comm
        my_N, D   = my_data['y'].shape
        H, Hprime = self.H, self.Hprime
        N         = comm.allreduce(my_N)

        W         = model_params['W']
        pies      = model_params['pi']
        sigma     = model_params['sigma']

        # Precompute 
        beta     = 1./anneal['T']
        pre1     = -1./2./sigma/sigma
        pil_bar  = np.log( pies/(1.-pies) )

        # Allocate return structures
        no_states = 1+H+len(self.state_list)
        F = np.empty( [my_N, no_states] )
        pre_F = np.empty( [my_N, no_states] )
        Q = 0.0
        Shannon = 0.0
        denoms = np.zeros(my_N)

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
            SM = self.state_matrix                 # is (no_states, Hprime)
            pre_F[n,1+H:] = pil_bar * self.state_abs   # is (no_states,)
            W_ = W[cand]                           # is (Hprime x D)

            Wbar = np.dot(SM,W_)
            log_prod_joint = pre1 * ((Wbar-y)**2).sum(axis=1)
            F[n,1+H:] = log_prod_joint
            corr = (pre_F[n,:]+F[n,:]).max()
            denom = (np.exp(pre_F[n,:] + F[n,:] - corr).sum())
            denoms[n] = corr + np.log(denom)
            
            Q += np.inner(np.exp(pre_F[n,:]+F[n,:]-corr),pre_F[n,:]+F[n,:])/ denom
            Shannon += np.inner(np.exp(F[n,:])/denom, F[n,:]- np.log(denom))

            Q = Q - Shannon
            #print Shannon
        if anneal['anneal_prior']:
            F = beta * (pre_F + F)
        else:
            F = pre_F + beta * F

        tracing.tracepoint("E_step:end")
        return { 'logpj': F, 'denoms': denoms, 'Q_sum2': Q }

    def M_step(self, anneal, model_params, my_suff_stat, my_data):
        tracing.tracepoint("M_step:begin")

        comm      = self.comm
        H, Hprime = self.H, self.Hprime
        gamma     = self.gamma
        W         = model_params['W']
        pies      = model_params['pi']
        sigma     = model_params['sigma']
        Q         = my_suff_stat['Q_sum2']
        
        # Read in data:
        my_y       = my_data['y'].copy()
        candidates = my_data['candidates'].copy()
        logpj_all  = my_suff_stat['logpj'].copy()
        my_N, D    = my_y.shape
        N          = comm.allreduce(my_N)

        # Prefactor for pi update and N_gamma
        A_pi_gamma = 0
        B_pi_gamma = 0
        for gamma_p in range(gamma+1):
            A_pi_gamma += comb(H,gamma_p) * (pies**gamma_p) * ((1-pies)**(H-gamma_p))
            B_pi_gamma += gamma_p * comb(H,gamma_p) * (pies**gamma_p) * ((1-pies)**(H-gamma_p))
        E_pi_gamma = pies * H * A_pi_gamma / B_pi_gamma
        
        # Truncate data
        if anneal['Ncut_factor'] > 0.0:
            tracing.tracepoint("M_step:truncating")
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

        # Precompute
        pil_bar  = np.log( pies/(1.-pies) )

        # Allocate 
        my_Wp = np.zeros_like(W)   # H x D
        my_Wq = np.zeros((H,H))    # H x H
        my_pi = 0.0                #
        my_sigma = 0.0

        # Iterate over all datapoints
        for n in xrange(my_N):
            tracing.tracepoint("M_step:iterating")
            y     = my_y[n,:]                  # length D
            cand  = candidates[n,:] # length Hprime
            logpj = logpj_all[n,:] # length no_states
            corr  = (logpj).max()         # scalar

            #this_Wp = np.zeros_like(my_Wp)    # numerator for current datapoint   (H, D)
            #this_Wq = np.zeros_like(my_Wq)    # denominator for current datapoint (H, H)
            #this_pi = 0.0                      # numerator for pi update (current datapoint)

            ###########################################################################
            # Zero active hidden cause (do nothing for the W and pi case) 
###########################################################################
            # One active hidden cause
            pjb     = np.exp(logpj - corr)
            this_Wp = np.outer(pjb[1:(H+1)],y)
            this_Wq = pjb[1:(H+1)] * np.identity(H)
            this_pi = pjb[1:(H+1)].sum()
            
            ###########################################################################
            # Handle hidden states with more than 1 active cause
            SM    = self.state_matrix # SM    is (no_states, Hprime)
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

            W_new  = np.dot(np.linalg.inv(Wq),Wp)
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
                y     = my_y[n,:]          # length D
                cand  = candidates[n,:] # length Hprime
                logpj = logpj_all[n,:] # length no_states
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

        Q += N_use * (H * np.log(1-pi_new) - D/2 * np.log(2*np.pi*(sigma_new**2)))

        dlog.append('N_use', N_use)

        tracing.tracepoint("M_step:end")
        return { 'W': W_new, 'pi': pi_new, 'sigma': sigma_new, 'Q': Q}

    def free_energy(self, model_params, my_data):
        return 0.0

    def gain(self, old_parameters, new_parameters):
        return 0.
