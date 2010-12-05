# -*- coding: utf-8 -*-

from __future__ import division

from mpi4py import MPI

from scipy import comb
import numpy as np

import pulp.utils.parallel as parallel


def select_Hprimes(H, Hprime, W, my_y, my_N):
    cdef int n
    
    candidates = np.zeros( (my_N, Hprime), dtype=np.int )
    
    for n in xrange(my_N):
        sim = np.inner(W,my_y[n])/ np.sqrt(np.diag(np.inner(W,W)))/ np.sqrt(np.inner(my_y[n],my_y[n]))
        candidates[n] = np.argsort(sim)[-Hprime:]
        
    return candidates

def E_step(T, comm, my_N, D, H, Hprime, W, pies, sigma, my_data, state_list, state_matrix, my_suff_stat):
    cdef int n
    cdef float Q, pre1
    
    # Precompute 
    pre1     = -1./2./sigma/sigma
    pil_bar  = np.log( pies/(1.-pies) )

    # Allocate return structures
    no_states = 1+H+len(state_list)
    F = np.empty( [my_N, no_states] )
    Q = 0.0

    # Iterate over all datapoints
    for n in xrange(my_N):
        y    = my_data['y'][n,:]
        cand = my_data['candidates'][n,:]

        # Zero active hidden causes
        log_prod_joint = pre1 * (y**2).sum()
        F[n,0] = log_prod_joint

        # Hidden states with one active cause
        log_prod_joint = pil_bar + pre1 * ((W-y)**2).sum(axis=1)
        F[n,1:H+1] = log_prod_joint

        # Handle hidden states with more than 1 active cause
        SM = state_matrix                      # is (no_states, Hprime)
        log_prior = pil_bar * SM.sum(axis=1)   # is (no_states,)
        W_ = W[cand]                           # is (Hprime x D)

        Wbar = np.dot(SM,W_)
        log_prod_joint = log_prior + pre1 * ((Wbar-y)**2).sum(axis=1)
        F[n,1+H:] = log_prod_joint
        
        if my_suff_stat == {}:
            Q = -np.inf
        else:
            Q += np.inner(F[n,:], np.exp(my_suff_stat['logpj'][n,:]))/ my_suff_stat['logpj'][n,:].sum()
            
        denoms = np.exp(F).sum(axis=1)
            
    return { 'logpj': F, 'denoms': denoms, 'Q_sum2': Q }

def M_step(T, comm, position, H, Hprime, gamma, W, pies, sigma, Q, my_data, state_matrix, my_suff_stat, to_learn):
    cdef int n, my_N, D
    cdef float A_pi_gamma, B_pi_gamma, E_pi_gamma, beta, my_pi, my_sigma, this_pi, this_sigma, corr, pi_new, sigma_new
    
    # Prefactor for pi update and N_gamma
    A_pi_gamma = 0
    B_pi_gamma = 0
    for gamma_p in range(gamma+1):
        A_pi_gamma += comb(H,gamma_p) * (pies**gamma_p) * ((1-pies)**(H-gamma_p))
        B_pi_gamma += gamma_p * comb(H,gamma_p) * (pies**gamma_p) * ((1-pies)**(H-gamma_p))
    E_pi_gamma = pies * H * A_pi_gamma / B_pi_gamma
    
    # Read in data:
    my_y      = my_data['y']
    my_N, D   = my_y.shape

    # Truncate data
    if position > 2./3:
        sort_args = parallel.allargsort(-my_suff_stat['denoms'])
        my_y      = my_y[sort_args,:]
        #N_use   = my_N - (3 * position - 2) * (my_N - A_pi_gamma * my_N)
        N_use   = my_N * (1 - (3 * position -2) * (1 - A_pi_gamma))
        N_use   = np.int(np.floor(N_use/ comm.size) * comm.size)
        my_y    = my_y[0:N_use,:]
        my_N, D = my_y.shape

    percent_done = np.str(np.int(np.round(100 * position)))
    percent_done = (3-percent_done.__len__()) * ' ' + percent_done
    #print 33 * '-' + ' Done: ' + percent_done + '%', 33 * '-'

    # Precompute
    beta     = 1./T
    pil_bar  = np.log( pies/(1.-pies) )
    
    # Allocate 
    my_Wp = np.zeros_like(W)   # H x D
    my_Wq = np.zeros((H,H))    # H x H
    my_pi = 0.0                #
    my_sigma = 0.0

    # Iterate over all datapoints
    for n in xrange(my_N):
        y     = my_data['y'][n,:]          # length D
        cand  = my_data['candidates'][n,:] # length Hprime
        logpj = my_suff_stat['logpj'][n,:] # length no_states
        corr  = (beta*logpj).max()         # scalar

        this_Wp = np.zeros_like(my_Wp)    # numerator for current datapoint   (H, D)
        this_Wq = np.zeros_like(my_Wq)    # denominator for current datapoint (H, H)
        this_pi = 0.0                     # numerator for pi update (current datapoint)

        ###########################################################################
        # Zero active hidden causes
        # this_Wp += 0.     # nothing to do
        # this_Wq += 0.     # nothing to do

        ###########################################################################
        # One active hidden cause
        pjb      = np.exp(beta*logpj - corr)
        this_Wp += np.outer(pjb[1:(H+1)],y)
        this_Wq += pjb[1:(H+1)] * np.identity(H)
        this_pi += pjb[1:(H+1)].sum()

        ###########################################################################
        # Handle hidden states with more than 1 active cause
        SM    = state_matrix # SM    is (no_states, Hprime)
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
    Wp = np.empty_like(my_Wp)
    Wq = np.empty_like(my_Wq)
    comm.Allreduce( [my_Wp, MPI.DOUBLE], [Wp, MPI.DOUBLE] )
    comm.Allreduce( [my_Wq, MPI.DOUBLE], [Wq, MPI.DOUBLE] )
    
    if to_learn.__contains__('W'):
        W_new  = np.dot(np.linalg.inv(Wq),Wp)
    else:
        W_new = W
    
    # Calculate updated pi
    if to_learn.__contains__('pi'):
        pi_new = E_pi_gamma * comm.allreduce(my_pi) / H / comm.allreduce(my_N)
    else:
        pi_new = pies

    # Calculate updated sigma
    if to_learn.__contains__('sigma'):
        # Loop for sigma update:
        for n in xrange(my_N):
            y     = my_data['y'][n,:]          # length D
            cand  = my_data['candidates'][n,:] # length Hprime
            logpj = my_suff_stat['logpj'][n,:] # length no_states
            corr  = (beta*logpj).max()         # scalar
            pjb      = np.exp(beta*logpj - corr)
            
            # Zero active hidden causes
            this_sigma = pjb[0] * (y**2).sum()
            
            # Hidden states with one active cause
            this_sigma += (pjb[1:(H+1)] * ((W-y)**2).sum(axis=1)).sum()
            
            # Handle hidden states with more than 1 active cause
            SM = state_matrix                 # is (no_states, Hprime)
            log_prior = pil_bar * SM.sum(axis=1)   # is (no_states,)
            W_ = W[cand]                           # is (Hprime x D)
            Wbar = np.dot(SM,W_)
            this_sigma += (pjb[(H+1):] * ((Wbar-y)**2).sum(axis=1)).sum()
            
            denom = pjb.sum()
            my_sigma += this_sigma/ denom

        sigma_new = np.sqrt(comm.allreduce(my_sigma) / D / comm.allreduce(my_N))
    else:
        sigma_new = sigma
    
    Q += my_N * (H * np.log(1-pies) - D/2 * np.log(2*np.pi*(sigma**2)))
    
    return { 'W': W_new, 'pi': pi_new, 'sigma': sigma_new, 'Q': Q, 'to_learn': to_learn }, my_N, percent_done
