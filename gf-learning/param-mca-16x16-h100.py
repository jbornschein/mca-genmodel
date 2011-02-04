# -*- coding: utf-8
#
# Parameterfile for MCA-ET run
#

from __future__ import division

# Training-data
datafile = "../data/patches16.h5"
N = 50000

# Operate on channel-splitted patches
channel_splitted = True

# Choose model
from pulp.em.camodels.mca_et import MCA_ET
model_class=MCA_ET

# Number of hidden causes
H = 100

# Which parameters should be learned:
#to_learn = ['W']
to_learn = ['W', 'pi', 'sigma']

# ET approximation parameters
Hprime = 10
gamma = 4

# Initial model parameters
W_init = 'estimate'
pi_init = 2/H
sigma_init = 'estimate'

# Annealing:
anneal_steps = 120
anneal_start = 40
anneal_end   = 80
temp_start = 1.0
temp_end = 1.0
cut_start = 40
cut_end = 80
noise_decrease = 40
noise_end = 80

W_noise     = 0.05
pi_noise    = 0.
sigma_noise = 0.

# -- Postprocessing --

# Postprocessing: run convolution to approx. reverse correlation
processing = 'deconvolve'   
s_m = 1.  # convolution parameters
s_p = 3.  # convolution parameters

# Automatically fit resulting RFs with gabors?
fit = False

