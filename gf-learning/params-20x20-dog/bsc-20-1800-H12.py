# -*- coding: utf-8
#
# LinCA run for the MCA/BSC Journal Paper
#

from __future__ import division

# Training-data
data_factor = 1.
datafile = "../data/patches-20-dog.h5"
N = 200000

# Model to use
from pulp.em.camodels.linca_et import LinCA_ET
model_class=LinCA_ET

# Number of hidden causes
H = 1800

# Which parameters should be learned:
#to_learn = ['W']
to_learn = ['W', 'pi', 'sigma']

# ET approximation parameters
Hprime = 12
gamma = 10

#In this case, we need an estimation for the parameters
W_init = 'estimate'
pi_init = 'estimate'
sigma_init = 'estimate'


# np.random.normal(scale=W_noise_intensity, size=(H, D)) is added after each run
W_noise     = 0.0
pi_noise    = 0.
sigma_noise = 0.


# Annealing:
temp_start = 1.0
temp_end = 1.0

anneal_steps = 100
anneal_start = 20
anneal_end   = 80

cut_start = 1./3
cut_end = 2./3

noise_decrease = 80
noise_end = 90

anneal_prior = False


# Images used:
channel_splitted = False #This means single channel (this convention'll be reversed)

# Post-Processing:
s_p = 3.
s_m = 1.


processing = 'deconvolve'
fit = True

