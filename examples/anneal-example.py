#!/usr/bin/env python
#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: Academic Free License (AFL) v3.0
#

"""
  Demonstrate how to use the annealing class to generate 
  piecewise-linear annealing schedules.
"""

from __future__ import division

import sys
sys.path.insert(0, '..')

from pulp.em.annealing import LinearAnnealing

Tsteps = 80
Tstart = 20
Tend = 1.05


# Choose annealing schedule
anneal = LinearAnnealing(Tsteps)

anneal['T'] = [(10, Tstart) , (-10, Tend)]
anneal['Ncut_factor'] = [(2/3, 0.) , (-10, 1.)]
anneal['pi_noise'] = 0.5
    
assert anneal['bla'] == 0.0

while not anneal.finished:
    print "[%3d] T=%.2f   Ncut_factor=%.2f     pi_noise=%.2f" % (anneal['step'], anneal['T'], anneal['Ncut_factor'], anneal["pi_noise"])
    anneal.next()

