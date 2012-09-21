#!/usr/bin/env python

from __future__ import division

import sys
sys.path.insert(0, "lib/")

import numpy as np
import tables
from optparse import OptionParser
from scipy.signal import convolve2d

from pulp.utils.autotable import AutoTable
from pulp.utils.datalog import dlog
import pulp.utils.parallel as parallel

#from viz import *

def DoG(sigma_pos, sigma_neg, size):
    """ Difference of gaussians kernel of (size, size)-shape.
        The kernel is constructed to be mean free and to have a peak 
        amplitude of 1.
    """
    s2 = size // 2

    gy, gx = np.ogrid[ -s2:size-s2 , -s2:size-s2 ]

    G1 = np.exp( -(gx*gx+gy*gy) / (2.*sigma_pos**2) ) / (2*np.pi*sigma_pos**2)
    G2 = np.exp( -(gx*gx+gy*gy) / (2.*sigma_neg**2) ) / (2*np.pi*sigma_neg**2)
 
    G2 = G2 / G2.sum()*G1.sum()     # make DC free
    G = G1-G2                       # combine positive and negative Gaussians
    G = G / G.max()                 # mormalize peak to 1.
    return G



#=============================================================================
if __name__ == "__main__":
    parser = OptionParser(usage="Usage: %prog [options] <patches.h5>")
    parser.add_option("--mf", dest="mf", action="store_true", 
                        help="make each patch individually mean-free")
    parser.add_option("--norm", dest="norm", action="store_true",
                        help="normalize each patch to [-1 .. 1]")
    parser.add_option("--varnorm", dest="varnorm", action="store_true",
                        help="normalize each patch to variance 1")
    parser.add_option("-n", "--num-patches", type="int", dest="num_patches", default=None,
                        help="number of patches to generate")

    options, args = parser.parse_args()
    
    if len(args) != 1:
        parser.print_help()
        exit(1)

    # Open input file
    in_fname = args[0]
    in_h5 = tables.openFile(in_fname, "r")

    in_patches = in_h5.root.patches
    in_oversized = in_h5.root.oversized

    # Some asserts in the input data
    assert in_patches.shape[0] == in_oversized.shape[0]     # number of patches
    assert in_patches.shape[1] == in_patches.shape[2]       # sqare patches
    assert in_oversized.shape[1] == in_oversized.shape[2]   # square oversized

    # Number of patches to extract
    N_patches = in_patches.shape[0]
    if options.num_patches is not None:
        N_patches = min(N_patches, options.num_patches)

    # Size of the patches
    size = in_patches.shape[1]
    oversize = in_oversized.shape[1]
    
    # Output file name
    out_fname = "patches-%d-dog" % size
    if options.mf:
        out_fname += "-mf"
    if options.norm:
        out_fname += "-norm"
    if options.varnorm:
        out_fname += "-varnorm"
    
    #
    print "Input file:   %s" % in_fname
    print "Output file:  %s" % out_fname
    print "# of patches: %d" % N_patches
    print "Patch size :  %d x %d" % (size, size)

    # Create output file
    tbl_out = AutoTable(out_fname+".h5")

    # Size magic
    left = (oversize // 2)-(size //2)
    right = left + size
    
    #============================================================
    # Start to do some real work
    batch_size = 1000
    
    dog = DoG(1., 3., 9)
    for n in xrange(0, N_patches):
        if n % batch_size == 0:
            dlog.progress("Preprocessing...", n/N_patches)

        P = in_oversized[n,:,:]
        P_ = convolve2d(P, dog, 'same')
        P_ = P_[left:right, left:right]

        # Normalize and mean-free
        if options.mf:
            P_ -= P_.mean()
        if options.norm:
            P_max = max(P_.max(), -P_.min())
            P_ /= (P_max+1e-5)
        if options.varnorm:
            P_var = np.var(P_)
            P_ /= (np.sqrt(P_var)+1e-5)
    
        tbl_out.append("patches", P_)

    in_h5.close()
    tbl_out.close()
    exit(0)

    #============================================================
    # Safe debug-output
    zoom = 6

    grid = U.transpose().reshape( (D, size,  size) ) 
    img = tiled_gfs(grid, sym_cm=False, global_cm=True)
    img = img.resize( (zoom*img.size[0], zoom*img.size[1]) )
    img.save(out_fname+"-components.png")

    grid = P[:100,:,:]
    img = tiled_gfs(grid, sym_cm=False, global_cm=False)
    img = img.resize( (zoom*img.size[0], zoom*img.size[1]) )
    img.save(out_fname+"-orig.png")

    grid = P_[:100,:,:]
    img = tiled_gfs(grid, sym_cm=True, global_cm=False)
    img = img.resize( (zoom*img.size[0], zoom*img.size[1]) )
    img.save(out_fname+"-patches.png")

