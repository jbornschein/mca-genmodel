#!/usr/bin/env python
#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: Academic Free License (AFL) v3.0
#

from __future__ import division

import sys
sys.path.insert(0, "lib/")

import numpy as np
import tables
from optparse import OptionParser
from numpy import fft

from pulp.utils.autotable import AutoTable
from pulp.utils.datalog import dlog
import pulp.utils.parallel as parallel

#from viz import *

import debug

def pseudo_kernel(shape):
    """Generate and return the convolution filter in freq. space"""
    h, w = shape
    h2 = h // 2
    w2 = w // 2
    f0 = int(0.8 * min(h2, w2))
    y,x = np.ogrid[-h2:h-h2 , -w2:w-w2 ]
    F   = np.sqrt(x*x+y*y) * np.exp( -(np.sqrt(x*x+y*y)/f0)**4 )
    return fft.fftshift(F)


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
    out_fname = "patches-%d-pseudo" % size
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
    
    # Generate pseudo whitening kernel in freq. space
    mult_kernel = pseudo_kernel( (oversize, oversize) )

    for n in xrange(0, N_patches):
        if n % batch_size == 0:
            dlog.progress("Preprocessing...", n/N_patches)

        P = in_oversized[n,:,:].astype(np.float)
        P_ = fft.fft2(P)
        Q_ = mult_kernel * P_
        Q = np.real(fft.ifft2(Q_))
        Q = Q[left:right,left:right]

        # Normalize and mean-free
        if options.mf:
            Q -= Q.mean()
        if options.norm:
            Q_max = max(Q.max(), -Q.min())
            Q /= (Q_max+1e-5)
        if options.varnorm:
            Q_var = np.var(Q_)
            Q_ /= (np.sqrt(Q_var)+1e-5)
 
    
        # Ensure sanity
        assert np.isfinite(Q).all()

        tbl_out.append("patches", Q)

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
