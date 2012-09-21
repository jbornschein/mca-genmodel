#!/usr/bin/env python
#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: GPL V3.0
#

from __future__ import division

import sys
sys.path.insert(0, "lib/")

import numpy as np
import tables
from optparse import OptionParser

from pulp.utils.autotable import AutoTable
from pulp.utils.datalog import dlog
import pulp.utils.parallel as parallel

#from viz import *


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
    in_oversized = in_h5.root.patches

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
    out_fname = "patches-%d-zca3" % size
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

    # Internal parameters
    batch_size = 1000
    epsilon = 1e-3
    D = size**2
    dim = D
    
    #============================================================
    # Start to do some real work
    dlog.progress("Loading patches...")

    P = in_patches[:N_patches,:,:].reshape(N_patches, -1)
    P_mean = P.mean(axis=0)

    # Create covariance matrix
    cov_mat = np.zeros((D, D))
    for n in xrange(0, N_patches):
        if n % batch_size == 0:
            dlog.progress("Creating covariance matrix (%d)" % n, n/N_patches)

        cov_mat += np.outer( P[n]-P_mean, P[n]-P_mean )
    cov_mat /= N_patches


    # Eigenvalue decomposition
    dlog.progress("Computing principal components...")
    U, S, V = np.linalg.svd(cov_mat)
    
    # Construct transformation matrix
    diag = np.diag( 1./np.sqrt(S[:dim]+epsilon) )
    trans_mat = np.dot(U[:,:dim], np.dot(diag, U.T[:dim,:]))

    assert trans_mat.shape == (D, D)

    # Apply transform (whiten)
    dlog.progress("Applying transform...")
    P_ = np.inner(P, trans_mat)
    
    # Normalize and mean-free
    if options.mf:
        dlog.progress("Making mean-free...")
        P_ -= P_.mean(axis=1)[:, None]
    if options.norm:
        dlog.progress("Normalizing...")
        P_max = np.maximum(P_.max(axis=1), -P_.min(axis=1))
        print P_max.shape
        P_ /= P_max[:, None]+1e-5
    if options.varnorm:
        dlog.progress("Normalizing...")
        P_var = np.var(P_, axis=1)
        P_ /= np.sqrt(P_var[:, None])+1e-5
    
    # Make patches to 2d images again...
    P_ = P_.reshape( (N_patches, size, size) )

    # And safe resulting patches
    for n in xrange(0, N_patches):
        if n % batch_size == 0:
            dlog.progress("Saving whitened patches (%d)" % n, n/N_patches)
        tbl_out.append("patches", P_[n])

    # Safe principal components & eigenvalues
    for d in xrange(D):
        tbl_out.append("eig_vec", U[:,d])
        tbl_out.append("eig_val", S[d])


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

    #============================================================
    # Saving...
    print "Saving patches..."
    P2 = np.zeros( (2*size,size) )
    for n in xrange(num_patches): 
        if n % 1000 == 0:
            dlog.progress("Writing patches", n/num_patches)

        # Channel Splitting, saving and renormalizing (to 10) 
        P2[:size, :] =    P[n, :, :]
        P2[size:, :] = -1*P[n, :, :]
        P2[P2<0] = 0.
        P2 = 10.*(P2/np.max(P2))
        tbl.append('orig', P2)

        # Channel Splitting, saving and renormalizing (to 10) 
        P2[:size, :] =    P_[n, :, :]
        P2[size:, :] = -1*P_[n, :, :]
        P2[P2<0] = 0.
        P2 = 10.*(P2/np.max(P2))
        tbl.append('patches', P2)

    tbl.close()












