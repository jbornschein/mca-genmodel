#!/usr/bin/env python
#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: Academic Free License (AFL) v3.0
#

"""

Given an result-directory this script produces two plots:

 a) time-course of the sparseness (pi*H) plottet over  EM-steps
 b) time-course of the noise-parameter (sigma) over the EM-steps


"""

from __future__ import division

import sys
from math import pi

sys.path.insert(0, "../../../pylib")

from optparse import OptionParser
import re
import tables
import pylab
import Image
import numpy as np


sys.path.insert(0, "../lib")
from viz import *
from gabor_fitting import gabor_kern
from dog_fitting import dog_kern, find_globulars

NoSuchNodeError = tables.exceptions.NoSuchNodeError

zoom = 3

if __name__ == "__main__":
    parser = OptionParser(usage="Usage: %prog [options] <directory>")
    parser.add_option("-p", "--pdf", dest="pdf", action="store_true",
                  help="Create some PDF files")

    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.error("icorrect number of argumenst")

    outdir = sys.argv[1]
    
    # Read parameter file
    params = {}
    execfile(outdir+"/paramfile.py", params)

    # Read data from result.h5
    h5 = tables.openFile(outdir+"/result.h5", "r")

    #=============================================================================
    # General information
    steps, H, D = h5.root.W.shape

    print "Loaded data from %s" % outdir
    print "  EM steps: %d" % steps
    print "  D=%d; H=%d" % (D, H)
    print

    #=============================================================================
    # Sparseness plot
    try:
        pies = h5.root.pi[:]

        print "Saving sparseness plot into %s/sparseness.png..." % outdir
        pylab.plot(H*pies)
        pylab.xlabel("EM steps")
        pylab.ylabel("H*pi")
        pylab.title("Sparseness [H=%d]" % H)
        pylab.savefig(outdir+"/sparseness.png")
    except NoSuchNodeError:
        print "Skipped sparseness plot"


    #=============================================================================
    # Sigma plot
    try:
        sigma = h5.root.sigma[:]

        print "Saving sigma plot into %s/sigma.png..." % outdir
        pylab.cla()
        pylab.plot(sigma)
        pylab.xlabel("EM steps")
        pylab.ylabel("sigma")
        pylab.title("Noise level")
        pylab.savefig(outdir+"/sigma.png")
    except NoSuchNodeError:
        print "Skipped sigma plot"

    
    #=============================================================================
    # N_use plot
    try:
        N_use = h5.root.N_use[:]

        print "Saving N_use plot into %s/n_use.png..." % outdir
        pylab.cla()
        pylab.plot(N_use)
        pylab.xlabel("EM steps")
        pylab.ylabel("N_use")
        pylab.title("Number of datapoints used (ET cuttting)")
        pylab.savefig(outdir+"/n_use.png")
    except NoSuchNodeError:
        print "Skipped N_use plot"

    #=============================================================================
    # nx/ny plot
    try:
        RF = h5.root.RF[0]

        gabor_params = h5.root.gabor_params[:]
        gabor_errors = h5.root.gabor_errors[:]

        dog_params = h5.root.dog_params[:]
        dog_errors = h5.root.dog_errors[:]

        # Proper units for nx/ny plot
        nx = np.abs(gabor_params[:, 6] * gabor_params[:, 2] / 2 / pi)   # XXX
        ny = np.abs(gabor_params[:, 6] * gabor_params[:, 3] / 2 / pi)
        error = gabor_errors[:]

        mask = (nx < 2) * (ny < 2)

        # filter DoG vs. Gabors
        dogs = find_globulars(RF, gabor_params, gabor_errors, dog_params, dog_errors)
        gabors = (dogs == False)

        # load Ringach nx/ny 
        #ringach_h5 = tables.openFile("../ringach-data/old-ringach-nxny.h5", "r")
        ringach_h5 = tables.openFile("../ringach-data/self-fitted-ringach-nxny.h5", "r")
        rnx = ringach_h5.root.R_data[0,:,0]
        rny = ringach_h5.root.R_data[0,:,1]
        rmask = (rnx < 1.5) * (rny < 1.5)
        rnx = rnx[rmask]
        rny = rny[rmask]
        ringach_h5.close()

        # Choose colormap based on filename (green vs. blue)
        cmap = pylab.cm.Blues_r
        if re.findall("linca", outdir):
            cmap = pylab.cm.Greens_r;

        print "Saving nxny plot into %s/nxny-all.png..." % outdir
        print "  Error min/max: %f / %f" % (error.min(), error.max())
        print "  nx-mean:  %4.2f" % nx[gabors*mask].mean()
        print "  ny-mean:  %4.2f" % ny[gabors*mask].mean()
        print "  rnx-mean: %4.2f" % rnx.mean()
        print "  rny-mean: %4.2f" % rny.mean()

        pylab.figure()
        pylab.cla()
        pylab.scatter(nx[mask], ny[mask], c=error[mask], marker='o', cmap=cmap, linewidths=0.0, alpha=0.7)
        pylab.scatter(rnx, rny, c="r", marker='^')
        diag = np.linspace(0, 1, 100)
        pylab.plot(diag, diag, color='r')
        pylab.axis([0.0, 1.4, 0.0, 1.4])
        pylab.title("nx/ny plot")
        pylab.xlabel("nx")
        pylab.xlabel("ny")
        pylab.savefig(outdir+"/nxny-all.png")
        if options.pdf:
            pylab.savefig(outdir+"/nxny-all.pdf")


        print "Saving nxny plot into %s/nxny.png..." % outdir
        pylab.figure()
        pylab.cla()
        pylab.scatter(nx[gabors*mask], ny[gabors*mask], c=error[gabors*mask], marker='o', cmap=cmap, linewidths=0.0, alpha=0.7)
        pylab.scatter(rnx, rny, c="r", marker='^')
        diag = np.linspace(0, 1, 100)
        pylab.plot(diag, diag, color='r')
        pylab.axis([0.0, 1.4, 0.0, 1.4])
        pylab.title("nx/ny plot")
        pylab.xlabel("nx")
        pylab.xlabel("ny")
        pylab.savefig(outdir+"/nxny.png")
        if options.pdf:
            pylab.savefig(outdir+"/nxny.pdf")
    except NoSuchNodeError:
        print "Skipped nx/ny plot!"
        gabor_params = None
        gabor_errors = None

    #=============================================================================
    # Fitting plot
    try:
        RF = h5.root.RF[0]

        dog_params = h5.root.dog_params[:]
        dog_errors = h5.root.dog_errors[:]

        gabor_params = h5.root.gabor_params[:]
        gabor_errors = h5.root.gabor_errors[:]

        print "Saving fitting into %s/fitting-[dog|gabor].png..." % outdir

        def save_recons(idx, fname):
            """ Take a list of h's... save the RFs and threir reconstructions to *fname* """
            # Construct a H*3 shaped grid of fields to be displayed
            n_fields = len(idx)
            shape = RF[0].shape
        
            if n_fields == 0:
                return

            grid = np.zeros( (n_fields, 3, shape[0], shape[1]) )
            for i in xrange(n_fields):
                h = idx[i]

                # column 0: RF from h5 file
                grid[i,0] = RF[h]

                # column 1: Gabor reconstruction
                gabor_recons = gabor_kern(gabor_params[h], shape)
                grid[i,1] = gabor_recons

                # column 2: DoG reconstruction
                dog_recons = dog_kern(dog_params[h], shape)
                grid[i,2] = dog_recons
            grid = grid.reshape( (n_fields*3, shape[0], shape[1]) )

            img = tiled_gfs(grid, sym_cm=True, global_cm=False, rows=n_fields)
            img = img.resize( (zoom*img.size[0], zoom*img.size[1]) )
            img.save(fname)

        dogs = find_globulars(RF, gabor_params, gabor_errors, dog_params, dog_errors)

        save_recons(np.where(dogs)[0], outdir+"/fitting-dog.png")
        save_recons(np.where(dogs == False)[0], outdir+"/fitting-gabor.png")

        # Save errors
        f_gabor_error = open(outdir+"/fitting-gabor-error.txt", "w+")
        f_dog_error = open(outdir+"/fitting-dog-error.txt", "w+")
        for h in xrange(H):
            if dogs[h]:
                f_dog_error.write("%6.3f\n" % dog_errors[h])
            else:
                f_gabor_error.write("%6.3f\n" % gabor_errors[h])
        f_gabor_error.close()
        f_dog_error.close()
    except NoSuchNodeError:
        print "Skipped fitting-plot!"
        gabor_params = None
        gabor_errors = None

    #=============================================================================
    # RF plot
    try:
        RF = h5.root.RF[0]

        print "Saving %d RFs into %s/RFs.png..." % (H, outdir)

        # Reorder according to gabor matching?
        if gabor_params is not None:
            if gabor_params.shape[0] == H:
                sigma_x = gabor_params[:, 2]
                sigma_y = gabor_params[:, 3]
                idx = np.argsort(sigma_y**2 + sigma_y**2)
                RF = RF[idx, :]
            else:
                print "Number of gabor_params != H (%d != %d); NOT SORTING" % (gabor_params.shape[0], H)

        # Fuse channel splitting??  (XXX subtracting  is technically wrong! XXX)

        img = tiled_gfs(RF, sym_cm=True, global_cm=False)
        img = img.resize( (zoom*img.size[0], zoom*img.size[1]) )
        img.save(outdir+"/RF.png")
    except NoSuchNodeError:
        print "Skipped RF plot!"
        gabor_params = None
        gabor_errors = None

    #=============================================================================
    # W plots
    W = h5.root.W[:]

    print "Saving %d GFs into %s/W_*.png..." % (H, outdir)

    # Reorder according to gabor matching?
    if gabor_params is not None:
        if gabor_params.shape[0] == H:
            sigma_x = gabor_params[:, 2]
            sigma_y = gabor_params[:, 3]
            idx = np.argsort(sigma_y**2 + sigma_y**2)
            W = W[:, idx, :]
        else:
            print "Number of gabor_params != H (%d != %d); NOT SORTING" % (gabor_params.shape[0], H)

    # Fuse channel splitting??  (XXX subtracting  is technically wrong! XXX)
    if params['channel_splitted']:
        D2 = np.sqrt(D // 2)
        W = W.reshape(steps, H, 2*D2, D2)
        W = W[:, :, 0:D2, :] - W[:, :, D2:, :]
    else:
        D2 = np.sqrt(D)
        W = W.reshape(steps, H, D2, D2)

    for s in xrange(steps):
        gfs = W[s]
        img = tiled_gfs(gfs, sym_cm=True, global_cm=False)
        img = img.resize( (zoom*img.size[0], zoom*img.size[1]) )
        img.save(outdir+"/W_%03d.png" % s)

    if options.pdf:
        # Save last step as PDF
        pylab.cla()
        H2 = int(np.sqrt(H))+1
        for h in xrange(H):
            vmax=max(W[-1,h].max(), -W[-1,h].min())
            pylab.subplot(H2, H2, h+1)
            pylab.axis("off");
            pylab.imshow(W[-1,h], vmin=-vmax, vmax=vmax)
        pylab.savefig("%s/W_%03d.pdf" % (outdir, steps-1), dpi=600)


    #=============================================================================
    # Cleanp and go home
    h5.close()

