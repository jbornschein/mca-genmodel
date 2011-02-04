#!/usr/bin/env python

"""

Given an result-directory this script produces multiple plots:

 a) time-course of the sparseness (pi*H) plottet over  EM-steps
 b) time-course of the noise-parameter (sigma) over the EM-steps
 c) ..


"""

from __future__ import division

import sys
sys.path.insert(0, "../")

from math import pi

import tables
import pylab
import Image
import numpy as np


NoSuchNodeError = tables.exceptions.NoSuchNodeError


def tiled_gfs(images, sym_cm=True, global_cm=True):
    """
    Return an Image (PIL) displaying all the fields stored
    in *images*.

    *images* is supposed to be of shape (N, h, w) where *N* is the number
    of fields and (h * w) is each fields size.

    *sym_cm*     Should the colormap be symmetric
    *global_cm*  Should the final image have a global colormap, or 
                 should the colormap be scaled on a per tile bases.
    """
    def vminmaxscale(arr, sym_cm):
        vmin = arr.min()
        vmax = arr.max()
        if sym_cm:
            vmax = max(-vmin, vmax)
            vmin = -vmax
        vscale = vmax - vmin
        return vmin, vmax, vscale

    N, h, w = images.shape

    H = int(np.sqrt(N))        # W, H are no. of horizontal/vertical tiles
    W = H
    if H*W < N:
        H += 1
    if H*W < N:
        W += 1

    vmin, vmax, vscale = vminmaxscale(images, sym_cm)

    img = np.zeros( (H*(h+1), W*(w+1), 4), dtype=np.uint8 )
    for i in xrange(N):
        tile_x = i % W
        tile_y = i // W

        x = tile_x * (w+1)
        y = tile_y * (h+1)

        if not global_cm:
            vmin, vmax, vscale = vminmaxscale(images[i], sym_cm)

        tile_img = (images[i]-vmin) / vscale
        img[y:(y+h), x:(x+w)] = (pylab.cm.jet(tile_img)*255).astype(np.uint8)

    return Image.fromarray(img[:,:,0:3])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: %s <directory>" % sys.argv[0]
        exit(1)

    outdir = sys.argv[1]
    
    # Read parameter file
    params = {}
    execfile(outdir+"paramfile.py", params)

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

        pylab.plot(H*pies)
        pylab.xlabel("EM steps")
        pylab.ylabel("H*pi")
        pylab.title("Sparseness [H=%d]" % H)
        pylab.savefig(outdir+"/sparseness.png")
        print "Saved sparseness plot into %s/sparseness.png" % outdir
    except NoSuchNodeError:
        print "Skipped sparseness plot..."


    #=============================================================================
    # Sigma plot
    try:
        sigma = h5.root.sigma[:]

        pylab.cla()
        pylab.plot(sigma)
        pylab.xlabel("EM steps")
        pylab.ylabel("sigma")
        pylab.title("Noise level")
        pylab.savefig(outdir+"/sigma.png")
        print "Saved sigma plot into %s/sigma.png" % outdir
    except NoSuchNodeError:
        print "Skipped sigma plot"

    
    #=============================================================================
    # N_use plot
    try:
        N_use = h5.root.N_use[:]

        pylab.cla()
        pylab.plot(N_use)
        pylab.xlabel("EM steps")
        pylab.ylabel("N_use")
        pylab.title("Number of datapoints used (ET cuttting)")
        pylab.savefig(outdir+"/n_use.png")
        print "Saved N_use plot into %s/n_use.png" % outdir
    except NoSuchNodeError:
        print "Skipped N_use plot"

    #=============================================================================
    # nx/ny plot
    try:
        from old_off_plot import off_plot

        gabor_params = h5.root.gabor_params[:]
        gabor_errors = h5.root.gabor_errors[:]

        nx = gabor_params[:, 6] * gabor_params[:, 2] / 2 / pi   # XXX
        ny = gabor_params[:, 6] * gabor_params[:, 3] / 2 / pi
        error = -gabor_errors[:]

        pylab.cla()
        pylab.scatter(nx, ny, marker='o', cmap=pylab.cm.Greys, linewidths=0.0)
        diag = np.linspace(0, 1, 100)
        pylab.plot(diag, diag, color='r')
        pylab.axis([0, 1, 0, 1])
        pylab.title("nx/ny plot")
        pylab.xlabel("nx")
        pylab.xlabel("ny")
        pylab.savefig(outdir+"/nxny.png")


        pylab.figure()
        pylab.cla()
        off_plot(gabor_params, gabor_errors, outdir+"/nxny_old.png", discard_frac=0.01, option=3, axis=[0., 1.])
        print "Saved old_nxny plot into %s/nxny_old.png" % outdir

        print "Saved nxny plot into %s/nxny.png" % outdir
    except (NoSuchNodeError, ImportError):
        print "Skipped nx/ny plot!"
        gabor_params = None
        gabor_errors = None

    #=============================================================================
    # W plots
    W = h5.root.W[:]

    # Reorder according to gabor matching?
    #if gabor_params is not None:
    #    if gabor_params.shape[0] == H:
    #        sigma_x = gabor_params[:, 2]
    #        sigma_y = gabor_params[:, 3]
    #        idx = np.argsort(sigma_y)
    #        W = W[:, idx, :]
    #    else:
    #        print "Number of gabor_params != H (%d != %d)" % (gabor_params.shape[0], H)

    # Fuse channel splitting??  (XXX subtracting  is technically a wrong visualization! XXX)
    if params['channel_splitted']:
        D2 = np.sqrt(D // 2)
        W = W.reshape(steps, H, 2*D2, D2)
        W = W[:, :, 0:D2, :] - W[:, :, D2:, :]
    else:
        D2 = np.sqrt(D)
        W = W.reshape(steps, H, D2, D2)

    zoom = 3
    for s in xrange(steps):
        gfs = W[s]
        img = tiled_gfs(gfs, sym_cm=True, global_cm=False)
        img = img.resize( (zoom*img.size[0], zoom*img.size[1]) )
        img.save(outdir+"/W_%03d.png" % s)

    h5.close()

    #=============================================================================
    # nx/ny plot
    #if gabor_params is not None:
    #    from gabor_fitting import off_plot 
    # 

