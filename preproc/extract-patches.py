#!/usr/bin/env python
#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: Academic Free License (AFL) v3.0
#

from __future__ import division

import sys
sys.path.insert(0, "lib/")

import os.path
import numpy as np
import tables
#import pylab

from pulp.utils.datalog import dlog
from pulp.utils.autotable import AutoTable
import pulp.preproc.image as pri


#=============================================================================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: %s <images.h5> <size>" % sys.argv[0]
        exit(1)

    images_fname = sys.argv[1]
    size = int(sys.argv[2])
    oversize = 2*size
    N_patches = 1000000
    min_var = 0.0001

    out_fname = "patches-%d" % size
    out_tbl = AutoTable(out_fname+".h5")

    images_h5 = tables.openFile(images_fname, "r")
    images = images_h5.root.images

    N_images = images.shape[0]
    #ppi = (N_patches // N_images // 10) + 1
    ppi = 4
    
    for n in xrange(N_patches):
        if n % 1000 == 0:
            dlog.progress("Extracting patch %d" % n, n/N_patches)
        if n % ppi == 0:
            while True:
                img = images[np.random.randint(N_images)]
                img = img / img.max()
                oversized_batch = pri.extract_patches_from_single_image(img, (oversize, oversize), ppi)
                patches_batch = oversized_batch[:, (size//2):(size//2+size), (size//2):(size//2+size)]

                variance = np.var( patches_batch.reshape([ppi, -1] ), axis=1)
                if np.alltrue(variance > min_var):
                    break

        out_tbl.append('oversized', oversized_batch[n%ppi])
        out_tbl.append('patches', patches_batch[n%ppi])

out_tbl.close()

