#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: Academic Free License (AFL) v3.0
#

from __future__ import division

import numpy as np
from math import sqrt
from tables import openFile
from pulp.utils import iter_to_str
from os import mkdir, environ

from mpi4py import MPI

from matplotlib import pyplot
import pylab as pl


class PictureCreator():

    def __init__(self, h5file):
        self.h5file = openFile(h5file, 'r')
        self.variables = self.h5file.root.__members__

    def create_W_pic(self, W, savepath, fields, fig_shape=None, conv_cmap=None):
        """ Creates GF/RF picture from W matrix. 

        Expects *W*: of shape (num_GF, D=size**2), *savepath*, 
        *fields* : a list with the GF to plot. Generates and Saves a 
        (close to squared) figure at *savepath*.

        *fig_shape* is an optional tuple : (num_rows ,num_columns, scale)

        *conv* optional list of the colormaps to use per basis function.
        When *conv* == None, jet cmap is used by default.
        """
        H = len(fields)
        W_tmp = W[fields, :]
        H, D = W_tmp.shape
        D2 = np.int(np.sqrt(D))

        #Set cmap list
        if conv_cmap == None: 
            conv_list = [pyplot.cm.jet]
            conv_list *= H
        else: 
            if len(conv_cmap) != H: raise ValueError("Input list *conv_cmap* must be of length H")
            conv_list = conv_cmap

        #Set picture's shape 
        if fig_shape == None:
            rows = int(np.ceil(sqrt(H)))
            cols = int(np.round(sqrt(H)))
            figsize = ((8.*cols)/rows, 8.)
        else:
            rows, cols, scale = fig_shape
            figsize = ((scale*cols)/rows, scale)


        pyplot.figure(1, figsize=figsize)
        for ind_plot in xrange(H):
            spl = pl.subplot(rows, cols, ind_plot+1)
            spl.cla()
            val_max = np.max(W_tmp[ind_plot])
            val_min = np.min(W_tmp[ind_plot])
            val_max = max(np.abs(val_max), np.abs(val_min))
            if val_max == 0.: val_max = 0.001 #We want the zero case to be "green"
            val_min = -val_max

            spl.imshow(W_tmp[ind_plot].reshape(D2,D2), interpolation='nearest', cmap=conv_list[ind_plot], vmin=val_min , vmax=val_max)

            pyplot.axis('off')
        pl.savefig(savepath)
        pyplot.close(1)


    def create_W_pics(self, it=None, comm=MPI.COMM_WORLD, out_path=None, comp_rate=1, filenames=[]):
        """ Creates sequence of images from W.

        Requires None or list. If *it*=None creates pictures of all GF 
        saved at folder *.../h5file* (-.h5). If *it*=list checks for lenght. 
        Lenght == 1 : saves this iteration number. If length == 2, first < 
        second, saves pictures in this interval. *comp_rate*, integer that allows
        to plot ONLY each comp_rate*i (i integer) field.

        """
        if 'W' in self.variables:
            W = self.h5file.root.W[:]
            num_W, H, D = W.shape
            D2 = np.int(np.sqrt(D)) 
            if D2**2 != D: #We correct for channel splitting
                D2 = np.int(np.sqrt(D//2))
                D = D2**2
                temp_W = W.reshape(num_W, H, 2*D2, D2)
                W = (temp_W[:, :, 0:D2, :] - temp_W[:, :, D2:2*D2, :]).reshape(num_W, H, D)
                del temp_W

        if it == None: iter_list = xrange(num_W) #Print_all case
        #In all the rest of the cases it should be a list
        elif it.__class__ != [].__class__: raise TypeError("Input should be a list")
        else: #Print a specific number           
            if len(it) == 1: iter_list = it
            elif len(it) == 2:
                if it[0] >= it[1]: print '\n\n OK, nothing will be printed \n\n'
                iter_list = xrange(it[0], it[1])
            else: raise ValueError("list should contain 1 or 2 elements!")

        if out_path == None: out_dir = self.h5file.filename[0:-3]
        else: out_dir = out_path
        if comm.rank == 0: mkdir(out_dir)  #The path is only generated at processor 0

        fields = range(0, H, comp_rate)

        if len(iter_list) != len(filenames):
            for indW in iter_list:
                save_path = out_dir + '/W_' + iter_to_str(indW+1, num_W) + '.png'
                self.create_W_pic(W[indW, :,:], save_path, fields)
        else:
            num_order = 0
            for indW in iter_list:
                save_path = out_dir + filenames[num_order]
                self.create_W_pic(W[indW, :,:], save_path, fields)
                num_order +=1
    


    def close(self):
        self.h5file.close()

