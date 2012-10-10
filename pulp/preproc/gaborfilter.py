#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: Academic Free License (AFL) v3.0
#

"""
This module provides the GaborFilter class, which encapsulates a 
complete gabor transform and provides functions to extract jets
at arbitrary points from images.

Basic example::

    import numpy as np
    import Image
    from pulp.utils.gabor import GaborFilter

    # Load image and convert into grayscale matrix
    I = np.assarray( Image.open("image.png").convert('L'), dtype=np.float )
    
    # Create gabor filter
    gf = GaborFilter(scales=5, dirs=8)
    gf.absolute = True     # only return absolute values (non-complex)

    # Apply the filter and store the the result in I_
    I_ = gf.filter(I)

"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from numpy import pi, exp, cos, sin

class GaborFilter(object):
    """
    This class encapsulates a gabor transform and provides functions to extract 
    jets at arbitrary points from images.

    Instances of :class:`GaborFilter` have some public member attributes:

    .. attribute:: GaborFilter.absolute

        Return complex feature components or only absolute values (Boolean, default False)
        
    .. attribute:: GaborFilter.normalize

        Normalize the returned feature vectors to 1. (Boolean, default False)

    .. note::
         XXX ToDo XXX

    The gabor filter implemented is the discretized version of

    ## math::
        \\tilde{I}_{\sigma,\phi}(\\vec{x}_0) = \int I(\\vec{x}) e^{ \\vec{x} - \\vec{x}_0 }

    """
    #..  e^{-({\vec{x_0}-\vec{x})^2}} e^{-i \omega t}

    def __init__(self, scales = 5, dirs = 8):
        """ Initialize a gabor filterbank. """
        self.normalize = False
        self.absolute = False
        self.scales = scales
        self.dirs = dirs
        self.kernel_ = np.zeros( (scales, dirs), dtype=object )
        self.kernel_shape = (0, 0)

    def _make_kernel(self, scale, rot, sigma, shape):
        """ Generate gabor wavelet with given parameters. The last argument
            (shape) specifies the shape of the matrix onto this wavelet will
            be drawn.  Example: gabor_kernel( 1., pi/2, pi, (256,256) )
        """
        h,  w  = shape
        h2, w2 = h//2, w//2
        gy, gx = np.ogrid[ -h2:h-h2 , -w2:w-w2 ]

        k = scale * np.asarray([cos(rot), sin(rot)])
        k2      = k[0]*k[0] + k[1]*k[1]
        sigma2  = sigma * sigma
        norm    = k2 / sigma2
        dc_free = exp( -sigma2/2 )

        kern = norm * np.exp( -k2*(gx*gx+gy*gy)/2./sigma2 ) * (np.exp( (gy*k[0]+gx*k[1])*1j ) - dc_free)
        return fftshift(kern)

    def _cache_kernel(self, shape):
        """ Manage a cache of last used fft'ed version of the used kernels. This cache 
            needs to be updated, whenever the imagesize changes.
        """
        if self.kernel_shape == shape: return
        self.kernel_shape = shape
        norm = np.sqrt( shape[0]*shape[1] )

        for s in range(self.scales):
            scale = pi * 2.** (- (s+2.)/2)
            for d in range(self.dirs):
                dir = pi/self.dirs * d
                k  = self._make_kernel(scale, dir, pi/2., shape)
                k_ = fft2(k) / norm
                self.kernel_[s, d] = k_

    def filter(self, image):
        """ Apply the gabor filter to the given image and return a
            4-dimensional array.  Jet-component are the first two indices, 3rd and 4th
            index are image positions. [scale,dir,y,x]

            Depending on the boolean instance attibutes *absolute* and *normalized*,
            the returned array will contain complex values or their absolute.
        """
        sh   = image.shape
        res  = np.ndarray( (self.scales, self.dirs, sh[0], sh[1]), dtype=complex )
        img_ = fft2( image )
        self._cache_kernel( sh )
        for s in range( self.scales ):
            for r in range( self.dirs ):
                o_  = img_ * self.kernel_[s][r]
                res[s, r] = ifft2( o_ )

        if self.absolute:
            res = np.abs(res)

        if self.normalize:
            scaling = np.sqrt(   (np.abs(res)**2).sum(0).sum(0) )
            res = res / scaling

        return res

