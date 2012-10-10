#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: Academic Free License (AFL) v3.0
#

"""

"""

import numpy as np
import numpy.fft as fft

# ============================================================================
# Extracting image patches

def extract_patches_from_single_image(image, shape, n_patches):
    """
    Extract *n_patches* image patches of size *shape* from *image*, where
    *shape* specifies the intended size of the image patches as (height, width)
    tuple.

    *image* may be a simple (height, width) shaped array, but may also be a
    higher dimensional array (e.g. color image or a gabor-filtered image).  If, for
    example, image is a (iheight, iwidth, scales, dirs) shaped gabor image (with a
    2-dimensional gabor-jet attached to each point), the resulting patches will be
    a 5-dimensional array: (num_patches, pwidth, pheight, scales, dirs).
    """
    iheight, iwidth = image.shape[0:2]
    shape_per_pixel = image.shape[2:]
    pheight, pwidth = shape     

    # Allocate array for patches
    patches_shape = (n_patches, pheight, pwidth) + shape_per_pixel
    patches = np.empty(patches_shape, dtype=image.dtype)

    # and extract...
    for i in xrange(n_patches):
        y_pos = np.random.randint(iheight-pheight+1)
        x_pos = np.random.randint(iwidth-pwidth+1)

        patches[i] = image[y_pos:(y_pos+pheight), x_pos:(x_pos+pwidth)]

    return patches

def extract_patches_from_images(images, shape, n_patches):
    """
    Extract *n_patches* image patches of size *shape* from *images*, where
    *shape* specifies the intended size of the image patches as (height, width)
    tuple.

    *images* may be a simple (n_images, height, width) shaped array, but may
    also be a higher dimensional array (e.g. color image or a gabor-filtered
    image).  If, for example, image is a (iheight, iwidth, scales, dirs) shaped
    gabor image (with a 2-dimensional gabor-jet attached to each point), the 
    resulting patches will be a 5-dimensional array: (num_patches, pwidth, 
    pheight, scales, dirs).
    """
    n_images, iheight, iwidth = images.shape[0:3]
    shape_per_pixel = images.shape[3:]
    pheight, pwidth = shape
    
    # Allocate array for patches
    patches_shape = (n_patches, pheight, pwidth) + shape_per_pixel
    patches = np.empty(patches_shape, dtype=images.dtype)

    # and extract...
    for i in xrange(n_patches):
        # Choose random image
        src_img = np.random.randint(n_images)
        y_pos = np.random.randint(iheight-pheight+1)
        x_pos = np.random.randint(iwidth-pwidth+1)

        patches[i] = images[src_img, y_pos:(y_pos+pheight), x_pos:(x_pos+pwidth)]

    return patches
    
# ============================================================================
# Pseudo Whitening

class PseudoWhiteningFilter:
    """
    This class implements pseudo whitening as described by Olshausen & Field in 
    XXX.
    """
    def __init__(self):
        self.kernel = np.zeros(0)

    def _gen_kernel(self, shape):
        """Generate and return the convolution filter in freq. space"""
        h, w = shape
        h2 = h // 2
        w2 = w // 2
        f0 = int(0.8 * min(h2, w2))
        y,x = np.ogrid[-h2:h-h2 , -w2:w-w2 ]
        F   = np.sqrt(x*x+y*y) * np.exp( -(np.sqrt(x*x+y*y)/f0)**4 )
        return fft.fftshift(F)

    def filter(self, images):
        """
        Given an array of image-patches *images*, pseudo whiten them and
        return a identically shaped array.
        """
        n, h, w = images.shape

        # Do we have a correctly shaped cached filter?
        if self.kernel.shape != (h, w):
            self.kernel = self._gen_kernel((h, w))

        # Allocate return array
        filtered = np.empty((n, h, w))

        # Apply filter in freq. space
        for i in xrange(n):
            I = images[i].astype(np.float)
            I_ = fft.fft2(I)
            J_ = self.kernel * I_
            J  = np.real(fft.ifft2(J_))
            filtered[i] = J
        
        return filtered

