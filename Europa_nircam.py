# -------------------------------------------------
# JWST/NIRSpec IFU extraction
# Villanueva, Faggi, NASA-GSFC
# November 2022
# -------------------------------------------------
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import medfilt
from scipy.ndimage import rotate, center_of_mass

bfile = 'images/jw01250-o001_t001_nircam_clear-f140m-sub64p/jw01250-o001_t001_nircam_clear-f140m-sub64p_i2d.fits'
gfile = 'images/jw01250-o001_t001_nircam_clear-f212n-sub64p/jw01250-o001_t001_nircam_clear-f212n-sub64p_i2d.fits'

bdata = fits.getdata(bfile)
bdata/= np.max(bdata)
gdata = fits.getdata(gfile)
gdata/= np.max(gdata)
img = np.zeros([bdata.shape[0],bdata.shape[1],3])
img[:,:,1] = gdata*0.9
img[:,:,0] = (gdata - bdata*0.77)*4.0
img[:,:,2] = bdata*0.6

img = np.rot90(img)
img = np.flip(img,1)

plt.imshow(img)
plt.savefig('nircam.png', dpi=300)
plt.show()
exit()
