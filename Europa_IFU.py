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
from scipy.optimize import curve_fit

# Dither position
drad = 6
pos0 = [25,25]
pos1 = [22,23]
posc = [25,25]
fbase = 'data/jw01250002001'
gratings = ['G140H','G235H','G395H']
files = ['03101','03103','03105']
detectors = ['nrs1','nrs2']

# Define resulting array and extraction mask
xs = 49; ys = 47
icube = np.zeros([3,4000,xs,ys])
scube = np.zeros([3,4000,3])
ncube = np.zeros([3,3],dtype=int)
maskc = np.zeros([xs,ys]); npc=0
for i in range(xs):
    for j in range(ys):
        di = i-pos0[0]
        dj = j-pos0[1]
        dd = np.sqrt(di*di + dj*dj)/drad
        if dd<=1.0: maskc[i,j]=1.0; npc+=1
    #Endfor
#Endfor

# Iterate across settings
for igrating in range(1,len(gratings)):
    #if igrating!=2: continue
    for idetector in range(len(detectors)):
        detector = detectors[idetector]
        for dither in [0,1]:
            if dither==0: pos=pos0; sdither='00001'
            else: pos=pos1; sdither='00002'
            file = '%s_%s_%s_%s_s3d.fits' % (fbase,files[igrating],sdither,detector)
            fr = fits.open(file)
            dpx = fr[1].header['CDELT1']*3600.0
            dpy = fr[1].header['CDELT2']*3600.0
            wv0 = fr[1].header['CRVAL3']
            dwv = fr[1].header['CDELT3']
            dtt = fr[0].header['EFFEXPTM']
            ac2 = fr[1].header['PIXAR_A2']
            jac = fr[1].header['PHOTUJA2']*1e-6*ac2
            fr.close()
            data = fits.getdata(file)/(dtt*jac); frame = data*0.0
            if dither==0: npts = data.shape[0]-150; xs = data.shape[1]; ys = data.shape[2]; dx=0; dy=0                
            else: dx = pos0[0]-pos1[0]; dy = pos0[1]-pos1[1]
            if   dx>=0 and dy>=0: frame[0:npts,dx:,dy:] = data[0:npts,0:xs-dx,0:ys-dy]
            elif dx>=0 and dy<0:  frame[0:npts,dx:,0:ys+dy] = data[0:npts,0:xs-dx,-dy:]
            elif dx<0  and dy>=0: frame[0:npts,0:xs+dx,dy:] = data[0:npts,-dx:,0:ys-dy]
            elif dx<0  and dy<0:  frame[0:npts,0:xs+dx,0:ys+dy] = data[0:npts,-dx:,-dy:]
            if dither==0: data0 = frame*1.0
            else: data1 = frame*1.0
        #Endfor

        # Mask data outside of the processing region
        d0  = data0[1000,:,:]; d1=data1[1000,:,:]; img=0
        mask = 0.0*maskc
        ind = ((d0!=0)*(d1!=0)).nonzero()
        mask[ind] = 1.0

        # Iterate across wavelength
        wvls = np.arange(npts)*dwv + wv0
        for i in range(0,npts):
            d0  = data0[i,:,:]; d1=data1[i,:,:]
            if (d0[pos0[1],pos0[0]+drad+2]==0 and d0[pos0[1],pos0[0]+drad+3]==0): continue
            ind = ((d0*maskc!=0)*(d1*maskc!=0)).nonzero()
            if len(ind[0])!=npc: continue

            # Calculate rms and background
            dt  = (d0 - d1)*mask*(1.0-maskc)
            ind = (dt!=0).nonzero()
            dt[ind] -= np.median(dt[ind])
            rms = np.sqrt(np.median(dt[ind]*dt[ind]))

            # See if we have abnormally negative values
            ibad = ((d0<-rms*10.0)+(d0==0)+(d0>1e8)).nonzero()
            if len(ibad[0])>0: d0[ibad]=d1[ibad]
            ibad = ((d1<-rms*10.0)+(d1==0)+(d1>1e8)).nonzero()
            if len(ibad[0])>0: d1[ibad]=d0[ibad]

            # Remove background bad pixels
            dt  = (d0 - d1)*mask*(1.0-maskc)
            ibad = (dt>rms*10.0).nonzero()
            if len(ibad[0])>0: d0[ibad]=d1[ibad]
            ibad = (dt<-rms*10.0).nonzero()
            if len(ibad[0])>0: d1[ibad]=d0[ibad]

            # Remove disk bad pixels
            ind = (maskc!=0).nonzero()
            avg = np.median(d0[ind])
            dt  = (d0 - d1)*maskc
            ibad = (dt>avg*5.0).nonzero()
            if len(ibad[0])>0: d0[ibad]=d1[ibad]
            ibad = (dt<-avg*5.0).nonzero()
            if len(ibad[0])>0: d1[ibad]=d0[ibad]

            # Store clean image
            if igrating<=1: gamma=0.4
            else: gamma=0.4*0.86
            img = (d0 + d1)*mask/2.0
            npi = ncube[igrating,0]
            scube[igrating,npi,0] = wvls[i]
            scube[igrating,npi,1] = np.sum(img*maskc)*gamma
            scube[igrating,npi,2] = rms*np.sqrt(npc)*gamma
            icube[igrating,npi,:,:] = img*1.0*gamma
            ncube[igrating,0] += 1
            ncube[igrating,idetector+1] += 1
        #Endfor
    #Endfor

    # Save extract
    fw = open('spec_%s.txt' % gratings[igrating],'w')
    for i in range(0,ncube[igrating,0],1): fw.write('%.6f %e %e\n' % (scube[igrating,i,0],scube[igrating,i,1],scube[igrating,i,2]))
    fw.close()

    # Plot spectra
    npi = ncube[igrating,0]
    x = scube[igrating,0:npi,0]
    y = scube[igrating,0:npi,1]
    y[ncube[igrating,1]-1] = np.nan
    plt.plot(x,y)
#Endfor
plt.yscale('log')
plt.show()

