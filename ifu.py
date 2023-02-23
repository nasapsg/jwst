# -------------------------------------------------
# JWST/NIRSpec IFU/Prism extraction
# Villanueva, NASA-GSFC
# February 2023
# -------------------------------------------------
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy.ndimage as scimg

base = 'jw02416003001_03101'; px = 14; py = 32; ndither = 4;  # Comet
#base = 'jw01128002001_03102'; px = 24; py = 25; ndither = 20; # SNAP-2 (2MASS J16194609+5534178) v=16.23
nirang = 138.0
lmin = 20; lmax = 30
drad = 2
dbkg = 5

# Read and co-register frames
for i in range(ndither):
    file = 'data/%s_%05d_nrs1/%s_%05d_nrs1_s3d.fits' % (base,i+1,base,i+1)
    fr  = fits.open(file)
    ang = (fr[0].header['GS_V3_PA'] + nirang)*np.pi/180.0
    ox  = fr[0].header['XOFFSET']
    oy  = fr[0].header['YOFFSET']
    dpx = fr[1].header['CDELT1']*3600.0
    dpy = fr[1].header['CDELT2']*3600.0
    wv0 = fr[1].header['CRVAL3']
    dwv = fr[1].header['CDELT3']
    dtt = fr[0].header['EFFEXPTM']
    ac2 = fr[1].header['PIXAR_A2']
    jac = fr[1].header['PHOTUJA2']*1e-6*ac2
    fr.close()
    data = fits.getdata(file)#/(dtt*jac)
    data *= 2.35040007004737E-13*1e6

    data = data[lmin:-lmax,:,:]; wv0 += dwv*lmin
    if i==0: 
        # Create photometric masks
        npts = data.shape[0]; xs = data.shape[2]; ys = data.shape[1]        
        frames = np.zeros([ndither, npts, ys, xs])
        maskc = np.zeros([ys,xs]); npc=0.0
        maskb = np.zeros([ys,xs]); npb=0.0
        for k in range(xs):
            for j in range(ys):
                di = k-px
                dj = j-py
                dd = np.sqrt(di*di + dj*dj)
                if dd<=drad: maskc[j,k]=1.0; npc+=1.0
                elif dd>drad and dd<=dbkg+drad: maskb[j,k]=1.0; npb+=1.0
            #Endfor
        #Endfor
        
        mask = np.zeros([npts, ys, xs])
        for k in range(xs):
            for j in range(ys):
                di = j-xs/2.0
                dj = k-ys/2.0
                dri = -round(di*np.cos(ang) - dj*np.sin(ang))
                if abs(dri)<16: mask[:,j,k] = 1.0
            #Endfor
        #Endfor
    #Endif

    # Align frames
    dx = -round((ox/dpx)*np.cos(ang) - (oy/dpy)*np.sin(ang))
    dy = -round((ox/dpx)*np.sin(ang) + (oy/dpy)*np.cos(ang))
    if   dx>=0 and dy>=0: frames[i, 0:npts,dy:,dx:] = data[0:npts,0:ys-dy,0:xs-dx]
    elif dx>=0 and dy<0:  frames[i, 0:npts,0:ys+dy,dx:] = data[0:npts,-dy:,0:xs-dx]
    elif dx<0  and dy>=0: frames[i, 0:npts,dy:,0:xs+dx] = data[0:npts,0:ys-dy,-dx:]
    elif dx<0  and dy<0:  frames[i, 0:npts,0:ys+dy,0:xs+dx] = data[0:npts,-dy:,-dx:]
    ind = (np.isfinite(frames[i,10,:,:]))
    cmask = np.zeros([ys,xs]) + np.nan
    cmask[ind] = 1.0
    if i==0: tmask=cmask
    else: tmask=tmask*cmask   
#Endfor

# Apply total mask
for i in range(ndither):
    for j in range(npts):
        frames[i,j,:,:] = frames[i,j,:,:]*tmask
    #Endfor
#Endfor

# Clean frames and perform aperture integration
wvls = np.arange(npts)*dwv + wv0
cube = np.nanmedian(frames,0)
spec = np.zeros(npts)
noise = np.zeros(npts)
inb = (maskb!=0).nonzero()
for i in range(npts):
    img = cube[i,:,:]

    # Remove abnormally negative values
    dt = frames[0,i,:,:] - frames[1,i,:,:]
    ind = ((dt!=0)*(np.isfinite(dt))).nonzero()
    rms = np.sqrt(np.nanmedian(dt[ind]*dt[ind]))/2.0
    ind = (img<-5.0*rms).nonzero()
    img[ind] = 0.0
    cube[i,:,:] = img

    # Perform aperture integration
    bkg = np.nansum(img*maskb)*(npc/npb)
    flx = np.nansum(img*maskc)
    spec[i] = flx - bkg
    noise[i] = rms*np.sqrt(npc)

    # plt.cla()
    # plt.imshow(img)
    # plt.title('%d %.2f' % (i,wvls[i]))
    # plt.show(block=False)
    # plt.pause(0.01)
    # plt.show(); exit()
#Endfor

# Save spectrum
print('Aperture is %.2f arcsec in diameter' % (dpx*(drad*2+1)))
fw = open('spec.txt','w')
for i in range(npts): fw.write('%.6f %e %e\n' % (wvls[i],spec[i],noise[i]))
fw.close()
plt.plot(wvls,spec)
plt.show()
exit()

# Perform pixel-py-extractions
thermal = np.zeros([ys,xs]) + np.nan
reflec = np.zeros([ys,xs])
co2 = np.zeros([ys,xs])
for i in range(xs):
    for j in range(ys):
        if tmask[j,i]==0: continue
        reflec[j,i] = np.mean(cube[10:365,j,i])
        cont = (np.sum(cube[680:700,j,i]) + np.sum(cube[720:740,j,i]))*0.5
        co2[j,i] = np.sum(cube[700:720,j,i]) - cont
        cont = (np.sum(cube[660:700,j,i]) + np.sum(cube[720:760,j,i]))*0.5
        thermal[j,i] = cont
    #Endfor
#Endfor

# Derive scaling factors from models
model = np.genfromtxt('PSG/psg_rad_gas.txt')
mspec = np.interp(wvls, model[:,0], model[:,1])
tgas = np.sum(mspec[700:720])/6.8638E+15
co2 = 1e-14*co2/tgas

# Apply gaussian filter
sigma = [0.8, 0.8]
cgas = scimg.gaussian_filter(co2, sigma, mode='constant')
ccnt = scimg.gaussian_filter(reflec, sigma, mode='constant')

dmax = 3500
fig, axs = plt.subplots(1,3, figsize=[16,4])
extent = np.asarray([0-px-0.5,xs-px-0.5,ys-py-0.5,0-py-0.5])*398.94

im = axs[0].imshow(cgas,extent=extent,vmin=-1,cmap='RdBu_r')
plt.colorbar(im, ax=axs[0], label='CO2 column density [1E14 m-2]')
axs[0].set_xlim([-dmax,dmax])
axs[0].set_ylim([-dmax,dmax])

im = axs[1].imshow(ccnt*1e6,extent=extent,cmap='turbo')
plt.colorbar(im, ax=axs[1], label='Mean reflected flux [uJy]')
axs[1].set_xlim([-dmax,dmax])
axs[1].set_ylim([-dmax,dmax])

pb = (np.arange(29)-14)*398.94
pg = np.sum(co2*(maskc+maskb),0)[px-14:px+15] + np.sum(co2*(maskc+maskb),1)[py-14:py+15]
pg = pg/np.max(pg)

pr = np.sum(reflec*(maskc+maskb),0)[px-14:px+15] + np.sum(reflec*(maskc+maskb),1)[py-14:py+15]
pr = pr/np.max(pr)

ind = (maskb>0).nonzero()
bkg = np.nanmedian(thermal[ind])
thermal = thermal - bkg
pt = np.nansum(thermal*(maskc+maskb),0)[px-14:px+15] + np.nansum(thermal*(maskc+maskb),1)[py-14:py+15]
pt = pt/np.max(pt)

axs[2].plot(pb,pr)
axs[2].plot(pb,pt,':')
axs[2].step(pb,pg)
axs[2].set_xlim([-dmax,dmax])
plt.savefig('ifu.ps')
plt.show()

