# -------------------------------------------------
# JWST pipeline processing
# Villanueva, NASA-GSFC
# February 2023
# -------------------------------------------------
# To call the pipeline, first get the 'rate' files from MAST, then
# >conda activate jwst
# >python pipeline.py
# -------------------------------------------------
# To install the pipeline, first install anaconda: brew install anaconda (restart shell after installing)
# Add these lines to your initial script (e.g., .zshrc)
# export CRDS_PATH="/Users/glvillan/data/software/python/jwst_crds"
# export CRDS_SERVER_URL="https://jwst-crds.stsci.edu"
# Then run these commands
# >conda create -n jwst python
# >conda activate jwst
# >pip install jwst
# -------------------------------------------------
# To update pipeline
# >pip uninstall jwst
# >pip install jwst
# -------------------------------------------------
import os
os.environ["CRDS_PATH"] = "/Users/glvillan/data/software/python/jwst_crds"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
import jwst
from jwst.pipeline.calwebb_spec2 import Spec2Pipeline

base = 'jw02416003001_03101'; ndither = 4;  # Comet
#base = 'jw01128002001_03102'; ndither = 20; # SNAP-2 (2MASS J16194609+5534178) v=16.23
overwrite = False

# Read and co-register frames
for i in range(ndither):
    fdir = 'data/%s_%05d_nrs1' % (base,i+1)
    ifile = '%s/%s_%05d_nrs1_rate.fits' % (fdir,base,i+1)
    ofile = '%s/%s_%05d_nrs1_s3d.fits' % (fdir,base,i+1)

    # Run the pipeline
    if os.path.exists(ofile) and not overwrite: continue
    result = Spec2Pipeline.call(ifile, output_dir=fdir, save_results=True)
#Endfor
