import os
import sys
this_file_path = os.path.abspath(__file__)
rare_recon_dir = os.path.abspath(os.path.join(os.path.dirname(this_file_path), '..'))
sys.path.append(rare_recon_dir)
from snraware.fromMATtoMRD3D_RARE_noise import matToMRD
from snraware.fromMATtoMRD3D_RARE_old import matToMRD_old
from snraware.fromMRDtoMAT3D_noise import export
import subprocess
import time
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path

############################### INPUTS #####################################

pathMAT = '/home/teresa/Descargas/RarePyPulseq.2025.07.28.08.55.05.171.mat'

out_field = 'image3D_den'
out_field_k = 'kSpace3D_den'

runTyger = 1

############################################################################

pathMRD_or = pathMAT.replace("/mat/", "/mrd_local/").replace(".mat", ".mrd")
pathMRD_ia = pathMAT.replace("/mat/", "/mrd_ia/").replace(".mat", "_ia.mrd")

for p in (pathMRD_or, pathMRD_ia):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

pathSHpipeline = 'snraware/pipeline_denoising.sh'
# pathSHpipeline = 'snraware/pipeline_denoising_info.sh'   # Show info 

if runTyger == 1:
    
    try:
        matToMRD(pathMAT, pathMRD_or)          # Actual rawDatas
    except:
        matToMRD_old(pathMAT, pathMRD_or)      # Old rawDatas

    start_time = time.time()
    subprocess.run(
        ["bash", pathSHpipeline, pathMRD_or, pathMRD_ia],
        check=True
    )
    
    export(pathMRD_ia, pathMAT, out_field, out_field_k)
    
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Denoising pipeline time: {total_duration:.2f} seconds")


## CHECKING RESULT

rawData_pos = sio.loadmat(pathMAT)
img3D_tyger = rawData_pos[out_field]
try:
    img3D_or = np.abs(rawData_pos['image3D_odd_echoes'])
except:
    img3D_or = np.abs(rawData_pos['image3D'])
# img3D_or = np.abs(rawData_pos['img_corr'])
# img3D_or = np.abs(rawData_pos['img_denoising'][0])
# img3D_or = np.abs(rawData_pos['denoisingImg'])[0]
print(img3D_tyger.shape)
# img3D_or = img3D_or - img3D_tyger

## PLOT slicer
nSlice1 = img3D_or.shape[0] // 2
nSlice2 = img3D_tyger.shape[0] // 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.25) 

# im1 = ax1.imshow(img3D_or[nSlice1, :, :], cmap='gray',vmax=np.max(img3D_or)/10,vmin=np.min(img3D_or))
im1 = ax1.imshow(img3D_or[nSlice1, :, :], cmap='gray')
ax1.axis('off')
ax1.set_title('Original')

# im2 = ax2.imshow(img3D_tyger[nSlice2,:,:], cmap='gray',vmax=np.max(img3D_tyger)/10,vmin=np.min(img3D_tyger))
im2 = ax2.imshow(img3D_tyger[nSlice2,:,:], cmap='gray')
ax2.axis('off')
ax2.set_title('Tyger')

# Sliders
ax_slider1 = plt.axes([0.15, 0.1, 0.3, 0.03])
slider1 = Slider(ax_slider1, '', 0, img3D_or.shape[0]-1, valinit=nSlice1, valfmt='%d')

ax_slider2 = plt.axes([0.55, 0.1, 0.3, 0.03])
slider2 = Slider(ax_slider2, '', 0, img3D_tyger.shape[0]-1, valinit=nSlice2, valfmt='%d')

def update1(val):
    idx = int(slider1.val)
    im1.set_data(img3D_or[idx, :, :])
    fig.canvas.draw_idle()

def update2(val):
    idx = int(slider2.val)
    im2.set_data(img3D_tyger[idx, :, :])
    fig.canvas.draw_idle()

slider1.on_changed(update1)
slider2.on_changed(update2)

plt.show()