import os
import sys
this_file_path = os.path.abspath(__file__)
rare_recon_dir = os.path.abspath(os.path.join(os.path.dirname(this_file_path), '..'))
sys.path.append(rare_recon_dir)
from snraware.fromMATtoMRD3D_RAREdouble_noise import matToMRD
from snraware.fromMATtoMRD3D_RAREdouble_old import matToMRD_old
from snraware.fromMRDtoMAT3D_noise import export
import subprocess
import time
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path

############################### INPUTS #####################################
pathMAT = '/media/teresa/E090-BAA0/SNRAware/brain_neurho/mat/brainIR.mat'
input_echoes = 'odd'    # Options: 'odd', 'even', 'all'

out_field = 'image3D_den'
out_field_k = 'kSpace3D_den'

runTyger = 1

############################################################################

if input_echoes == 'even': 
    input_field_raw = 'sampled_eve'
    out_field1 = out_field + '_even'
    out_field_k1 = out_field_k + '_even'
else: # Odd, all, or any other (wrong option)
    input_field_raw = 'sampled_odd'
    out_field1 = out_field + '_odd' 
    out_field_k1 = out_field_k + '_odd'

pathMRD_or = pathMAT.replace("/mat/", "/mrd_local/").replace(".mat", ".mrd")
pathMRD_ia = pathMAT.replace("/mat/", "/mrd_ia/").replace(".mat", "_ia.mrd")

for p in (pathMRD_or, pathMRD_ia):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

pathSHpipeline = 'snraware/pipeline_denoising.sh'
# pathSHpipeline = 'snraware/pipeline_denoising_info.sh'   # Show info 

if runTyger == 1:
    
    try:
        matToMRD(pathMAT, pathMRD_or,input_field_raw)          # Actual rawDatas
    except:
        matToMRD_old(pathMAT, pathMRD_or,input_field_raw)      # Old rawDatas

    start_time = time.time()
    subprocess.run(
        ["bash", pathSHpipeline, pathMRD_or, pathMRD_ia],
        check=True
    )

    export(pathMRD_ia, pathMAT, out_field1, out_field_k1)
    
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Denoising pipeline time: {total_duration:.2f} seconds")
    
    if input_echoes == 'all':
        out_field2 = out_field + '_even'
        out_field_k2 = out_field_k + '_even'
        out_field_all = out_field + '_all'
        out_field_k_all = out_field_k + '_all'
        input_field_raw = 'sampled_eve'
        try:
            matToMRD(pathMAT, pathMRD_or,input_field_raw)          # Actual rawDatas
        except:
            matToMRD_old(pathMAT, pathMRD_or,input_field_raw)      # Old rawDatas

        start_time = time.time()
        subprocess.run(
            ["bash", pathSHpipeline, pathMRD_or, pathMRD_ia],
            check=True
        )

        export(pathMRD_ia, pathMAT, out_field2, out_field_k2)
        
        end_time = time.time()
        total_duration = end_time - start_time
        print(f"Denoising pipeline time 2: {total_duration:.2f} seconds")
        
        rawData = sio.loadmat(pathMAT)
        img_odd_den = rawData[out_field1]
        img_eve_den = rawData[out_field2]
        img_den = (np.abs(img_odd_den) + np.abs(img_eve_den)) / 2
        rawData[out_field_all] = img_den
        kSpace3D_den = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(img_den)))
        rawData[out_field_k_all] = kSpace3D_den
        sio.savemat(pathMAT, rawData)

## CHECKING RESULT

rawData_pos = sio.loadmat(pathMAT)
img3D_tyger = np.abs(rawData_pos[out_field1])
try:
    img3D_or = np.abs(rawData_pos['image3D_odd_echoes'])
except:
    img3D_or = np.abs(rawData_pos['image3D'])

print(img3D_tyger.shape)

## PLOT slicer
nSlice1 = img3D_or.shape[0] // 2
nSlice2 = img3D_tyger.shape[0] // 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.25) 

im1 = ax1.imshow(img3D_or[nSlice1, :, :], cmap='gray')
ax1.axis('off')
ax1.set_title('Original')

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