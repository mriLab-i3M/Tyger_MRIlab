import os
import sys
this_file_path = os.path.abspath(__file__)
rare_recon_dir = os.path.abspath(os.path.join(os.path.dirname(this_file_path), '..'))
sys.path.append(rare_recon_dir)
from snraware.fromMATtoMRD3D_RARE_noise import matToMRD
from snraware.fromMRDtoMAT3D_noise import export
import subprocess
import time
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

out_field = 'denoisingImg'

pathMAT = '/home/teresa/Documentos/Next1/microsoft_ruido/RarePyPulseq.2025.10.24.14.18.10.422.mat'
pathMRD_or = '/home/teresa/Documentos/Next1/microsoft_ruido/RarePyPulseq.2025.10.24.14.18.10.422_TESTmarge.mrd'
pathMRD_ia = '/home/teresa/Documentos/Next1/microsoft_ruido/RarePyPulseq.2025.10.24.14.18.10.422_ia_TESTmarge.mrd'

pathSHpipeline = 'snraware/pipeline_denoising.sh'

runTyger = 1

if runTyger == 1:
    matToMRD(pathMAT, pathMRD_or)

    start_time = time.time()
    subprocess.run(
        ["bash", pathSHpipeline, pathMRD_or, pathMRD_ia],
        check=True
    )

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Denoising pipeline time: {total_duration:.2f} seconds")

    export(pathMRD_ia, pathMAT, out_field)


## CHECKING RESULT

rawData_pos = sio.loadmat(pathMAT)
img3D_tyger = rawData_pos[out_field][0]
img3D_or = np.abs(rawData_pos['image3D'])
print(img3D_or.shape)

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