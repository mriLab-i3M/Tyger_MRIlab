import time
import subprocess
import io
import scipy.io as sio
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import os
import sys
this_file_path = os.path.abspath(__file__)
rare_recon_dir = os.path.abspath(os.path.join(os.path.dirname(this_file_path), '..'))
sys.path.append(rare_recon_dir)
from recon_scripts.fromMATtoMRD3D_RARE import matToMRD
from recon_scripts.fromMRDtoMAT3D import export
import bm4d 


## INPUTS

# rawData_path = '/home/teresa/Documentos/MBARARA25/brains_09.07.25/'
# rawData = "RareDoubleImage.2025.07.09.20.16.07.281.mat"     # T2. Contraste majo. Poco FOV en todas dir. 
# rawData = "RarePyPulseq.2025.07.09.17.57.50.004.mat"        # T1. JA. RF arriba. Poco FOV axial. 18 slices. Con zp.Artefacto de los grad en todos los slices.
# rawData = "RarePyPulseq.2025.07.09.19.13.55.213.mat"        # T1. Maja. RF arriba. Poco FOV axial. 18 slices. Sin zp. 


# rawData_path = '/home/teresa/Documentos/MBARARA25/10.07/'
# rawData = "RareDoubleImage.2025.07.10.17.49.55.504.mat"   # T2 movida. No vale nada. 
# rawData = "RarePyPulseq.2025.07.10.16.09.45.210.mat"      # T1. Poco FOV plano (RF arriba)
# rawData = "RarePyPulseq.2025.07.10.17.16.36.853.mat"      # T1. Buena con mucha estructura. RF abajo. Poco FOV axial. 

rawData_path = '/home/teresa/Documentos/MBARARA25/11.07/'
# rawData = "RareDoubleImage.2025.07.11.18.54.46.963.mat"   # T2 buena. 
rawData = "RarePyPulseq.2025.07.11.17.17.31.305.mat"      # T1 corta 6 ms - Para comparativa tiempos grad. PhTime = 2
# rawData = "RarePyPulseq.2025.07.11.17.33.06.602.mat"      # T1 corta 6 ms - Para comparativa BW! PhTime = 1
# rawData = "RarePyPulseq.2025.07.11.17.48.44.287.mat"      # T1. 4 ms. - Para comparativa BW! PhTime = 2
# rawData = "RarePyPulseq.2025.07.11.20.04.34.620.mat"      # STIR 
# rawData = "RarePyPulseq.2025.07.11.21.09.25.687.mat"        # T1 long. Se ven hasta los ojos!

rawData = rawData_path + rawData
# yml_file = "RARE_recon/yml_files/must_05.07.yml"
yml_file = "RARE_recon/yml_files/must_11.07.yml"

# out_field = "tygerCP_withoutShim"
out_field = "tygerCP"
# out_field = "tygerARTpk"

runTyger = 1

if runTyger == 1:
    print('Running Tyger Reconstruction...')
    start_time = time.time()

    # From MAT to MRD
    mrd_buffer = io.BytesIO()
    matToMRD(input=rawData, output_file=mrd_buffer)
    mrd_buffer.seek(0) 
    tyger_input_data = mrd_buffer.getvalue()

    # Run Tyger
    p2 = subprocess.run(
        ["tyger", "run", "exec", "-f", yml_file],
        input=tyger_input_data,
        stdout=subprocess.PIPE
    )

    p2_stdout_data = p2.stdout

    # From MRD to MAT
    tyger_output_buffer = io.BytesIO(p2_stdout_data)
    export(tyger_output_buffer, rawData, out_field)


    # Time monitorization 
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Tyger recon time: {total_duration:.2f} seconds")


## CHECKING RESULT
rawData_pos = sio.loadmat(rawData)
print('AxesOrientation: ', rawData_pos['axesOrientation'])
print('dFov: ', rawData_pos['dfov'])
print('BW: ', rawData_pos['bw_MHz'])
print(rawData_pos['phGradTime'])

img3D_tyger = rawData_pos[out_field][0]
# img3D_tyger = bm4d.bm4d(img3D_tyger, sigma_psd=1.4)
# print('Tyger img shape: ',img3D_tyger.shape)
img3D_or = np.abs(rawData_pos['image3D'])

# ## PLOT compSlice
nSlice = 10
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.imshow(img3D_or[nSlice,:,:], cmap='gray')
ax1.axis('off')  
ax1.set_title('Original')

ax2.imshow(img3D_tyger[nSlice,:,:], cmap='gray')
ax2.axis('off')
ax2.set_title('Tyger')

plt.tight_layout()
# plt.savefig('RARE_recon/compTyger.png', bbox_inches='tight', dpi=300)
# # plt.show()

# ## PLOT slicer
# nSlice1 = img3D_or.shape[0] // 2
# nSlice2 = img3D_tyger.shape[0] // 2

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# plt.subplots_adjust(bottom=0.25) 

# im1 = ax1.imshow(img3D_or[nSlice1, :, :], cmap='gray')
# ax1.axis('off')
# ax1.set_title('Original')

# im2 = ax2.imshow(img3D_tyger[nSlice2,:,:], cmap='gray')
# ax2.axis('off')
# ax2.set_title('Tyger')

# # Sliders
# ax_slider1 = plt.axes([0.15, 0.1, 0.3, 0.03])
# slider1 = Slider(ax_slider1, '', 0, img3D_or.shape[0]-1, valinit=nSlice1, valfmt='%d')

# ax_slider2 = plt.axes([0.55, 0.1, 0.3, 0.03])
# slider2 = Slider(ax_slider2, '', 0, img3D_tyger.shape[0]-1, valinit=nSlice2, valfmt='%d')

# def update1(val):
#     idx = int(slider1.val)
#     im1.set_data(img3D_or[idx, :, :])
#     fig.canvas.draw_idle()

# def update2(val):
#     idx = int(slider2.val)
#     im2.set_data(img3D_tyger[idx, :, :])
#     fig.canvas.draw_idle()

# slider1.on_changed(update1)
# slider2.on_changed(update2)

# plt.show()

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

# Flag para evitar recursión infinita
updating = {"active": False}

def update_synchronized(val):
    if updating["active"]:
        return
    updating["active"] = True

    idx = int(val)

    # Asegurarnos de estar dentro de rango
    idx1 = min(max(0, idx), img3D_or.shape[0]-1)
    idx2 = min(max(0, idx), img3D_tyger.shape[0]-1)

    # Actualizar sliders (sin desencadenar recursión)
    slider1.set_val(idx1)
    slider2.set_val(idx2)

    # Actualizar imágenes
    im1.set_data(img3D_or[idx1, :, :])
    im2.set_data(img3D_tyger[idx2, :, :])
    fig.canvas.draw_idle()

    updating["active"] = False

# Conectamos ambos sliders a la misma función
slider1.on_changed(update_synchronized)
slider2.on_changed(update_synchronized)

plt.show()