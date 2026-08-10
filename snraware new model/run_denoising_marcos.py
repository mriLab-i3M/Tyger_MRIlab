import time
from pathlib import Path
import tyger_functions as tf
import scipy.io as sio
import numpy as np
from pathlib import Path

############################### INPUTS #####################################
pathMAT = r"C:\Users\teres\OneDrive\Documentos\_Projects\SNRAware\Data\marcos\RarePyPulseq.2025.11.07.18.05.42.770.mat"

out_field = 'image3D_den'
out_field_k = 'kSpace3D_den'

runTyger = 1

# model options 500, 219
model = 500

if model == 219:
    out_field += '_219'
    out_field_k += '_219'
elif model == 500:
    out_field += '_500'
    out_field_k += '_500'


########################### Run Denoising ##################################
pathMRD_or = pathMAT.replace(".mat", ".mrd")
pathMRD_ia = pathMAT.replace(".mat", "_ia.mrd")

for p in (pathMRD_or, pathMRD_ia):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

if runTyger == 1:
    tf.matToMRD(pathMAT, pathMRD_or)

    start_time = time.time()

    if model == 219:
        tf.run_tyger_denoising_219(pathMRD_or, pathMRD_ia)
    elif model == 500:
        tf.run_tyger_denoising_500(pathMRD_or, pathMRD_ia)  

    tf.exportMAT(pathMRD_ia, pathMAT, out_field, out_field_k)

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Denoising pipeline time: {total_duration:.2f} seconds")

########################### Check result ##################################

rawData_pos = sio.loadmat(pathMAT)
img3D_or = np.abs(rawData_pos['image3D'])
print(img3D_or.shape)
img3D_tyger = rawData_pos[out_field]
print(img3D_tyger.shape)
img3D_qcoil = rawData_pos[out_field+'_ori']
print(img3D_qcoil.shape)

# tf.plot_slider_2(np.abs(img3D_or), np.abs(img3D_tyger), titles=("Original", "SNRAware"), vmax_init=1.0)
tf.plot_slider_3(img3D_or, np.abs(img3D_qcoil), np.abs(img3D_tyger), titles=("FFT", "Q-coil correction", "SNRAware"), vmax_init=0.7)