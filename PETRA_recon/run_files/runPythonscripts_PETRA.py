import subprocess
import time
import scipy.io as sio
import yaml
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


## INPUTS

# rawData = "/home/tyger/tyger_BoMapsShow/PETRA_Phys1/PETRA.2025.06.20.14.10.11.662.mat"  
# rawData = "/home/tyger/tyger_BoMapsShow/PETRA_Phys1/PETRA.2024.12.19.19.38.08.208.mat"  

rawData = "/home/tyger/Tyger_MRIlab/toTest/Petra_tyger/PETRA.2025.07.23.19.52.53.484.mat"
out_field = "imgReconTygerOct"
runTyger = 1
## RECON CODE
if runTyger == 1:
    # Tiempo total
    start_total = time.time()

    # Paso 1: fromMATtoMRD
    start1 = time.time()
    p1 = subprocess.Popen(
        ["python3", "PETRA_recon/recon_scripts/fromMATtoMRD3D_PETRA.py", "-i", rawData],
        stdout=subprocess.PIPE,
    )

    # Paso 2: Código python que ejecutaré desde Tyger
    yml_path = "PETRA_recon/yml_files/stream_recon_gpu_phys1_python_PETRA.yml"
    with open(yml_path, "r") as f:
        config = yaml.safe_load(f)

    args = config["args"]

    p2 = subprocess.Popen(
        ["python3", "PETRA_recon/recon_scripts/stream_recon_PETRA.py"]+args,
        stdin=p1.stdout,
        stdout=subprocess.PIPE
    )

    p1.stdout.close()
    p1.wait()  # <-- Esperamos a que termine p1
    end1 = time.time()
    # Paso 3: fromMRDtoMAT
    p3 = subprocess.Popen(
        ["python3", "PETRA_recon/recon_scripts/fromMRDtoMAT3D.py", "-o", rawData, "-of", out_field],
        stdin=p2.stdout
    )
    p2.stdout.close()
    p2.wait()  # <-- Esperamos a que termine p2
    end2 = time.time()
    p3.communicate()  # <-- Esperamos a que termine p3
    end3 = time.time()

    # Medir duraciones
    duration1 = end1 - start1           # fromMATtoMRD
    duration2 = end2 - end1             # Tyger
    duration3 = end3 - end2             # fromMRDtoMAT
    total_duration = end3 - start_total

    # Mostrar resultados
    print(f"Duración fromMATtoMRD:   {duration1:.2f} segundos")
    print(f"Duración Tyger recon:    {duration2:.2f} segundos")
    print(f"Duración fromMRDtoMAT:   {duration3:.2f} segundos")
    print(f"Tiempo total del proceso: {total_duration:.2f} segundos")


## CHECKING RESULT

rawData_pos = sio.loadmat(rawData)
img3D_tyger = rawData_pos[out_field][0]
print('Tyger img shape: ',img3D_tyger.shape)
# img3D_or = np.abs(rawData_pos['ImageFFT'])
print(rawData_pos['axesOrientation'])
kSpaceArray = rawData_pos['kSpaceArray']
img3D_or = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kSpaceArray)))

print(img3D_or.shape, np.max(img3D_or), np.min(img3D_or))
## PLOT compSlice
nSlice = 20
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.imshow(np.abs(img3D_or[:,nSlice,:]), cmap='gray')
ax1.axis('off')  
ax1.set_title('Original')

ax2.imshow(np.abs(img3D_tyger[nSlice,:,:]), cmap='gray')
ax2.axis('off')
ax2.set_title('Tyger')

plt.tight_layout()
plt.savefig('PETRA_recon/compTyger.png', bbox_inches='tight', dpi=300)
# plt.show()


## PLOT slicer
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
