import subprocess
import time
import scipy.io as sio
import yaml
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


## INPUTS

rawData_path = '/home/teresa/marcos_tyger/Next1_10.06/'
rawData = "RarePyPulseq.2025.06.10.13.03.32.887.mat"   # [2,1,0] 
# rawData = "RarePyPulseq.2025.06.10.13.18.00.752.mat"   # [2,1,0] 
# rawData = "RarePyPulseq.2025.06.10.13.05.56.797.mat"     # [1,2,0] 
# rawData = "RarePyPulseq.2025.06.10.13.08.21.374.mat"     # [1,0,2] 
# rawData = "RarePyPulseq.2025.06.10.13.10.48.496.mat"     # [0,1,2] 
# rawData = "RarePyPulseq.2025.06.10.13.13.13.566.mat"     # [0,2,1] 
# rawData = "RarePyPulseq.2025.06.10.13.15.36.936.mat"     # [2,0,1] 
# rawData = "RAREprotocols.2025.06.10.13.20.29.085.mat"   # [2,1,0]
rawData = rawData_path + rawData

# rawData = '/home/teresa/marcos_tyger/Brain_Images/brainIR.mat'

out_field = "imgReconTyger"

## RECON CODE

# Tiempo total
start_total = time.time()

# Paso 1: fromMATtoMRD
start1 = time.time()
p1 = subprocess.Popen(
    ["python3", "RARE_recon/recon_scripts/fromMATtoMRD3D_RARE.py", "-i", rawData],
    stdout=subprocess.PIPE,
)

# Paso 2: Tyger
p2 = subprocess.Popen(
    ["tyger", "run", "exec", "-f", "RARE_recon/yml_files/neurho_brain.yml"],
    stdin=p1.stdout,
    stdout=subprocess.PIPE
)

p1.stdout.close()
p1.wait()  # <-- Esperamos a que termine p1
end1 = time.time()
# Paso 3: fromMRDtoMAT
p3 = subprocess.Popen(
    ["python3", "RARE_recon/recon_scripts/fromMRDtoMAT3D.py", "-o", rawData, "-of", out_field],
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
print(f"Duration fromMATtoMRD:   {duration1:.2f} seconds")
print(f"Duration Tyger recon:    {duration2:.2f} seconds")
print(f"Duration fromMRDtoMAT:   {duration3:.2f} seconds")
print(f"Total Time: {total_duration:.2f} seconds")


## CHECKING RESULT

rawData_pos = sio.loadmat(rawData)
print('AxesOrientation: ', rawData_pos['axesOrientation'])
print('dFov: ', rawData_pos['dfov'])
img3D_tyger = rawData_pos[out_field][0]
print('Tyger img shape: ',img3D_tyger.shape)
img3D_or = np.abs(rawData_pos['image3D'])

# ## PLOT compSlice
nSlice = 20
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.imshow(img3D_or[nSlice,:,:], cmap='gray')
ax1.axis('off')  
ax1.set_title('Original')

ax2.imshow(img3D_tyger[nSlice,:,:], cmap='gray')
ax2.axis('off')
ax2.set_title('Tyger')

plt.tight_layout()
plt.savefig('RARE_recon/compTyger.png', bbox_inches='tight', dpi=300)
# # plt.show()


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
