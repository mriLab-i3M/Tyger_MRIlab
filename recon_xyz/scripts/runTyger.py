import subprocess
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np 
from matplotlib.widgets import Slider

nSlice = 16
rawData = "/home/teresa/marcos_tyger/Next1_10.06/RarePyPulseq.2025.06.10.13.03.32.887.mat"   # [2,1,0] OK
# rawData = "/home/teresa/marcos_tyger/Next1_10.06/RarePyPulseq.2025.06.10.13.18.00.752.mat"   # [2,1,0] OK 120,120,28
# rawData = "/home/teresa/marcos_tyger/Next1_10.06/RarePyPulseq.2025.06.10.13.05.56.797.mat"     # [1,2,0] 120,28,120
# rawData = "/home/teresa/marcos_tyger/Next1_10.06/RarePyPulseq.2025.06.10.13.08.21.374.mat"     # [1,0,2] 28,120,120
# rawData = "/home/teresa/marcos_tyger/Next1_10.06/RarePyPulseq.2025.06.10.13.10.48.496.mat"     # [0,1,2] 28,120,120
# rawData = "/home/teresa/marcos_tyger/Next1_10.06/RarePyPulseq.2025.06.10.13.13.13.566.mat"     # [0,2,1] 120,28,120
# rawData = "/home/teresa/marcos_tyger/Next1_10.06/RarePyPulseq.2025.06.10.13.15.36.936.mat"     # [2,0,1] 120,120,28
# Tiempo total
start_total = time.time()

# Paso 1: fromMATtoMRD
start1 = time.time()
p1 = subprocess.Popen(
    ["python3", "recon_xyz/scripts/fromMATtoMRD3D_RARE_10June.py", "-i", rawData],
    stdout=subprocess.PIPE
)

# Paso 2: Tyger
p2 = subprocess.Popen(
    ["tyger", "run", "exec", "-f", "recon_xyz/scripts/stream_recon_CP_gpu_next1June11.yml"],
    stdin=p1.stdout,
    stdout=subprocess.PIPE
)

p1.stdout.close()
p1.wait()  # <-- Esperamos a que termine p1
end1 = time.time()
# Paso 3: fromMRDtoMAT
p3 = subprocess.Popen(
    ["python3", "recon_xyz/scripts/fromMRDtoMAT3D.py", "-o", rawData],
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

rawData_pos = sio.loadmat(rawData)
print('AxesOrientation: ', rawData_pos['axesOrientation'])
img3D_tyger = rawData_pos['imgReconTyger'][0]
print('Tyger img shape: ',img3D_tyger.shape)
img3D_or = np.abs(rawData_pos['image3D'])
img3D_or = np.transpose(img3D_or, [0,2,1])

# PLOT slicer
nSlice1 = img3D_or.shape[0] // 2
nSlice2 = img3D_tyger.shape[2] // 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.25) 

im1 = ax1.imshow(img3D_or[nSlice1, :, :], cmap='gray')
ax1.axis('off')
ax1.set_title('Original')

im2 = ax2.imshow(img3D_tyger[:,:, nSlice2], cmap='gray')
ax2.axis('off')
ax2.set_title('Tyger')

# Sliders OP 1 
ax_slider1 = plt.axes([0.15, 0.1, 0.3, 0.03])
slider1 = Slider(ax_slider1, '', 0, img3D_or.shape[0]-1, valinit=nSlice1, valfmt='%d')

ax_slider2 = plt.axes([0.55, 0.1, 0.3, 0.03])
slider2 = Slider(ax_slider2, '', 0, img3D_tyger.shape[2]-1, valinit=nSlice2, valfmt='%d')

def update1(val):
    idx = int(slider1.val)
    im1.set_data(img3D_or[idx, :, :])
    fig.canvas.draw_idle()

def update2(val):
    idx = int(slider2.val)
    im2.set_data(img3D_tyger[:, :, idx])
    fig.canvas.draw_idle()

slider1.on_changed(update1)
slider2.on_changed(update2)

plt.show()


# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# ax1.imshow(img3D_or[nSlice,:,:], cmap='gray')
# ax1.axis('off')  
# ax1.set_title('Original')

# ax2.imshow(img3D_tyger[:,nSlice,:], cmap='gray')
# ax2.axis('off')
# ax2.set_title('Tyger')

# plt.tight_layout()
# plt.savefig('compTyger.png', bbox_inches='tight', dpi=300)
# # plt.show()

# plt.figure(figsize = (5,8), dpi=240)
# gs1 = gridspec.GridSpec(5,8)
# gs1.update(wspace=0.020, hspace=0.020) # set the spacing between axes.

# for i in range(28):
#     if i > img3D_tyger.shape[0]:
#         break
#     ax1 = plt.subplot(gs1[i])
#     plt.axis('off')
#     ax1.set_xticklabels([])
#     ax1.set_yticklabels([])
#     ax1.set_aspect('equal')
#     imgAux = img3D_tyger[:,int(i),:]
#     ax1.imshow(imgAux,cmap='gray')

# plt.show()

# import matplotlib.pyplot as plt
# import ipywidgets as widgets
# from IPython.display import display

# # Supongo que img3D_or y img3D_tyger están definidos y tienen la misma cantidad de slices
# min_slice = 0
# max_slice = img3D_or.shape[0] - 1  # o img3D_tyger.shape[2] - 1, según corresponda

# def plot_slices(nSlice):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
#     ax1.imshow(img3D_or[nSlice,:,:], cmap='gray')
#     ax1.axis('off')
#     ax1.set_title('Original')
    
#     ax2.imshow(img3D_tyger[:,:,nSlice], cmap='gray')
#     ax2.axis('off')
#     ax2.set_title('Tyger')
    
#     plt.tight_layout()
#     # plt.show()

# slider = widgets.IntSlider(min=min_slice, max=max_slice, step=1, value=min_slice, description='Slice')

# widgets.interact(plot_slices, nSlice=slider)
# plt.show()