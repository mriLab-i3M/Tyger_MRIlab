import subprocess
import time

rawData = "/home/teresa/marcos_tyger/Brain_Images/brainIR.mat"

# Tiempo total
start_total = time.time()

# Paso 1: fromMATtoMRD
start1 = time.time()
p1 = subprocess.Popen(
    ["python3", "recon_xyz/scripts/fromMATtoMRD3D_RARE.py", "-i", rawData],
    stdout=subprocess.PIPE
)

# Paso 2: Tyger
p2 = subprocess.Popen(
    ["tyger", "run", "exec", "-f", "recon_xyz/scripts/stream_recon_FFT_gpu.yml"],
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