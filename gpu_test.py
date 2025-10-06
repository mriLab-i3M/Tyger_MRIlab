import cupy as cp

# Mostrar información de la GPU detectada
print("Número de GPUs detectadas:", cp.cuda.runtime.getDeviceCount())
print("Nombre de la GPU actual:", cp.cuda.runtime.getDeviceProperties(0)['name'].decode())

# Crear dos arrays en la GPU y hacer una operación
x = cp.arange(10**6, dtype=cp.float32)
y = cp.arange(10**6, dtype=cp.float32)

z = x * y  # esto ocurre en la GPU

print("Resultado en GPU:", z[:5])  # mostrar primeros valores
print("Ubicación del array z:", z.device)  # debería decir 'CUDA:0'