import os
import time
import numpy as np
try:
    import cupy as cp
    print("\nGPU will be used for ART reconstruction")
except ImportError:
    pass
import scipy.io as sio
import sys

nameFileOut = '-test.mat'
mat_data=sio.loadmat('test14.mat')
# def gradX(x,y,z):
#     return x + 0.2704798742054992*x**2 - 15.19822559023198*x**3 - 9.864356507012188*x**4 - 741.231671000555*x**5 + 1510.1360805750394*x**6 - 0.036643686533195476*y - 0.12467810809731651*x*y - 1.5048774859737877*x**2*y - 11.31075115752636*x**3*y - 152.93337594969987*x**4*y + 1093.8116190175774*x**5*y + 0.437970503897502*y**2 + 3.7012411170544604*x*y**2 + 60.07620539930176*x**2*y**2 - 2360.8955280827413*x**3*y**2 - 3885.2772769109692*x**4*y**2 + 0.18311860924976392*y**3 + 51.1211617913179*x*y**3 - 138.4155380048766*x**2*y**3 - 2174.7741717973017*x**3*y**3 - 119.8437961094866*y**4 - 730.3061371007344*x*y**4 - 21345.41780150154*x**2*y**4 + 208.1371119081829*y**5 - 4363.380424038656*x*y**5 + 8948.106389601735*y**6 - 0.13149311795907026*z - 0.04671135452461596*x*z - 1.2654625557777042*x**2*z - 3.4734824300387*x**3*z + 818.6797479261425*x**4*z + 330.4034151854348*x**5*z - 0.17187271566752713*y*z - 1.1867188151130812*x*y*z - 10.580795343670951*x**2*y*z - 52.70914578888994*x**3*y*z - 1007.3144338747185*x**4*y*z - 1.17346321889223*y**2*z + 31.71910668871547*x*y**2*z - 838.9415817067832*x**2*y**2*z - 4609.299756658872*x**3*y**2*z + 40.04249456271484*y**3*z - 328.1713552323458*x*y**3*z + 7493.305404135987*x**2*y**3*z - 187.05745330978164*y**4*z - 2008.148635287129*x*y**4*z - 5728.078545999211*y**5*z - 0.05956369668015409*z**2 + 38.20702587781121*x*z**2 - 71.05055832337898*x**2*z**2 + 4949.555680881201*x**3*z**2 + 16754.987392435545*x**4*z**2 + 2.6084440515046903*y*z**2 + 19.107008511977742*x*y*z**2 - 385.1741927368656*x**2*y*z**2 - 3581.955748890018*x**3*y*z**2 - 68.56376689866873*y**2*z**2 - 105.0246914257714*x*y**2*z**2 - 12280.500485890063*x**2*y**2*z**2 - 255.4777453670214*y**3*z**2 - 2352.3061867803904*x*y**3*z**2 - 1725.0183135754958*y**4*z**2 - 4.098501482653433*z**3 + 45.24433803107687*x*z**3 - 254.7062974864712*x**2*z**3 - 5713.7987887771205*x**3*z**3 + 54.24322753850124*y*z**3 + 69.89542141998966*x*y*z**3 - 3776.3354194368335*x**2*y*z**3 + 777.7710677761032*y**2*z**3 - 4438.255227159979*x*y**2*z**3 - 2730.9862525867384*y**3*z**3 - 57.65473751329662*z**4 - 1858.669700658146*x*z**4 - 12019.681652893925*x**2*z**4 - 57.8783309853123*y*z**4 + 622.455887572102*x*y*z**4 + 24083.589823869886*y**2*z**4 + 511.3114632945956*z**5 - 2385.319160180739*x*z**5 - 4483.043780673903*y*z**5 + 6800.744504497688*z**6

def gradX(x,y,z):
    return (
            x + 0.2704798742054992 * x**2 - 15.19822559023198 * x**3
            - 9.864356507012188 * x**4 - 741.231671000555 * x**5 + 1510.1360805750394 * x**6
            - 0.036643686533195476 * y - 0.12467810809731651 * x * y - 1.5048774859737877 * x**2 * y
            - 11.31075115752636 * x**3 * y - 152.93337594969987 * x**4 * y + 1093.8116190175774 * x**5 * y
            + 0.437970503897502 * y**2 + 3.7012411170544604 * x * y**2 + 60.07620539930176 * x**2 * y**2
            - 2360.8955280827413 * x**3 * y**2 - 3885.2772769109692 * x**4 * y**2
            + 0.18311860924976392 * y**3 + 51.1211617913179 * x * y**3 - 138.4155380048766 * x**2 * y**3
            - 2174.7741717973017 * x**3 * y**3 - 119.8437961094866 * y**4 - 730.3061371007344 * x * y**4
            - 21345.41780150154 * x**2 * y**4 + 208.1371119081829 * y**5 - 4363.380424038656 * x * y**5
            + 8948.106389601735 * y**6 - 0.13149311795907026 * z - 0.04671135452461596 * x * z
            - 1.2654625557777042 * x**2 * z - 3.4734824300387 * x**3 * z + 818.6797479261425 * x**4 * z
            + 330.4034151854348 * x**5 * z - 0.17187271566752713 * y * z - 1.1867188151130812 * x * y * z
            - 10.580795343670951 * x**2 * y * z - 52.70914578888994 * x**3 * y * z - 1007.3144338747185 * x**4 * y * z
            - 1.17346321889223 * y**2 * z + 31.71910668871547 * x * y**2 * z - 838.9415817067832 * x**2 * y**2 * z
            - 4609.299756658872 * x**3 * y**2 * z + 40.04249456271484 * y**3 * z - 328.1713552323458 * x * y**3 * z
            + 7493.305404135987 * x**2 * y**3 * z - 187.05745330978164 * y**4 * z - 2008.148635287129 * x * y**4 * z
            - 5728.078545999211 * y**5 * z - 0.05956369668015409 * z**2 + 38.20702587781121 * x * z**2
            - 71.05055832337898 * x**2 * z**2 + 4949.555680881201 * x**3 * z**2 + 16754.987392435545 * x**4 * z**2
            + 2.6084440515046903 * y * z**2 + 19.107008511977742 * x * y * z**2 - 385.1741927368656 * x**2 * y * z**2
            - 3581.955748890018 * x**3 * y * z**2 - 68.56376689866873 * y**2 * z**2 - 105.0246914257714 * x * y**2 * z**2
            - 12280.500485890063 * x**2 * y**2 * z**2 - 255.4777453670214 * y**3 * z**2 - 2352.3061867803904 * x * y**3 * z**2
            - 1725.0183135754958 * y**4 * z**2 - 4.098501482653433 * z**3 + 45.24433803107687 * x * z**3
            - 254.7062974864712 * x**2 * z**3 - 5713.7987887771205 * x**3 * z**3 + 54.24322753850124 * y * z**3
            + 69.89542141998966 * x * y * z**3 - 3776.3354194368335 * x**2 * y * z**3 + 777.7710677761032 * y**2 * z**3
            - 4438.255227159979 * x * y**2 * z**3 - 2730.9862525867384 * y**3 * z**3 - 57.65473751329662 * z**4
            - 1858.669700658146 * x * z**4 - 12019.681652893925 * x**2 * z**4 - 57.8783309853123 * y * z**4
            + 622.455887572102 * x * y * z**4 + 24083.589823869886 * y**2 * z**4 + 511.3114632945956 * z**5
            - 2385.319160180739 * x * z**5 - 4483.043780673903 * y * z**5 + 6800.744504497688 * z**6
        )
# def gradX(x,y,z):
#     return x**2 - y

fov = np.reshape(mat_data['FOV'], -1) * 1e-2
offset = 0
offset = offset*1e-2
nPoints = np.reshape(mat_data['nPoints'], -1)
k = mat_data['kSpace3D'][:, 0:3]
s = mat_data['kSpace3D'][:, 3]
print(k.shape)
print(s.shape)

# Points where rho will be estimated
x = np.linspace(-0.2/ 2, 0.2/ 2, nPoints[0])
y = np.linspace(-0.16 / 2, 0.16 / 2, nPoints[1])
z = np.linspace(-0.16 / 2, 0.16 / 2, nPoints[2])
y, z, x = np.meshgrid(y, z, x)
x = np.reshape(x, (-1, 1))
y = np.reshape(y, (-1, 1))
z = np.reshape(z, (-1, 1))
print(x.dtype)
if 'cp' in globals():
    GX = cp.array(gradX(-x+offset,y,z))
else:
    GX = np.array(gradX(-x+offset,y,z))
print(GX.dtype)
# k-points
kx = np.reshape(k[:, 0].real, (-1, 1))
ky = np.reshape(k[:, 1].real, (-1, 1))
kz = np.reshape(k[:, 2].real, (-1, 1))

# Iterative process
lbda = 1/float(1)
n_iter = int(1)
index = np.arange(len(s))

# def iterative_process_gpu(kx, ky, kz, x, y, z, s, rho, lbda, n_iter, index):
#     n = 0
#     n_samples = len(s)
#     m = 0
#     for iteration in range(n_iter):
#         # cp.random.shuffle(index)
#         for jj in range(n_samples):
#             ii = index[jj]
#             x0 = cp.exp(1j * 2 * cp.pi * (kx[ii] * GX + ky[ii] * y + kz[ii] * z))
#             x1 = (x0.T @ rho) - s[ii]
#             # x2 = x1 * cp.conj(x0) / (cp.conj(x0.T) @ x0)
#             x2 = x1 * cp.conj(x0) / (nPoints[0]*nPoints[1]*nPoints[2])
#             d_rho = lbda * x2
#             rho -= d_rho
#             n += 1
#             if n / n_samples > 0.01:
#                 m += 1
#                 n = 0
#                 print("ART iteration %i: %i %%" % (iteration + 1, m))

#     return rho

def iterative_process_gpu(kx, ky, kz, x, y, z, s, rho, lbda, n_iter, index):
    n = 0
    n_samples = len(s)
    m = 0
    for iteration in range(n_iter):
        # cp.random.shuffle(index)
        for jj in range(n_samples):
            ii = index[jj]
            x0 = cp.exp(1j * 2 * cp.pi * (kx[ii] * GX + ky[ii] * y + kz[ii] * z))
            x1 =  s[ii]-(x0.T @ rho)
            # x2 = x1 * cp.conj(x0) / (cp.conj(x0.T) @ x0)
            x2 = x1 * cp.conj(x0) / (nPoints[0]*nPoints[1]*nPoints[2])
            d_rho = lbda * x2
            rho += d_rho
            n += 1
            if n / n_samples > 0.01:
                m += 1
                n = 0
                print("ART iteration %i: %i %%" % (iteration + 1, m))

    return rho

def iterative_process_cpu(kx, ky, kz, x, y, z, s, rho, lbda, n_iter, index):
    n = 0
    n_samples = len(s)
    m = 0
    for iteration in range(n_iter):
        np.random.shuffle(index)
        for jj in range(n_samples):
            ii = index[jj]
            x0 = np.exp(-1j * 2 * np.pi * (kx[ii] * x + ky[ii] * y + kz[ii] * z))
            x1 = (x0.T @ rho) - s[ii]
            x2 = x1 * np.conj(x0) / (np.conj(x0.T) @ x0)
            d_rho = lbda * x2
            rho -= d_rho
            n += 1
            if n / n_samples > 0.01:
                m += 1
                n = 0
                print("ART iteration %i: %i %%" % (iteration + 1, m))

    return rho

# Launch the GPU function
rho = np.reshape(np.zeros((nPoints[0] * nPoints[1] * nPoints[2]), dtype=complex), (-1, 1))
start = time.time()
if 'cp' in globals():
    print('Executing ART in GPU...')

    # Transfer numpy arrays to cupy arrays
    kx_gpu = cp.asarray(kx)
    ky_gpu = cp.asarray(ky)
    kz_gpu = cp.asarray(kz)
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)
    z_gpu = cp.asarray(z)
    s_gpu = cp.asarray(s)
    index = cp.asarray(index)
    rho_gpu = cp.asarray(rho)

    # Execute ART
    rho_gpu = iterative_process_gpu(kx_gpu, ky_gpu, kz_gpu, x_gpu, y_gpu, z_gpu, s_gpu, rho_gpu, lbda, n_iter,
                                    index)
    rho = cp.asnumpy(rho_gpu)
else:
    print('Executing ART in CPU...')

    rho = iterative_process_cpu(kx, ky, kz, x, y, z, s, rho, lbda, n_iter,
                                    index)
end = time.time()
print("Reconstruction time = %0.1f s" % (end - start))

rho = np.reshape(rho, nPoints[-1::-1])

# Create dict
new_dict = {
    'rho': rho
}
sio.savemat(nameFileOut, new_dict)