import scipy.io as sio

mat_data = sio.loadmat(input)
kspace = mat_data['kSpace_odd']
print(kspace.shape)