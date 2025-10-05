import time
import subprocess
import io
import scipy.io as sio
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import os
import sys

rawData_path = '/home/teresa/Documentos/Next1/brain_bw_sweep/'
rawData = 'RareDoubleImage.2025.09.12.17.16.49.417.mat'
rawData = rawData_path + rawData
out_field = "tygerCP_2"

rawData_pos = sio.loadmat(rawData)
img3D_tyger = rawData_pos[out_field][0]
img3D_or = np.abs(rawData_pos['image3D'])
