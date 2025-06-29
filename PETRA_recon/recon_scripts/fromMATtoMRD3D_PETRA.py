import sys
import argparse
import numpy as np
from typing import Generator
import mrd
import scipy.io as sio
import sys

def matToMRD(input, output_file):
    print('From MAT to MRD...')
    
    # OUTPUT - write .mrd
    output = sys.stdout.buffer
    if output_file is not None:
        output = output_file
        
    # INPUT - Read .mat
    mat_data = sio.loadmat(input)
    
    # Head info
    nPoints = mat_data['nPoints'][0]    # x,y,z (?)
    nPoints = [int(x) for x in nPoints]
    fov = mat_data['fov'][0]*1e1; 
    fov = fov.astype(int); fov = [int(x) for x in fov] # mm; x, y, z (?)
    
    print('nPoints',  nPoints)
    print('fov:,', fov)
    
    # Signal vector
    sampledCartesian = mat_data['kSpaceRaw']
    lenT = len(sampledCartesian)
    signal = sampledCartesian[:,3]         
    kSpace = np.reshape(signal, (1,lenT,1,1)) # Expand to MRD requisites

    # k vectors
    kTrajec = np.real(sampledCartesian[:,0:3]).astype(np.float32)    # x,y,z (?)
    
    kx = kTrajec[:,0]
    kx = np.reshape(kx, (1,lenT,1,1))

    ky = kTrajec[:,1]
    ky = np.reshape(ky, (1,lenT,1,1))
    
    kz = kTrajec[:,2]
    kz = np.reshape(kz, (1,lenT,1,1))
    
    # # Position vectors
    # # rd_pos = np.linspace(-fov_adq[0] / 2 + fov_adq[0] / (2 * nPoints[0]) , fov_adq[0] / 2 + fov_adq[0] / (2 * nPoints[0]), nPoints[0], endpoint=False)
    # # ph_pos = np.linspace(-fov_adq[1] / 2 + fov_adq[1] / (2 * nPoints[1]) , fov_adq[1] / 2 + fov_adq[1] / (2 * nPoints[1]), nPoints[1], endpoint=False)
    # # sl_pos = np.linspace(-fov_adq[2] / 2 + fov_adq[2] / (2 * nPoints[2]) , fov_adq[2] / 2 + fov_adq[2] / (2 * nPoints[2]), nPoints[2], endpoint=False)
    # x_pos = np.linspace(-fov[0] / 2 , fov[0] / 2 , nPoints[0], endpoint=False)
    # y_pos = np.linspace(-fov[1] / 2 , fov[1] / 2 , nPoints[1], endpoint=False)
    # z_pos = np.linspace(-fov[2] / 2 , fov[2] / 2 , nPoints[2], endpoint=False)
    # y_posFull, z_posFull, x_posFull = np.meshgrid(y_pos, z_pos, x_pos)
    # x_posFull = np.reshape(x_posFull, newshape=(lenT, 1))
    # y_posFull = np.reshape(y_posFull, newshape=(lenT, 1))
    # z_posFull = np.reshape(z_posFull, newshape=(lenT, 1))
    # xyz_matrix = np.concatenate((x_posFull, y_posFull, z_posFull), axis=1) # rd, ph, sl
    
    # x_esp = xyz_matrix[:,0]  
    # x_esp = np.reshape(x_esp, (1,lenT,1,1))

    # y_esp = xyz_matrix[:,1]
    # y_esp = np.reshape(y_esp, (1,lenT,1,1))

    # z_esp = xyz_matrix[:,2]   
    # z_esp = np.reshape(z_esp, (1,lenT,1,1))
    
    # OUTPUT - write .mrd
    # MRD Format
    h = mrd.Header()

    sys_info = mrd.AcquisitionSystemInformationType()
    sys_info.receiver_channels = 1
    h.acquisition_system_information = sys_info

    e = mrd.EncodingSpaceType()
    e.matrix_size = mrd.MatrixSizeType(x=nPoints[0], y=nPoints[1], z=nPoints[2])
    e.field_of_view_mm = mrd.FieldOfViewMm(x=fov[0], y=fov[1], z=fov[2])

    r = mrd.EncodingSpaceType()
    r.matrix_size = mrd.MatrixSizeType(x=nPoints[0], y=nPoints[1], z=nPoints[2])
    r.field_of_view_mm = mrd.FieldOfViewMm(x=fov[0], y=fov[1], z=fov[2])

    enc = mrd.EncodingType()
    enc.trajectory = mrd.Trajectory.RADIAL
    enc.encoded_space = e
    enc.recon_space = r
    h.encoding.append(enc)
    
    # axes_param = mrd.UserParameterStringType()
    # axes_param.name = "axesOrientation"
    # axes_param.value = ",".join(map(str, axesOrientation))  
    
    # if h.user_parameters is None:
    #     h.user_parameters = mrd.UserParametersType()
    # h.user_parameters.user_parameter_string.append(axes_param)

    def generate_data() -> Generator[mrd.StreamItem, None, None]:
        acq = mrd.Acquisition()

        acq.data.resize((1, nPoints[0]))
        acq.trajectory.resize((7, nPoints[0]))
        acq.center_sample = round(nPoints[0] / 2)

        for s in range(lenT):
            acq.idx.kspace_encode_step_1 = 1
            acq.idx.kspace_encode_step_2 = s
            acq.idx.slice = s
            acq.idx.repetition = 0
            acq.data[:] = kSpace[:, s, :, :]
            acq.trajectory[0,:] = kx[:, s, :, :]
            acq.trajectory[1,:] = ky[:, s, :, :]
            acq.trajectory[2,:] = kz[:, s, :, :]
            
            yield mrd.StreamItem.Acquisition(acq)

    with mrd.BinaryMrdWriter(output) as w:
        w.write_header(h)
        w.write_data(generate_data())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert mat to MRD")
    parser.add_argument('-i', '--input', type=str, required=False, help="Input file path")
    parser.add_argument('-o', '--output', type=str, required=False, help="Output MRD file")

    parser.set_defaults(
        input = '/home/tyger/tyger_repo_may/PETRA_Phys1/PETRA.2024.12.19.19.38.08.208.mat',
        output= '/home/tyger/tyger_repo_may/Tyger_MRIlab/CP_ARTPK_ART/recon_scripts/testPETRA.bin',
    )
    
    args = parser.parse_args()
    matToMRD(args.input, args.output)