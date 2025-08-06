import sys
import argparse
import numpy as np
from typing import BinaryIO, Iterable, Union
import mrd
import cupy as cp 
import ast

def acquisition_reader(input: Iterable[mrd.StreamItem]) -> Iterable[mrd.Acquisition]:
    for item in input:
        if not isinstance(item, mrd.StreamItem.Acquisition):
            # Skip non-acquisition items
            continue
        if item.value.flags & mrd.AcquisitionFlags.IS_NOISE_MEASUREMENT:
            # Currently ignoring noise scans
            continue
        yield item.value

def stream_item_sink(input: Iterable[Union[mrd.Acquisition, mrd.Image[np.float32]]]) -> Iterable[mrd.StreamItem]:
    for item in input:
        if isinstance(item, mrd.Acquisition):
            yield mrd.StreamItem.Acquisition(item)
        elif isinstance(item, mrd.Image) and item.data.dtype == np.float32:
            yield mrd.StreamItem.ImageFloat(item)
        else:
            raise ValueError("Unknown item type")

def mrdRecon(reconMode: str, sign:str, BoFit:str,
              head: mrd.Header, input: Iterable[mrd.Acquisition]) -> Iterable[mrd.Image[np.float32]]:
    
    ## HEAD 
    
    # Custom parameters
    vecSign = ast.literal_eval(args.sign)
    try:
        cp_batchsize = vecSign[8]
    except:
        cp_batchsize = 1000
        
    if head.user_parameters and head.user_parameters.user_parameter_double:
        for param in head.user_parameters.user_parameter_double:
            if param.name == "readout_gradient_intensity":
                rdGradAmplitude = param.value
                break
    
    if head.user_parameters and head.user_parameters.user_parameter_string:
        for param in head.user_parameters.user_parameter_string:
            if param.name == "axesOrientation":
                axesOrientation = list(map(int, param.value.split(',')))
            if param.name == "dfov":
                dfov = list(map(float, param.value.split(',')))
                
    axesOrientation = np.array(axesOrientation) # rd, ph, sl
    rd_dir = np.array([1,0,0]) # rd, ph, sl
    inverse_axesOrientation = np.argsort(axesOrientation) # x,y,z
    rd_dir = rd_dir[inverse_axesOrientation] # x,y,z
    dfov = np.array(dfov) # x, y, z
    
    print('rd_dir', rd_dir)
    print('axesOrientation:', axesOrientation)
    print('dfov:', dfov)
    
    enc = head.encoding[0]

    # Matrix size
    if enc.encoded_space and enc.recon_space and enc.encoded_space.matrix_size and enc.recon_space.matrix_size:
        eNx = enc.encoded_space.matrix_size.x
        eNy = enc.encoded_space.matrix_size.y
        eNz = enc.encoded_space.matrix_size.z
        rNx = enc.recon_space.matrix_size.x
        rNy = enc.recon_space.matrix_size.y
        rNz = enc.recon_space.matrix_size.z
        nPoints_sig = np.array([eNx,eNy,eNz]) # x,y,z
        nPoints_sig = nPoints_sig[axesOrientation] # rd, ph, sl
        nPoints_sig = nPoints_sig[[2,1,0]] # sl, ph, rd
    else:
        raise Exception('Required encoding information not found in header')

    # Field of view
    if enc.recon_space and enc.recon_space.field_of_view_mm:
        rFOVx = enc.recon_space.field_of_view_mm.x*1e-3
        rFOVy = enc.recon_space.field_of_view_mm.y*1e-3
        rFOVz = enc.recon_space.field_of_view_mm.z*1e-3 if enc.recon_space.field_of_view_mm.z else 1
    else:
        raise Exception('Required field of view information not found in header')

    # Signal, ks, positions
    kSpace_buffer = None
    kx_buffer = None
    ky_buffer = None
    kz_buffer = None
    times_buffer = None
    x_buffer = None
    y_buffer = None
    z_buffer = None
    bo_buffer = None

    def produce_image(img: np.ndarray) -> Iterable[mrd.Image[np.float32]]:
        mrd_image = mrd.Image[np.float32](image_type=mrd.ImageType.MAGNITUDE, data=img)
        yield mrd_image
    
    kSpace_buffer = []
    kx_buffer = []
    ky_buffer = []
    kz_buffer = []
    times_buffer = []
    x_buffer = []
    y_buffer = []
    z_buffer = []
    bo_buffer = []

    for acq in input:

        k1 = acq.idx.kspace_encode_step_1 if acq.idx.kspace_encode_step_1 is not None else 0
        k2 = acq.idx.kspace_encode_step_2 if acq.idx.kspace_encode_step_2 is not None else 0

        # # kSpace_buffer = np.concatenate((kSpace_buffer, acq.data[0]), axis = 0) # Much slower!
        kSpace_buffer.append(acq.data[0])
        kx_buffer.append(acq.trajectory[0,:])
        ky_buffer.append(acq.trajectory[1,:])
        kz_buffer.append(acq.trajectory[2,:])
        times_buffer.append(acq.trajectory[3,:])
        x_buffer.append(acq.trajectory[4,:])
        y_buffer.append(acq.trajectory[5,:])
        z_buffer.append(acq.trajectory[6,:])
        bo_buffer.append(acq.trajectory[7,:])

    kSpace_buffer = np.reshape(kSpace_buffer, -1, order='C')
    kx_buffer = np.reshape(kx_buffer, -1, order='C')
    ky_buffer = np.reshape(ky_buffer, -1, order='C')
    kz_buffer= np.reshape(kz_buffer, -1, order='C')
    times_buffer = np.reshape(times_buffer, -1, order='C')
    x_buffer = np.reshape(x_buffer, -1, order='C')*1e-3   # Converting to m
    y_buffer = np.reshape(y_buffer, -1, order='C')*1e-3   # Converting to m
    z_buffer = np.reshape(z_buffer, -1, order='C')*1e-3    # Converting to m
    bo_buffer = np.reshape(bo_buffer, -1, order='C')
    
    def pythonfft(kSpace):        
        img = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kSpace[:, :, :])))
        img = np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
        img = np.abs(img).astype(np.float32)
        return img
    
    def pythonART():
        gammabar = 42.57747892*1e6

        signal = kSpace_buffer
        kx = 2*np.pi*kx_buffer
        ky = 2*np.pi*ky_buffer
        kz = 2*np.pi*kz_buffer
        x = x_buffer
        y = y_buffer
        z = z_buffer
        dBo = bo_buffer
        
        # boFit = eval(f"lambda x, y, z: {BoFit}")
        # # dBo = boFit(vecSign[0]*x, vecSign[1]*y, vecSign[2]*z)
        # dBo = boFit(vecSign[0]*x-vecSign[5]*dfov[0], vecSign[1]*y-vecSign[6]*dfov[1], vecSign[2]*z-vecSign[7]*dfov[2])
        
        rho = np.reshape(np.zeros((rNx*rNy*rNz), dtype=complex), (-1, 1))
        rho = rho[:,0]
        lbda = 1/float(1)
        n_iter = int(1)
        index = np.arange(len(rho))

        kx_gpu = cp.asarray(kx)
        ky_gpu = cp.asarray(ky)
        kz_gpu = cp.asarray(kz)
        sx_gpu = cp.asarray(x)
        sy_gpu = cp.asarray(y)
        sz_gpu = cp.asarray(z)
        signal_gpu = cp.asarray(signal)
        index_gpu = cp.asarray(index)
        rho_gpu = cp.asarray(rho)
        dBo_gpu = cp.asarray(dBo)
        timeVec = cp.asarray(times_buffer)

        def artPK(kx, ky, kz, x, y, z, s, rho, lbda, n_iter, index, times, dBo):
                    n = 0
                    n_samples = len(s)
                    m = 0
                    for iteration in range(n_iter):
                        # cp.random.shuffle(index)
                        for jj in range(n_samples):
                            ii = index[jj]
                            x0 = cp.exp(vecSign[4]*-1j *  (kx[ii] * x + ky[ii] * y + kz[ii] * z + 2 * np.pi * gammabar * times[ii]*dBo))
                            x1 =  s[ii]-(x0.T @ rho)
                            # x2 = x1 * cp.conj(x0) / (cp.conj(x0.T) @ x0)
                            x2 = x1 * cp.conj(x0) / (rNx*rNy*rNz)
                            d_rho = lbda * x2
                            rho += d_rho
                            n += 1
                            if n / n_samples > 0.01:
                                m += 1
                                n = 0
                                print("ART iteration %i: %i %%" % (iteration + 1, m))

                    return rho
        
        def art(kx, ky, kz, x, y, z, s, rho, lbda, n_iter, index, times, dBo):
                    n = 0
                    n_samples = len(s)
                    m = 0
                    for iteration in range(n_iter):
                        # cp.random.shuffle(index)
                        for jj in range(n_samples):
                            ii = index[jj]
                            x0 = cp.exp(vecSign[4]*-1j *  (kx[ii] * x + ky[ii] * y + kz[ii] * z))
                            x1 =  s[ii]-(x0.T @ rho)
                            # x2 = x1 * cp.conj(x0) / (cp.conj(x0.T) @ x0)
                            x2 = x1 * cp.conj(x0) / (rNx*rNy*rNz)
                            d_rho = lbda * x2
                            rho += d_rho
                            n += 1
                            if n / n_samples > 0.01:
                                m += 1
                                n = 0
                                print("ART iteration %i: %i %%" % (iteration + 1, m))

                    return rho

        if reconMode == 'artpk':
            imgART= artPK(kx_gpu,ky_gpu,kz_gpu,sx_gpu,sy_gpu,sz_gpu,signal_gpu, rho_gpu,lbda, n_iter, index_gpu,timeVec, dBo_gpu)
        elif reconMode == 'art':
            imgART= art(kx_gpu,ky_gpu,kz_gpu,sx_gpu,sy_gpu,sz_gpu,signal_gpu, rho_gpu,lbda, n_iter, index_gpu,timeVec, dBo_gpu)
        img = cp.asnumpy(imgART)
        img = np.reshape(img, (1,nPoints_sig[0],nPoints_sig[1],nPoints_sig[2]))
        img = np.abs(img).astype(np.float32)
        return img
    
    def pythonCP():

        signal = kSpace_buffer
        kx = 2*np.pi*kx_buffer
        ky = 2*np.pi*ky_buffer
        kz = 2*np.pi*kz_buffer

        x = x_buffer
        y = y_buffer
        z = z_buffer
        dBo = bo_buffer

        # boFit = eval(f"lambda x, y, z: {BoFit}")
        # dBo = boFit(vecSign[0]*x-vecSign[5]*dfov[0], vecSign[1]*y-vecSign[6]*dfov[1], vecSign[2]*z-vecSign[7]*dfov[2])
        dBo = dBo/rdGradAmplitude
        
        rho = np.reshape(np.zeros((rNx*rNy*rNz), dtype=complex), (-1, 1))
        rho = rho[:,0]

        kx_gpu = cp.asarray(kx)
        ky_gpu = cp.asarray(ky)
        kz_gpu = cp.asarray(kz)
        sx_gpu = cp.asarray(x)
        sy_gpu = cp.asarray(y)
        sz_gpu = cp.asarray(z)
        signal_gpu = cp.asarray(signal)
        rho_gpu = cp.asarray(rho)
        dBo_gpu = cp.asarray(dBo)

        def conjugatePhase(kx, ky, kz, x, y, z, s, rho, dBo):
            print('Running conjugate phase...')
            if cp_batchsize == 0:
                phase = cp.exp(vecSign[3]*1j * (cp.outer(kx, x) + cp.outer(ky, y) + cp.outer(kz, z)))
                rho = cp.dot(s, phase)
            else:
                for i in range(0, len(x), cp_batchsize):
                    # if int(i/len(x)*100) != int((i-1)/len(x)*100):
                        # print(int(i/len(x)*100), ' %')
                    x_batch = x[i:i+cp_batchsize] + dBo[i:i+cp_batchsize]*rd_dir[0]
                    y_batch = y[i:i+cp_batchsize] + dBo[i:i+cp_batchsize]*rd_dir[1]
                    z_batch = z[i:i+cp_batchsize] + dBo[i:i+cp_batchsize]*rd_dir[2]

                    phase_batch = cp.exp(vecSign[3]*1j * (cp.outer(kx, x_batch) + cp.outer(ky, y_batch) + cp.outer(kz, z_batch)))
                    rho[i:i+cp_batchsize] = cp.dot(s, phase_batch)
            
            return rho

        imgCP= conjugatePhase(kx_gpu,ky_gpu,kz_gpu,sx_gpu,sy_gpu,sz_gpu,signal_gpu, rho_gpu, dBo_gpu)
        img = cp.asnumpy(imgCP)
        img = np.reshape(img, (1,nPoints_sig[0],nPoints_sig[1],nPoints_sig[2]))
        img = np.abs(img).astype(np.float32)
        return img
    
    if reconMode == 'fft':
        kSpace = np.reshape(kSpace_buffer, [nPoints_sig[0],nPoints_sig[1],nPoints_sig[2]])
        imgRecon = pythonfft(kSpace)
    elif reconMode == 'art' or reconMode == 'artpk':
        imgRecon = pythonART()
    elif reconMode == 'cp':
        imgRecon = pythonCP()
        
    yield from produce_image(imgRecon)


def reconstruct_mrd_stream(reconMode: str, sign:str, BoFit:str,
                            input: BinaryIO, output: BinaryIO):
    with mrd.BinaryMrdReader(input) as reader:
        with mrd.BinaryMrdWriter(output) as writer:
            head = reader.read_header()
            if head is None:
                raise Exception("Could not read header")
            writer.write_header(head)
            writer.write_data(
                stream_item_sink(
                    mrdRecon(reconMode, sign, BoFit,
                              head, acquisition_reader(reader.read_data()))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstructs an MRD stream")
    parser.add_argument('-i', '--input', type=str, required=False, help="Input file, defaults to stdin")
    parser.add_argument('-o', '--output', type=str, required=False, help="Output file, defaults to stdout")
    parser.add_argument('-r', '--recon', type=str, required=False, help="Reconstruction mode (fft, cp, art, artpk)")
    parser.add_argument('-s', '--sign', type=str, required=False, help="Signs and others for code generalization [xsignG,ysignG,zsignG,cpPhase,artPhase,dfovx_sign,dfovy_sign,dfovz_sign, cp_batchsize]")
    parser.add_argument('-BoFit', '--BoFit', type=str, default = False, required=False, help="Bo Fit string")
    
    # parser.set_defaults(
    #     input = '/home/tyger/tyger_repo_may/next1SPDS/inputPP.bin', 
    #     output = '/home/tyger/tyger_repo_may/next1SPDS/reconPP.bin',
    #     recon = 'cp', 
    #     sign = "[-1,-1,-1,1,1,1,1,1,1000]",
    #     BoFit = """0*x+0*y+0*z"""
    #     )
    
    args = parser.parse_args()

    input = open(args.input, "rb") if args.input is not None else sys.stdin.buffer
    output = open(args.output, "wb") if args.output is not None else sys.stdout.buffer

    reconstruct_mrd_stream(args.recon, args.sign, args.BoFit,
                            input, output)
