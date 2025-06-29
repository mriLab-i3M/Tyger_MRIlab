import sys
import argparse
import numpy as np
from typing import BinaryIO, Iterable, Union
import mrd
import matplotlib.pyplot as plt
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

def mrdRecon(sign:str, reconMode: str, artMode:str, BoFit:str,
              head: mrd.Header, input: Iterable[mrd.Acquisition]) -> Iterable[mrd.Image[np.float32]]:
    
    vecSign = ast.literal_eval(args.sign)

    if head.user_parameters and head.user_parameters.user_parameter_double:
        for param in head.user_parameters.user_parameter_double:
            if param.name == "readout_gradient_intensity":
                rdGradAmplitude = param.value
                break
    
    if head.user_parameters and head.user_parameters.user_parameter_string:
        for param in head.user_parameters.user_parameter_string:
            if param.name == "axesOrientation":
                axesOrientation = list(map(int, param.value.split(',')))
                print(axesOrientation)
            if param.name == "dfov":
                dfov = list(map(float, param.value.split(',')))
                print(dfov)
    axesOrientation = np.array(axesOrientation) # rd, ph, sl
    rd_dir = np.array([1,0,0]) # rd, ph, sl
    inverse_axesOrientation = np.argsort(axesOrientation) # x,y,z
    rd_dir = rd_dir[inverse_axesOrientation] # x,y,z
    print('rd_dir', rd_dir)
    print('axesOrientation:', axesOrientation)
    dfov = np.array(dfov) # x, y, z
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

    kSpace_buffer = None
    kx_buffer = None
    ky_buffer = None
    kz_buffer = None
    times_buffer = None
    # axesOrientation = None
    x_buffer = None
    y_buffer = None
    z_buffer = None

    def produce_image(img: np.ndarray) -> Iterable[mrd.Image[np.float32]]:
        mrd_image = mrd.Image[np.float32](image_type=mrd.ImageType.MAGNITUDE, data=img)
        yield mrd_image
    
    # kSpace_buffer = np.empty((0,))
    kSpace_buffer = []
    kx_buffer = []
    ky_buffer = []
    kz_buffer = []
    times_buffer = []
    x_buffer = []
    y_buffer = []
    z_buffer = []

    for acq in input:
        # if axesOrientation == None:
        #     axesOrientation = acq.channel_order

        k1 = acq.idx.kspace_encode_step_1 if acq.idx.kspace_encode_step_1 is not None else 0
        k2 = acq.idx.kspace_encode_step_2 if acq.idx.kspace_encode_step_2 is not None else 0

        # # kSpace_buffer = np.concatenate((kSpace_buffer, acq.data[0]), axis = 0) # Mucho más lento!
        kSpace_buffer.append(acq.data[0])
        kx_buffer.append(acq.trajectory[0,:])
        ky_buffer.append(acq.trajectory[1,:])
        kz_buffer.append(acq.trajectory[2,:])
        times_buffer.append(acq.trajectory[3,:])
        x_buffer.append(acq.trajectory[4,:])
        y_buffer.append(acq.trajectory[5,:])
        z_buffer.append(acq.trajectory[6,:])

    kSpace_buffer = np.reshape(kSpace_buffer, -1, order='C')
    kx_buffer = np.reshape(kx_buffer, -1, order='C')
    ky_buffer = np.reshape(ky_buffer, -1, order='C')
    kz_buffer= np.reshape(kz_buffer, -1, order='C')
    times_buffer = np.reshape(times_buffer, -1, order='C')
    x_buffer = np.reshape(x_buffer, -1, order='C')*1e-3
    y_buffer = np.reshape(y_buffer, -1, order='C')*1e-3
    z_buffer= np.reshape(z_buffer, -1, order='C')*1e-3
    np.save('x_buffer.npy', x_buffer)
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
        
        boFit = eval(f"lambda x, y, z: {BoFit}")
        # dBo = boFit(vecSign[0]*x, vecSign[1]*y, vecSign[2]*z)
        dBo = boFit(vecSign[0]*x-vecSign[5]*dfov[0], vecSign[1]*y-vecSign[6]*dfov[1], vecSign[2]*z-vecSign[7]*dfov[2])
        
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
                                # print("ART iteration %i: %i %%" % (iteration + 1, m))

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
                                # print("ART iteration %i: %i %%" % (iteration + 1, m))

                    return rho

        if artMode == 'artPK':
            imgART= artPK(kx_gpu,ky_gpu,kz_gpu,sx_gpu,sy_gpu,sz_gpu,signal_gpu, rho_gpu,lbda, n_iter, index_gpu,timeVec, dBo_gpu)
        elif artMode == 'art':
            imgART= art(kx_gpu,ky_gpu,kz_gpu,sx_gpu,sy_gpu,sz_gpu,signal_gpu, rho_gpu,lbda, n_iter, index_gpu,timeVec, dBo_gpu)
        img = cp.asnumpy(imgART)
        img = np.reshape(img, (1,nPoints_sig[0],nPoints_sig[1],nPoints_sig[2]))
        img = np.abs(img).astype(np.float32)
        # print(img.shape)
        return img
    
    def pythonCP():

        signal = kSpace_buffer
        kx = 2*np.pi*kx_buffer
        ky = 2*np.pi*ky_buffer
        kz = 2*np.pi*kz_buffer

        x = x_buffer
        y = y_buffer
        z = z_buffer

        boFit = eval(f"lambda x, y, z: {BoFit}")
        dBo = boFit(vecSign[0]*x-vecSign[5]*dfov[0], vecSign[1]*y-vecSign[6]*dfov[1], vecSign[2]*z-vecSign[7]*dfov[2])
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
            batch_size = 1000
            # phase = cp.exp(-1j * (cp.outer(kx, x) + cp.outer(ky, y) + cp.outer(kz, z)))
            # rho = cp.dot(s, phase)
            
            for i in range(0, len(x), batch_size):
                # if int(i/len(x)*100) != int((i-1)/len(x)*100):
                    # print(int(i/len(x)*100), ' %')
                x_batch = x[i:i+batch_size] + dBo[i:i+batch_size]*rd_dir[0]
                y_batch = y[i:i+batch_size] + dBo[i:i+batch_size]*rd_dir[1]
                z_batch = z[i:i+batch_size] + dBo[i:i+batch_size]*rd_dir[2]

                phase_batch = cp.exp(vecSign[3]*1j * (cp.outer(kx, x_batch) + cp.outer(ky, y_batch) + cp.outer(kz, z_batch)))
                rho[i:i+batch_size] = cp.dot(s, phase_batch)
            
            return rho

        imgCP= conjugatePhase(kx_gpu,ky_gpu,kz_gpu,sx_gpu,sy_gpu,sz_gpu,signal_gpu, rho_gpu, dBo_gpu)
        img = cp.asnumpy(imgCP)
        img = np.reshape(img, (1,nPoints_sig[0],nPoints_sig[1],nPoints_sig[2]))
        img = np.abs(img).astype(np.float32)
        return img
    
    if reconMode == 'pythonfft':
        kSpace = np.reshape(kSpace_buffer, [eNx,eNy,eNz])
        kSpace = np.transpose(kSpace, axesOrientation)
        imgRecon = pythonfft(kSpace)
    elif reconMode == 'art':
        imgRecon = pythonART()
    elif reconMode == 'cp':
        imgRecon = pythonCP()
        
    yield from produce_image(imgRecon)


def reconstruct_mrd_stream(sign:str, reconMode: str, artMode:str, BoFit:str,
                            input: BinaryIO, output: BinaryIO):
    with mrd.BinaryMrdReader(input) as reader:
        with mrd.BinaryMrdWriter(output) as writer:
            head = reader.read_header()
            if head is None:
                raise Exception("Could not read header")
            writer.write_header(head)
            writer.write_data(
                stream_item_sink(
                    mrdRecon(sign, reconMode, artMode, BoFit,
                              head, acquisition_reader(reader.read_data()))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstructs an MRD stream")
    parser.add_argument('-i', '--input', type=str, required=False, help="Input file, defaults to stdin")
    parser.add_argument('-o', '--output', type=str, required=False, help="Output file, defaults to stdout")
    parser.add_argument('-s', '--sign', type=str, required=False, help="Signs for code generalization [xsignG,ysignG,zsignG,cpPhase, artPhase]")
    parser.add_argument('-r', '--recon', type=str, required=False, help="Reconstruction mode")
    parser.add_argument('-artMode', '--artMode', type=str, default = False, required=False, help="ART mode")
    parser.add_argument('-BoFit', '--BoFit', type=str, default = False, required=False, help="Bo Fit string")
    # parser.set_defaults(
    #     input = '/home/tyger/tyger_repo_may/Next1_10.06/testMRDtoMAT_021.bin', 
    #     output = '/home/tyger/tyger_repo_may/Next1_10.06/recon_021.bin',
    #     sign = "[-1,-1,-1,1,1,1,1,1]",
    #     recon = 'cp', 
    #     artMode = 'art',
    #     BoFit = """8.34561453622739e-06 + 6.82076083616762e-06*(x**1) + 0.00032753360184207864*(y**1) 
    #                 -0.0004292342475020359*(z**1) + 0.0025643290966848283*(x**2) + 0.0005284674432167523*(y**1)*(x**1) 
    #                 -0.0015721327248562206*(z**1)*(x**1) + 0.01191219952127018*(y**2) -0.004518356665454803*(z**1)*(y**1) 
    #                 -0.007667029011400046*(z**2) -0.0014299121242289528*(x**3) -0.021811725036390173*(y**1)*(x**2) 
    #                 + 0.05702674289751107*(z**1)*(x**2) + 0.04381051038765055*(y**2)*(x**1) + 0.01836492877903395*(z**1)*(y**1)*(x**1) 
    #                 -0.07295843684808717*(z**2)*(x**1) -0.010276748810797987*(y**3) -0.06829935235620052*(z**1)*(y**2) 
    #                 -0.04068775523510165*(z**2)*(y**1) -0.11772526205454148*(z**3) -0.1972151378380283*(x**4) 
    #                 + 0.27149674751624897*(y**1)*(x**3) -0.7868096897440706*(z**1)*(x**3) -0.4727212416941673*(y**2)*(x**2) 
    #                 -2.5094073494594777*(z**1)*(y**1)*(x**2) -0.2963185508915456*(z**2)*(x**2) + 0.4417061790134502*(y**3)*(x**1) 
    #                 + 0.8423805711854779*(z**1)*(y**2)*(x**1) + 0.5985258614238003*(z**2)*(y**1)*(x**1) + 0.16474066584054992*(z**3)*(x**1) 
    #                 -2.0773597913799984*(y**4) + 1.33635321109116*(z**1)*(y**3) + 0.07264520809517627*(z**2)*(y**2) 
    #                 -0.028646732946244535*(z**3)*(y**1) -0.46715501576816165*(z**4) -0.7207838513445222*(x**5) + 2.776190176114737*(y**1)*(x**4) 
    #                 -9.888838014761227*(z**1)*(x**4) -24.125551275689663*(y**2)*(x**3) -11.766558598564227*(z**1)*(y**1)*(x**3) 
    #                 -12.937677458560529*(z**2)*(x**3) + 14.764799836499542*(y**3)*(x**2) + 23.017072158550857*(z**1)*(y**2)*(x**2) 
    #                 -3.593937971322034*(z**2)*(y**1)*(x**2) + 2.135119087362114*(z**3)*(x**2) + 16.124254841208256*(y**4)*(x**1) 
    #                 + 2.782001108297337*(z**1)*(y**3)*(x**1) + 51.25395307407008*(z**2)*(y**2)*(x**1) -38.724431850249935*(z**3)*(y**1)*(x**1) 
    #                 + 24.27848568791185*(z**4)*(x**1) -21.720803985383917*(y**5) + 24.690243171745976*(z**1)*(y**4) 
    #                 + 36.971011047530055*(z**2)*(y**3) + 16.12044099899119*(z**3)*(y**2) + 3.186678917589108*(z**4)*(y**1) + 18.71571454287562*(z**5)"""
    #     )
    args = parser.parse_args()

    input = open(args.input, "rb") if args.input is not None else sys.stdin.buffer
    output = open(args.output, "wb") if args.output is not None else sys.stdout.buffer

    reconstruct_mrd_stream(args.sign, args.recon, args.artMode, args.BoFit,
                            input, output)
