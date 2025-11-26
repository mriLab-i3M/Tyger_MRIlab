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
    vecSign = ast.literal_eval(args.sign)
    enc = head.encoding[0]

    # Matrix size
    if enc.encoded_space and enc.recon_space and enc.encoded_space.matrix_size and enc.recon_space.matrix_size:
        eNx = enc.encoded_space.matrix_size.x
        eNy = enc.encoded_space.matrix_size.y
        eNz = enc.encoded_space.matrix_size.z
        rNx = enc.recon_space.matrix_size.x
        rNy = enc.recon_space.matrix_size.y
        rNz = enc.recon_space.matrix_size.z
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

    def produce_image(img: np.ndarray) -> Iterable[mrd.Image[np.float32]]:
        mrd_image = mrd.Image[np.float32](image_type=mrd.ImageType.MAGNITUDE, data=img)
        yield mrd_image
    
    kSpace_buffer = []
    kx_buffer = []
    ky_buffer = []
    kz_buffer = []

    for acq in input:

        k1 = acq.idx.kspace_encode_step_1 if acq.idx.kspace_encode_step_1 is not None else 0
        k2 = acq.idx.kspace_encode_step_2 if acq.idx.kspace_encode_step_2 is not None else 0

        # # kSpace_buffer = np.concatenate((kSpace_buffer, acq.data[0]), axis = 0) # Much slower!
        kSpace_buffer.append(acq.data[0])
        kx_buffer.append(acq.trajectory[0,:])
        ky_buffer.append(acq.trajectory[1,:])
        kz_buffer.append(acq.trajectory[2,:])

    # kSpace_buffer = np.reshape(kSpace_buffer, -1, order='C')
    # kx_buffer = np.reshape(kx_buffer, -1, order='C')
    # ky_buffer = np.reshape(ky_buffer, -1, order='C')
    # kz_buffer= np.reshape(kz_buffer, -1, order='C')
    
    kSpace_buffer = np.concatenate(kSpace_buffer)
    kx_buffer = np.concatenate(kx_buffer)
    ky_buffer = np.concatenate(ky_buffer)
    kz_buffer = np.concatenate(kz_buffer)
    
    def pythonfft(kSpace):        
        img = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kSpace[:, :, :])))
        img = np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
        img = np.abs(img).astype(np.float32)
        return img
    
    def pythonART():
        # gammabar = 42.57747892*1e6

        signal = kSpace_buffer
        kx = 2*np.pi*kx_buffer
        ky = 2*np.pi*ky_buffer
        kz = 2*np.pi*kz_buffer
        
        nn = int(rNx*rNy*rNz)
        x = np.zeros(nn)
        y = np.zeros(nn)
        z = np.zeros(nn)
        for ind in range(nn):
            i = ind % rNx
            j = (ind // rNx) % rNy
            k = ind // (rNx * rNy)
            x[ind] = -rFOVx/2 + i * rFOVx / (rNx - 1) + (rFOVx / (2 * rNx))
            y[ind] = -rFOVy/2 + j * rFOVy / (rNy - 1) + (rFOVy / (2 * rNy))
            z[ind] = -rFOVz/2 + k * rFOVz / (rNz - 1) + (rFOVz / (2 * rNz))
                
        # boFit = eval(f"lambda x, y, z: {BoFit}")
        # dBo = boFit(vecSign[0]*x, vecSign[1]*y, vecSign[2]*z)
        
        rho = np.reshape(np.zeros((rNx*rNy*rNz), dtype=complex), (-1, 1))
        rho = rho[:,0]
        lbda = 1/float(1)
        n_iter = int(1)
        # index = np.arange(len(signal))

        kx_gpu = cp.asarray(kx)
        ky_gpu = cp.asarray(ky)
        kz_gpu = cp.asarray(kz)
        sx_gpu = cp.asarray(x)
        sy_gpu = cp.asarray(y)
        sz_gpu = cp.asarray(z)
        signal_gpu = cp.asarray(signal)
        # index_gpu = cp.asarray(index)
        rho_gpu = cp.asarray(rho)
        
        def art(kx, ky, kz, x, y, z, s, rho, lbda, n_iter):
                    n = 0
                    n_samples = len(s)
                    m = 0
                    for iteration in range(n_iter):
                        # cp.random.shuffle(index)
                        for ii in range(n_samples):
                            # ii = index[jj]
                            x0 = cp.exp(vecSign[4]*-1j *  (kx[ii] * x + ky[ii] * y + kz[ii] * z))
                            x1 =  s[ii]-(x0.T @ rho)
                            x2 = x1 * cp.conj(x0) / (cp.conj(x0.T) @ x0)
                            x2 = x1 * cp.conj(x0) / (rNx*rNy*rNz)
                            d_rho = lbda * x2
                            rho += d_rho
                            n += 1
                            if n / n_samples > 0.01:
                                m += 1
                                n = 0
                                print("ART iteration %i: %i %%" % (iteration + 1, m))

                    return rho
        
        # def art_batch(kx, ky, kz, x, y, z, s, rho, lbda, n_iter, batch_size=500):
        #     """
        #     ART optimizado por bloques para GPU usando CuPy.
        #     """
        #     n_samples = len(s)
        #     N_voxels = x.shape[0]  # asumimos x, y, z tienen misma forma
        #     print("ART con batches de tamaño", batch_size)

        #     for iteration in range(n_iter):
        #         print(f"ART iteration {iteration + 1}/{n_iter}")

        #         for start in range(0, n_samples, batch_size):
        #             end = min(start + batch_size, n_samples)
        #             bsize = end - start

        #             # Extrae bloques de datos
        #             kx_batch = kx[start:end].reshape(-1, 1)  # shape (B,1)
        #             ky_batch = ky[start:end].reshape(-1, 1)
        #             kz_batch = kz[start:end].reshape(-1, 1)
        #             s_batch  = s[start:end].reshape(-1)      # shape (B,)

        #             # Calcula la fase: broadcasting eficiente
        #             # x, y, z: (N,), kx_batch: (B,1) → resultado: (B, N)
        #             phase = kx_batch * x[None, :] + ky_batch * y[None, :] + kz_batch * z[None, :]
        #             x0 = cp.exp(-1j * vecSign[4] * phase)  # shape (B, N)

        #             # Predicción: producto escalar por batch
        #             pred = x0 @ rho                        # shape (B,)
        #             residual = s_batch - pred              # shape (B,)

        #             # Actualización de rho
        #             update = residual[:, None] * cp.conj(x0) / N_voxels  # shape (B, N)
        #             d_rho = lbda * cp.sum(update, axis=0)                # shape (N,)
        #             rho += d_rho

        #     return rho
        
        imgART= art(kx_gpu,ky_gpu,kz_gpu,sx_gpu,sy_gpu,sz_gpu,signal_gpu, rho_gpu,lbda, n_iter)
        img = cp.asnumpy(imgART)
        img = np.reshape(img, (1,eNx,eNy,eNz))
        img = np.abs(img).astype(np.float32)
        return img
    
    if reconMode == 'fft':
        kSpace = np.reshape(kSpace_buffer, [rNx,rNy,rNz])
        imgRecon = pythonfft(kSpace)
    elif reconMode == 'art' or reconMode == 'artpk':
        imgRecon = pythonART()
        
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
    parser.add_argument('-r', '--recon', type=str, required=False, help="Reconstruction mode (fft, art, artpk)")
    parser.add_argument('-s', '--sign', type=str, required=False, help="Signs and others for code generalization [xsignG,ysignG,zsignG,cpPhase,artPhase,dfovx_sign,dfovy_sign,dfovz_sign, cp_batchsize]")
    parser.add_argument('-BoFit', '--BoFit', type=str, default = False, required=False, help="Bo Fit string")
    
    # parser.set_defaults(
    #     input = '/home/tyger/tyger_repo_may/Tyger_MRIlab/PETRA_recon/recon_scripts/testPETRA.bin', 
    #     output = '/home/tyger/tyger_repo_may/Tyger_MRIlab/PETRA_recon/recon_scripts/reconPETRA.bin',
    #     recon = 'art', 
    #     sign = "[-1,-1,-1,1,1,1,1,1,1000]",
    #     BoFit = """0*x+0*y+0*z"""
    #     )
    
    args = parser.parse_args()

    input = open(args.input, "rb") if args.input is not None else sys.stdin.buffer
    output = open(args.output, "wb") if args.output is not None else sys.stdout.buffer

    reconstruct_mrd_stream(args.recon, args.sign, args.BoFit,
                            input, output)
