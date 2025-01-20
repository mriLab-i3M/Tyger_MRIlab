import sys
import argparse
import numpy as np
from typing import BinaryIO, Iterable, Union
import mrd
import matplotlib.pyplot as plt
from bart_marcos import bart_marcos2D
import cupy as cp 

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

def mrdRecon(reconMode: str, bartMode: str, artMode:str, #metric: str,weight: str,fixMask: str,maskFile: str,central_fraction: str,kspace_fraction: str,
              head: mrd.Header, input: Iterable[mrd.Acquisition]) -> Iterable[mrd.Image[np.float32]]:
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
        rFOVx = enc.recon_space.field_of_view_mm.x
        rFOVy = enc.recon_space.field_of_view_mm.y
        rFOVz = enc.recon_space.field_of_view_mm.z if enc.recon_space.field_of_view_mm.z else 1
    else:
        raise Exception('Required field of view information not found in header')

    buffer = None

    def produce_image(img: np.ndarray) -> Iterable[mrd.Image[np.float32]]:
        mrd_image = mrd.Image[np.float32](image_type=mrd.ImageType.MAGNITUDE, data=img)
        yield mrd_image
    
    buffer = np.zeros((eNz, eNy, eNx), dtype=np.complex64)
    
    for acq in input:
        k1 = acq.idx.kspace_encode_step_1 if acq.idx.kspace_encode_step_1 is not None else 0
        k2 = acq.idx.kspace_encode_step_2 if acq.idx.kspace_encode_step_2 is not None else 0
        slice = acq.idx.slice
        buffer[k2, k1, :] = acq.data

    def pythonfft(buffer):        
        img = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(buffer[:, :, :])))
        img = np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
        img = np.abs(img).astype(np.float32)
        print(img.dtype)
        return img
    def bart_python2D(buffer,bartMode): #, metric, weight, fixMask, maskFile, central_fraction, kspace_fraction): 
        kSpace = buffer[0,:,:]       
        img = bart_marcos2D(kSpace,bartMode)#, metric, weight, fixMask, maskFile, central_fraction, kspace_fraction)
        print(img.shape)
        img = np.abs(img).astype(np.float32)
        img = np.reshape(img, [1,1,img.shape[0],img.shape[1]])
        print(img.dtype)
        return img
    def pythonART(buffer, artMode):
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

        def positionVectors2D(nPoints, fov):
            """""
            # Inputs
            nPoints = [rd,ph,sl]
            fov = [rd,ph,sl]
            # Outputs
            rd, ph = posVectors
            """""
            res = fov / nPoints
            rd = np.linspace(0, nPoints[0, 0], nPoints[0, 0]) * res[0, 0] - fov[0, 0] / 2
            ph = np.linspace(0, nPoints[0, 1], nPoints[0, 1]) * res[0, 1] - fov[0, 1] / 2
            sRD, sPH = np.meshgrid(rd, ph)
            sRDRes = np.reshape(sRD, (nPoints[0, 1] * nPoints[0, 0]))
            sPHRes = np.reshape(sPH, (nPoints[0, 1] * nPoints[0, 0]))
            return rd, ph, sRDRes, sPHRes
        
        def kVectors2D(nPoints, fov):
            """""
            # Inputs
            nPoints = [rd,ph,sl]
            fov = [rd,ph,sl]
            # Outputs
            kRDRes, kPHRes = kVectors
            """""
            # res = fov / nPoints
            dK = 1 / fov
            kMax = nPoints / 2 * dK
            krd = np.linspace(0, nPoints[0, 0], nPoints[0, 0], endpoint = True) * dK[0, 0] - kMax[0, 0]
            kph = np.linspace(0, nPoints[0, 1], nPoints[0, 1], endpoint = True) * dK[0, 1] - kMax[0, 1]
            kRD, kPH = np.meshgrid(krd, kph)
            kRDRes = np.reshape(kRD, (nPoints[0, 1] * nPoints[0, 0]))
            kPHRes = np.reshape(kPH, (nPoints[0, 1] * nPoints[0, 0]))
            return kRDRes, kPHRes
        
        def art_cpu(kx, ky, x, y, s, rho, lbda, n_iter, index):
            n = 0
            n_samples = len(s)
            m = 0
            for iteration in range(n_iter):
                np.random.shuffle(index)
                for jj in range(n_samples):
                    ii = index[jj]
                    x0 = np.exp(-1j * 2 * np.pi * (kx[ii] * x + ky[ii] * y))
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

        def art_gpu(kx, ky, x, y, s, rho, lbda, n_iter, index):
            n = 0
            n_samples = len(s)
            m = 0
            for iteration in range(n_iter):
                cp.random.shuffle(index)
                for jj in range(n_samples):
                    ii = index[jj]
                    x0 = cp.exp(-1j * 2 * cp.pi * (kx[ii] * x + ky[ii] * y))
                    x1 = (x0.T @ rho) - s[ii]
                    x2 = x1 * cp.conj(x0) / (cp.conj(x0.T) @ x0)
                    d_rho = lbda * x2
                    rho -= d_rho
                    n += 1
                    if n / n_samples > 0.01:
                        m += 1
                        n = 0
                        print("ART iteration %i: %i %%" % (iteration + 1, m))

            return rho
        
        nPoints = np.zeros([1,3])
        fov = np.zeros([1,3])
        nPoints[0,0] = eNx
        nPoints[0,1] = eNy
        nPoints[0,2] = eNz
        fov[0,0] = rFOVx
        fov[0,1] = rFOVy
        fov[0,2] = rFOVz
        nPoints= nPoints.astype(int)
        fov = fov*1e-3
        print(fov)
        kSpace2D = np.reshape(buffer[0], (nPoints[0,1]*nPoints[0,0]))
        rd, ph, sRD, sPH = positionVectors2D(nPoints, fov)
        kRd, kPh = kVectors2D(nPoints, fov)
        rho = np.reshape(np.zeros((nPoints[:,0] * nPoints[:,1]), dtype=complex), (-1, 1))
        rho = rho[:,0]
        lbda = 1/float(1)
        n_iter = int(1)
        index = np.arange(len(rho))
        if artMode == 'artPK':
            GX = gradX(sRD,sPH,sPH*0)
            imgART= art_cpu(kRd,kPh, GX, sPH,kSpace2D, rho,lbda, n_iter, index)
            imgART = np.reshape(imgART, (1,1,nPoints[0,1],nPoints[0,0]))
        elif artMode == 'art':
            imgART= art_cpu(kRd,kPh, sRD, sPH,kSpace2D, rho,lbda, n_iter, index)
            imgART = np.reshape(imgART, (1,1,nPoints[0,1],nPoints[0,0]))
        elif artMode == 'art_gpu':
            kRd_gpu = cp.asarray(kRd)
            kPh_gpu = cp.asarray(kPh)
            sRD_gpu = cp.asarray(sRD)
            sPH_gpu = cp.asarray(sPH)
            kSpace2D_gpu = cp.asarray(kSpace2D)
            index_gpu = cp.asarray(index)
            rho_gpu = cp.asarray(rho)

            imgART_gpu= art_gpu(kRd_gpu,kPh_gpu, sRD_gpu, sPH_gpu,kSpace2D_gpu, rho_gpu,lbda, n_iter, index_gpu)
            imgART = cp.asnumpy(imgART_gpu)
            imgART = np.reshape(imgART, (1,1,nPoints[0,1],nPoints[0,0]))
        imgART = np.abs(imgART).astype(np.float32)
        return imgART

    if reconMode == 'pythonfft':
        imgRecon = pythonfft(buffer)
    elif reconMode == 'bart':
        imgRecon = bart_python2D(buffer,bartMode) #, metric, weight, fixMask, maskFile, central_fraction, kspace_fraction)
    elif reconMode == 'art':
        imgRecon = pythonART(buffer,artMode)
    # else: 
    #     imgRecon = pythonfft(buffer)

    # imgRecon [1,nSl,nPh, nRd]
    yield from produce_image(imgRecon)


def reconstruct_mrd_stream(reconMode: str, bartMode: str, artMode:str, #metric: str,weight: str,fixMask: str,maskFile: str,central_fraction: str,kspace_fraction: str,
                            input: BinaryIO, output: BinaryIO):
    with mrd.BinaryMrdReader(input) as reader:
        with mrd.BinaryMrdWriter(output) as writer:
            head = reader.read_header()
            if head is None:
                raise Exception("Could not read header")
            writer.write_header(head)
            writer.write_data(
                stream_item_sink(
                    mrdRecon(reconMode, bartMode, artMode,#metric, weight, fixMask, maskFile, central_fraction, kspace_fraction, 
                              head, acquisition_reader(reader.read_data()))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstructs an MRD stream")
    parser.add_argument('-i', '--input', type=str, required=False, help="Input file, defaults to stdin")
    parser.add_argument('-o', '--output', type=str, required=False, help="Output file, defaults to stdout")
    parser.add_argument('-r', '--recon', type=str, required=False, help="Reconstruction mode")
    parser.add_argument('-bartMode', '--bartMode', type=str, default = False, required=False, help="BART mode")
    parser.add_argument('-artMode', '--artMode', type=str, default = False, required=False, help="ART mode")
    # parser.add_argument('-metric', '--metric', type=str, default = '-l1', required=False, help="Metric")
    # parser.add_argument('-weight', '--weight', type=str, default = '-r0.15', required=False, help="Weight")
    # parser.add_argument('-fixMask', '--fixMask', type=int, default = 1, required=False, help="Fix Mask")
    # parser.add_argument('-maskFile', '--maskFile', type=str, default = 'maskGood2', required=False, help="Mask File")
    # parser.add_argument('-central_fraction', '--central_fraction', type=float, default = 0.01, required=False, help="Central KSpace Fraction")
    # parser.add_argument('-kspace_fraction', '--kspace_fraction', type=float, default = 0.5, required=False, help="KSpace Fraction")
    
    # parser.set_defaults(
    #     input = 'testART.bin', 
    #     output = 'resultART.bin',
    #     recon = 'art', 
    #     bartMode= 'cs', #,metric= None,weight= None,fixMask= None,maskFile= None,central_fraction= None,kspace_fraction= None
    #     artMode = 'artPK'
    #     )
    args = parser.parse_args()

    input = open(args.input, "rb") if args.input is not None else sys.stdin.buffer
    output = open(args.output, "wb") if args.output is not None else sys.stdout.buffer

    reconstruct_mrd_stream(args.recon,args.bartMode, args.artMode,#args.metric, args.weight, args.fixMask, 
                            #args.maskFile, args.central_fraction, args.kspace_fraction,
                            input, output)
