import sys
import argparse
import numpy as np
from typing import BinaryIO, Iterable, Union
import mrd
import matplotlib.pyplot as plt
from bart_marcos import bart_marcos2D

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

def mrdRecon(reconMode: str, bartMode: str, #metric: str,weight: str,fixMask: str,maskFile: str,central_fraction: str,kspace_fraction: str,
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
    if reconMode == 'pythonfft':
        imgRecon = pythonfft(buffer)
    elif reconMode == 'bart':
        imgRecon = bart_python2D(buffer,bartMode) #, metric, weight, fixMask, maskFile, central_fraction, kspace_fraction)
    else: 
        imgRecon = pythonfft(buffer)

    # imgRecon [1,nSl,nPh, nRd]
    yield from produce_image(imgRecon)


def reconstruct_mrd_stream(reconMode: str, bartMode: str, #metric: str,weight: str,fixMask: str,maskFile: str,central_fraction: str,kspace_fraction: str,
                            input: BinaryIO, output: BinaryIO):
    with mrd.BinaryMrdReader(input) as reader:
        with mrd.BinaryMrdWriter(output) as writer:
            head = reader.read_header()
            if head is None:
                raise Exception("Could not read header")
            writer.write_header(head)
            writer.write_data(
                stream_item_sink(
                    mrdRecon(reconMode, bartMode, #metric, weight, fixMask, maskFile, central_fraction, kspace_fraction, 
                              head, acquisition_reader(reader.read_data()))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstructs an MRD stream")
    parser.add_argument('-i', '--input', type=str, required=False, help="Input file, defaults to stdin")
    parser.add_argument('-o', '--output', type=str, required=False, help="Output file, defaults to stdout")
    parser.add_argument('-r', '--recon', type=str, required=False, help="Reconstruction mode")
    parser.add_argument('-bartMode', '--bartMode', type=str, default = False, required=False, help="BART mode")
    # parser.add_argument('-metric', '--metric', type=str, default = '-l1', required=False, help="Metric")
    # parser.add_argument('-weight', '--weight', type=str, default = '-r0.15', required=False, help="Weight")
    # parser.add_argument('-fixMask', '--fixMask', type=int, default = 1, required=False, help="Fix Mask")
    # parser.add_argument('-maskFile', '--maskFile', type=str, default = 'maskGood2', required=False, help="Mask File")
    # parser.add_argument('-central_fraction', '--central_fraction', type=float, default = 0.01, required=False, help="Central KSpace Fraction")
    # parser.add_argument('-kspace_fraction', '--kspace_fraction', type=float, default = 0.5, required=False, help="KSpace Fraction")
    
    # parser.set_defaults(
    #     input = 'toTestBart.bin', 
    #     output = 'testResult.bin',
    #     recon = 'bart', 
    #     bartMode= 'cs'#,metric= None,weight= None,fixMask= None,maskFile= None,central_fraction= None,kspace_fraction= None
    #     )
    args = parser.parse_args()

    input = open(args.input, "rb") if args.input is not None else sys.stdin.buffer
    output = open(args.output, "wb") if args.output is not None else sys.stdout.buffer

    reconstruct_mrd_stream(args.recon,args.bartMode, #args.metric, args.weight, args.fixMask, 
                            #args.maskFile, args.central_fraction, args.kspace_fraction,
                            input, output)
