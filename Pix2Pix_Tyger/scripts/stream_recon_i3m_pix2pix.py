import sys
import argparse
import numpy as np
from typing import BinaryIO, Iterable, Union
import mrd
import matplotlib.pyplot as plt
from test_tyger import pix2pix_knee

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

def mrdRecon(head: mrd.Header, input: Iterable[mrd.Acquisition]) -> Iterable[mrd.Image[np.float32]]:
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

    img = np.fft.ifftshift(np.fft.ifftn((buffer)))
    img2D = np.abs(img[0, :, :])
    img2D = np.transpose(img2D,(1,0))
    img2D = np.flip(img2D, 0)
    img2D = np.flip(img2D, 1)
    imgRed = pix2pix_knee(img2D,img2D)
    imgRed = np.reshape(imgRed,(1,1,imgRed.shape[0],imgRed.shape[1]))
    imgRed = np.abs(imgRed).astype(np.float32)

    yield from produce_image(imgRed)


def reconstruct_mrd_stream(input: BinaryIO, output: BinaryIO):
    with mrd.BinaryMrdReader(input) as reader:
        with mrd.BinaryMrdWriter(output) as writer:
            head = reader.read_header()
            if head is None:
                raise Exception("Could not read header")
            writer.write_header(head)
            writer.write_data(
                stream_item_sink(
                    mrdRecon(head,
                            acquisition_reader(reader.read_data()))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstructs an MRD stream")
    parser.add_argument('-i', '--input', type=str, required=False, help="Input file, defaults to stdin")
    parser.add_argument('-o', '--output', type=str, required=False, help="Output file, defaults to stdout")
    # parser.add_argument('-r', '--recon', type=str, required=False, help="Reconstruction mode")
    # parser.set_defaults(
    #     input = None, 
    #     output = None,
    #     recon = None)
    # parser.set_defaults(
    #     input = '163ks.bin', 
    #     output = 'resultRed.bin',
    #     )
    args = parser.parse_args()

    input = open(args.input, "rb") if args.input is not None else sys.stdin.buffer
    output = open(args.output, "wb") if args.output is not None else sys.stdout.buffer

    reconstruct_mrd_stream(input, output)
