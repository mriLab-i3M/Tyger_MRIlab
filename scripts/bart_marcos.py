import numpy as np
import subprocess
import os
import scipy.io as sio
import argparse
import matplotlib.pyplot as plt

def bart_marcos2D(kspace_full, bartMode): #, metric, weight, fixMask, maskFile, central_fraction, kspace_fraction):
    metric = '-l1' 
    weight = '-r0.15'
    fixMask = 1
    maskFile = 'maskFix2D' 
    central_fraction = 0.01 
    kspace_fraction = 0.5

    ## Functions
    # Save the k-space data and the mask to files in the BART format (.cfl/.hdr)
    def save_bart_data(filename, array):
        """Helper function to save numpy arrays to BART format (.cfl/.hdr)"""
        array = np.ascontiguousarray(array)
        with open(filename + '.cfl', 'wb') as f:
            f.write(array.astype(np.complex64).T.tobytes())
        with open(filename + '.hdr', 'w') as f:
            f.write('# Dimensions\n')
            f.write(' '.join(map(str, array.shape[::-1])) + '\n')

    # Load the reconstructed image
    def load_bart_data(filename):
        """Helper function to load BART format (.cfl/.hdr) files"""
        with open(filename + '.hdr', 'r') as f:
            dims = tuple(map(int, f.readlines()[1].split()))
        with open(filename + '.cfl', 'rb') as f:
            data = np.fromfile(f, dtype=np.complex64)
            return data.reshape(dims[::-1]).T

    ## BART reconstruction
    fileBart = 'kspace'
    if bartMode == 'cs':
        ## Mask - kSpace undersampled
        if fixMask == 1:
            mask = load_bart_data(maskFile)
        else:
            mask = np.zeros(kspace_full.shape, dtype=bool)
            phase_lines = np.random.rand(*kspace_full[0].shape) < kspace_fraction
            mask[phase_lines, :] = True

            central_width = int(kspace_full.shape[0] * central_fraction)  
            center_start = kspace_full.shape[0] // 2 - central_width // 2
            center_end = kspace_full.shape[0] // 2 + central_width // 2
            mask[center_start:center_end, :] = True

        kspace_undersampled = kspace_full * mask

        ## Save file compatible with BART
        save_bart_data(fileBart, kspace_undersampled)

        ## Campressed Sensing
        # Command: bart pics -l1 -r0.01 kspace mask output
        subprocess.run(['bart', 'pics', metric, weight, fileBart, 'sens1coil', 'reconstructed'],stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        ## Load reconstructed image and changing orientations
        reconstructed_image = load_bart_data('reconstructed')
        reconstructed_image = np.squeeze(reconstructed_image)
        reconstructed_image = np.transpose(reconstructed_image, [1,0])
        reconstructed_image = np.flip(reconstructed_image, [0])

    elif bartMode == 'fft_us':
        ## Mask - kSpace undersampled
        if fixMask == 1:
            mask = load_bart_data(maskFile)
        else:
            mask = np.zeros(kspace_full.shape, dtype=bool)
            phase_lines = np.random.rand(*kspace_full[0].shape) < kspace_fraction
            mask[phase_lines, :] = True

            central_width = int(kspace_full.shape[0] * central_fraction)  
            center_start = kspace_full.shape[0] // 2 - central_width // 2
            center_end = kspace_full.shape[0] // 2 + central_width // 2
            mask[center_start:center_end, :] = True

        kspace_undersampled = kspace_full * mask
        ## Save file compatible with BART
        save_bart_data(fileBart, kspace_undersampled)

        ## FFT
        subprocess.run(['bart', 'fft', '-u', '3', fileBart, 'reconstructed'])

        ## Load reconstructed image and changing orientations
        reconstructed_image = load_bart_data('reconstructed')
        reconstructed_image = np.squeeze(reconstructed_image)
        reconstructed_image = np.transpose(reconstructed_image, [1,0])
        reconstructed_image = np.flip(reconstructed_image, [1])

    elif bartMode == 'fft':
        ## Save file compatible with BART
        save_bart_data(fileBart, kspace_full)

        ## FFT
        subprocess.run(['bart', 'fft', '-u', '3', fileBart, 'reconstructed'])

        ## Load reconstructed image and changing orientations
        reconstructed_image = load_bart_data('reconstructed')
        reconstructed_image = np.squeeze(reconstructed_image)
        reconstructed_image = np.transpose(reconstructed_image, [1,0])
        reconstructed_image = np.flip(reconstructed_image, [1])
        
            
    # ## Plot to check
    # plt.figure()
    # plt.imshow(np.abs(img2DPlot), cmap= 'gray')
    # plt.title('Full KSpace')
    # plt.axis('off')

    # plt.figure()
    # plt.imshow(np.abs(reconstructed_image), cmap='gray')
    # plt.title('Reconstructed Image')
    # plt.axis('off')
    # plt.show()

    return reconstructed_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert mat to MRD")
    parser.add_argument('-bartMode', '--bartMode', type=str, default = 'cs', required=False, help="BART mode")
    # parser.add_argument('-metric', '--metric', type=str, default = '-l1', required=False, help="Metric")
    # parser.add_argument('-weight', '--weight', type=str, default = '-r0.15', required=False, help="Weight")
    # parser.add_argument('-fixMask', '--fixMask', type=int, default = 1, required=False, help="Fix Mask")
    # parser.add_argument('-maskFile', '--maskFile', type=str, default = 'maskGood2', required=False, help="Mask File")
    # parser.add_argument('-central_fraction', '--central_fraction', type=float, default = 0.01, required=False, help="Central KSpace Fraction")
    # parser.add_argument('-kspace_fraction', '--kspace_fraction', type=float, default = 0.5, required=False, help="KSpace Fraction")
    args = parser.parse_args()
     
    ## Load data - Esto lo haré fuera de tyger
    input = '/home/teresa/Documentos/Tyger/bart/knee44L.mat'
    mat_data = sio.loadmat(input)
    kSpace3D = mat_data['kSpace3D']
    nSlice2D = kSpace3D.shape[0]//2
    img = np.fft.ifftshift(np.fft.ifftn((kSpace3D)))
    img2D = img[int(nSlice2D), :, :]
    kspace_full = np.fft.fftn(np.fft.fftshift((img2D)))
    img2DPlot = np.fft.ifftshift(np.fft.ifftn((kspace_full)))    
    img2DPlot = np.transpose(img2DPlot, [1,0])
    img2DPlot = np.flip(img2DPlot, [0])

    reconstructed_image = bart_marcos2D(kspace_full, args.bartMode)#, args.metric, args.weight, args.fixMask, args.maskFile, args.central_fraction, args.kspace_fraction)
    