import os
import subprocess
import sys
import scipy.io as sio
import numpy as np
import mrd
from typing import Generator
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path

############################### Run Tyger ############################################

def run_cmd(cmd, stdin_file=None, stdout_file=None):
    """
    Runs a command and optionally redirects stdin/stdout to files.
    Returns stdout as stripped text when no stdout_file is provided.
    """
    if stdin_file is not None and stdout_file is not None:
        with open(stdin_file, "rb") as fin, open(stdout_file, "wb") as fout:
            subprocess.run(cmd, stdin=fin, stdout=fout, check=True)
        return None

    if stdin_file is not None:
        with open(stdin_file, "rb") as fin:
            subprocess.run(cmd, stdin=fin, check=True)
        return None

    if stdout_file is not None:
        with open(stdout_file, "wb") as fout:
            subprocess.run(cmd, stdout=fout, check=True)
        return None

    result = subprocess.check_output(cmd, text=True)
    return result.strip()

def create_tyger_buffer(tags=None):
    cmd = ["tyger", "buffer", "create"]
    if tags:
        for key, value in tags.items():
            cmd.extend(["--tag", f"{key}={value}"])
    return run_cmd(cmd)

def run_tyger_denoising(path_input_mrd, path_output_mrd):
    # Create input buffer
    data_id = create_tyger_buffer()

    # Write input MRD into Tyger buffer
    run_cmd(["tyger", "buffer", "write", data_id], stdin_file=path_input_mrd)

    # Create noise covariance buffer
    noisecovariance_id = create_tyger_buffer()

    # Run noise adjustment
    noise_adjustment_run_id = run_cmd([
        "tyger", "run", "create",
        "-f", r"noise_adjustment.yml",
        "--buffer", f"noisecovariance={noisecovariance_id}",
        "--buffer", f"input={data_id}",
    ])
    print(f"Noise adjustment run ID: {noise_adjustment_run_id}")

    # Create reconstruction buffer
    recon_id = create_tyger_buffer(tags={"scan_type": "3D", "recon": "ai"})

    # Run denoising
    denoising_run_id = run_cmd([
        "tyger", "run", "create",
        "-f", r"run_denoising.yml",
        "--buffer", f"noisecovariance={noisecovariance_id}",
        "--buffer", f"input={data_id}",
        "--buffer", f"recon={recon_id}",
    ])
    print(f"Denoising run ID: {denoising_run_id}")

    # Read reconstruction buffer into output MRD
    run_cmd(["tyger", "buffer", "read", recon_id], stdout_file=path_output_mrd)

########################## MaRCoS/MaRGE rawDatas ######################################

def matToMRD(input, output_file, input_field=None):
    # print('From MAT to MRD...')
   
    # OUTPUT - write .mrd
    output = sys.stdout.buffer
    if output_file is not None:
        output = output_file
       
    # INPUT - Read .mat
    mat_data = sio.loadmat(input)
   
    # Head info
    axesOrientation = mat_data['axesOrientation'][0] # indicate the order of the dimensions in the data (rd, ph, sl) and how they are oriented in space (x, y, z)
    nPoints = mat_data['nPoints'][0] # rd, ph, sl
    nPoints = [int(x) for x in nPoints]
   
    try: # RAREpp and RARE_double_image
        rdGradAmplitude = mat_data['rd_grad_amplitude']
    except: # RAREprotocols
        rdGradAmplitude = mat_data['rdGradAmplitude']
       
    fov = mat_data['fov'][0]*1e1; # fov is in x, y, z order in the .mat file, but we need to reorder it to match the axes orientation (rd, ph, sl)
    fov_adq = [fov[axesOrientation[k]] for k in range(3)] # rd, ph, sl
       
    dfov = mat_data['dfov'][0]*1e-3; dfov = dfov.astype(np.float32)  # mm; x, y, z
    dfov_adq = [dfov[axesOrientation[k]] for k in range(3)] # rd, ph, sl
   
    acqTime = mat_data['acqTime'][0]*1e-3 # s
    bw = mat_data['bw_MHz'][0][0]*1e6 # Hz
    dwell = 1/bw* 1e9 # ns
   
    # parFourierFraction: Partial fourier fraction. Fraction of k planes aquired in slice direction
    parFourierFraction = mat_data['parFourierFraction'][0][0].item()
   
    # partialAcquisition: While doing partial acquisition, this is the number of extra slices acquired next to half nSlices / 2
    partialAcquisition = mat_data['partialAcquisition'][0][0].item()
   
    print(f"axesOrientation: {axesOrientation}, all data are in the order of rd, ph, sl, but the orientation of these dimensions in space is given by axesOrientation, where 0, 1, 2 correspond to x, y, z respectively")
    print(f"axesOrientation: {axesOrientation}")
    print(f"nPoints: {nPoints}")
    print(f"fov_adq: {fov_adq}, dfov_adq: {dfov_adq}")
    print(f"acqTime: {acqTime}, bw: {bw}, dwell: {dwell}")
    print(f"parFourierFraction: {parFourierFraction}, partialAcquisition: {partialAcquisition}")
   
    # Signal vector
    # sampledCartesian is a 4-D array with the following columns: kx, ky, kz, signal. The rows are ordered according to the acquisition order (rd, ph, sl)
    sampledCartesian = mat_data['sampledCartesian']
    signal = sampledCartesian[:,3]
    if input_field is not None:
        kSpace = mat_data[input_field]
    else:        
        kSpace = np.reshape(signal, (1, nPoints[2], nPoints[1], nPoints[0])) # sl, ph, rd
 
    # k vectors, sampled kspace trajectory in rd, ph, sl order
    kTrajec = np.real(sampledCartesian[:,0:3]).astype(np.float32)    # rd, ph, sl
 
    krd = kTrajec[:,0]
    krd = np.reshape(krd, (1, nPoints[2], nPoints[1], nPoints[0])) # sl, ph, rd
 
    kph = kTrajec[:,1]
    kph = np.reshape(kph, (1, nPoints[2], nPoints[1], nPoints[0])) # sl, ph, rd
 
    ksl = kTrajec[:,2]
    ksl = np.reshape(ksl, (1, nPoints[2], nPoints[1], nPoints[0])) # sl, ph, rd
 
    rdTimes = np.linspace(-acqTime / 2, acqTime / 2, num=nPoints[0])
    rdTimes = np.reshape(rdTimes, (1,1,1, rdTimes.shape[0]))
 
    # Position vectors
    rd_pos = np.linspace(-fov_adq[0] / 2 , fov_adq[0] / 2 , nPoints[0], endpoint=False)
    ph_pos = np.linspace(-fov_adq[1] / 2 , fov_adq[1] / 2 , nPoints[1], endpoint=False)
    sl_pos = np.linspace(-fov_adq[2] / 2 , fov_adq[2] / 2 , nPoints[2], endpoint=False)
    ph_posFull, sl_posFull, rd_posFull = np.meshgrid(ph_pos, sl_pos, rd_pos)
    rd_posFull = np.reshape(rd_posFull, (nPoints[0] * nPoints[1] * nPoints[2], 1))
    ph_posFull = np.reshape(ph_posFull, (nPoints[0] * nPoints[1] * nPoints[2], 1))
    sl_posFull = np.reshape(sl_posFull, (nPoints[0] * nPoints[1] * nPoints[2], 1))
    rd_ph_sl_matri = np.concatenate((rd_posFull, ph_posFull, sl_posFull), axis=1) # rd, ph, sl
   
    rd_esp = rd_ph_sl_matri[:,0]  # sl, ph, rd    
    rd_esp = np.reshape(rd_esp, (1, nPoints[2], nPoints[1], nPoints[0]))
 
    ph_esp = rd_ph_sl_matri[:,1] # sl, ph, rd
    ph_esp = np.reshape(ph_esp, (1, nPoints[2], nPoints[1], nPoints[0]))
 
    sl_esp = rd_ph_sl_matri[:,2] # sl, ph, rd
    sl_esp = np.reshape(sl_esp, (1, nPoints[2], nPoints[1], nPoints[0]))
   
    ## Noise acq
    data_noise = mat_data['data_noise']
    nNoise = mat_data['nNoise'][0][0].item()
 
    print(f"nNoise: {nNoise}")
    print(f"sampledCartesian shape: {sampledCartesian.shape}")
    print(f"kSpace shape: {kSpace.shape}, krd shape: {krd.shape}, kph shape: {kph.shape}, ksl shape: {ksl.shape}")
 
    # OUTPUT - write .mrd
    # MRD Format
    h = mrd.Header()
 
    sys_info = mrd.AcquisitionSystemInformationType()
    sys_info.receiver_channels = 1
    sys_info.system_field_strength_t = 0.088
    sys_info.system_vendor = "PhysioMRI"
    sys_info.system_model = "odin"
    sys_info.relative_receiver_noise_bandwidth = 0.72
    sys_info.receiver_channels = 1
    sys_info.coil_label = [mrd.CoilLabelType(coil_number=0, coil_name="coil_1")]
    sys_info.institution_name = "MSR"
    sys_info.station_name = "ice_box"
    sys_info.device_id = "001"
    sys_info.device_serial_number = "001"
 
    h.acquisition_system_information = sys_info
 
    e = mrd.EncodingSpaceType()
    e.matrix_size = mrd.MatrixSizeType(x=nPoints[0], y=nPoints[1], z=nPoints[2])
    e.field_of_view_mm = mrd.FieldOfViewMm(x=fov_adq[0], y=fov_adq[1], z=fov_adq[2])
 
    r = mrd.EncodingSpaceType()
    r.matrix_size = mrd.MatrixSizeType(x=nPoints[0], y=nPoints[1], z=nPoints[2])
    r.field_of_view_mm = mrd.FieldOfViewMm(x=fov_adq[0], y=fov_adq[1], z=fov_adq[2])
 
    enc = mrd.EncodingType()
    enc.trajectory = mrd.Trajectory.CARTESIAN
    enc.encoded_space = e
    enc.recon_space = r
 
    enc.encoding_limits = mrd.EncodingLimitsType()
    enc.encoding_limits.kspace_encoding_step_0 = mrd.LimitType(minimum=0, maximum=nPoints[0]-1, center=(nPoints[0])//2)
    enc.encoding_limits.kspace_encoding_step_1 = mrd.LimitType(minimum=0, maximum=nPoints[1]-1, center=(nPoints[1])//2)
   
    if nPoints[2] >1 and parFourierFraction < 1.0 and partialAcquisition > 0:
        # partial fourier acquisition in slice direction
        acquired_e2 = nPoints[2]//2 + partialAcquisition
        enc.encoding_limits.kspace_encoding_step_2 = mrd.LimitType(minimum=0, maximum=acquired_e2, center=(nPoints[2])//2) # post-zero type partial fourier, we acquire more than half of the k-space lines in slice direction, so the center is still in the middle of the acquired lines
        print(f"Partial Fourier acquisition in slice direction is detected: acquired_e2={acquired_e2} out of {nPoints[2]}")
    else:
        enc.encoding_limits.kspace_encoding_step_2 = mrd.LimitType(minimum=0, maximum=nPoints[2]-1, center=(nPoints[2])//2)
        print(f"Partial Fourier acquisition in slice direction is not detected")
       
    enc.encoding_limits.average = mrd.LimitType(minimum=0, maximum=0, center=0)
    enc.encoding_limits.slice = mrd.LimitType(minimum=0, maximum=0, center=0)
    enc.encoding_limits.contrast = mrd.LimitType(minimum=0, maximum=0, center=0)
    enc.encoding_limits.phase = mrd.LimitType(minimum=0, maximum=0, center=0)
    enc.encoding_limits.repetition = mrd.LimitType(minimum=0, maximum=0, center=0)
    enc.encoding_limits.set = mrd.LimitType(minimum=0, maximum=0, center=0)
    enc.encoding_limits.segment = mrd.LimitType(minimum=0, maximum=0, center=0)
 
    parallel_imaging = mrd.ParallelImagingType()
    parallel_imaging.acceleration_factor = mrd.AccelerationFactorType()
    parallel_imaging.acceleration_factor.kspace_encoding_step_1 = 1
    parallel_imaging.acceleration_factor.kspace_encoding_step_2 = 1
    parallel_imaging.calibration_mode = mrd.CalibrationMode.NOACCELERATION 
    enc.parallel_imaging = parallel_imaging

    h.encoding.append(enc)
   
    readout_gradient = mrd.UserParameterDoubleType()
    readout_gradient.name = "readout_gradient_intensity"
    readout_gradient.value = float(np.squeeze(rdGradAmplitude))
   
    axes_param = mrd.UserParameterStringType()
    axes_param.name = "axesOrientation"
    axes_param.value = ",".join(map(str, axesOrientation))  
   
    d_fov = mrd.UserParameterStringType()
    d_fov.name = "dfov"
    d_fov.value = ",".join(map(str, dfov_adq))  
   
    if h.user_parameters is None:
        h.user_parameters = mrd.UserParametersType()
    h.user_parameters.user_parameter_double.append(readout_gradient)
    h.user_parameters.user_parameter_string.append(axes_param)
    h.user_parameters.user_parameter_string.append(d_fov)
 
    print(f"mrd header: {h}")
 
    def generate_data() -> Generator[mrd.StreamItem, None, None]:
        acq = mrd.Acquisition()
 
        acq.data = np.zeros((1, nPoints[0]), dtype=np.complex64)
        acq.trajectory = np.zeros((7, nPoints[0]), dtype=np.float32)
        acq.head.center_sample = round(nPoints[0] / 2)
       
        for n in range(nNoise):
            noise = mrd.Acquisition()
            noise.data = np.zeros((1, nPoints[0]), dtype=np.complex64)
            noise.trajectory = np.zeros((0, 0), dtype=np.float32)
            noise.head.center_sample = round(nPoints[0] / 2)
 
            noise.head.scan_counter = n
            noise.head.sample_time_ns = int(dwell)
            noise.head.acquisition_time_stamp_ns = int(n * 2.5 * 1e3)
            noise.head.physiology_time_stamp_ns = [int(2.5 * n * 1e3), 0, 0]
 
            noise.head.channel_order = [0]
 
            noise.head.flags = mrd.AcquisitionFlags(0)
            noise.head.flags |= mrd.AcquisitionFlags.IS_NOISE_MEASUREMENT
            noise.head.idx.kspace_encode_step_1 = n
            noise.head.idx.kspace_encode_step_2 = 0
            noise.head.idx.slice = 0
            noise.head.idx.repetition = 0
            noise.head.idx.average = 0
            noise.head.idx.phase = 0
            noise.head.idx.set = 0
            noise.head.idx.contrast = 0
            noise.head.idx.segment = 0
            noise.data[:] = data_noise[n, :]
            yield mrd.StreamItem.Acquisition(noise)
       
        for s in range(nPoints[2]):
            for line in range(nPoints[1]):
 
                num = (line + s * nPoints[1])
 
                acq.head.flags = mrd.AcquisitionFlags(0)
                if line == 0:
                    acq.head.flags |= mrd.AcquisitionFlags.FIRST_IN_ENCODE_STEP_1
                    acq.head.flags |= mrd.AcquisitionFlags.FIRST_IN_SLICE
                    acq.head.flags |= mrd.AcquisitionFlags.FIRST_IN_REPETITION
                if line == nPoints[1] - 1:
                    acq.head.flags |= mrd.AcquisitionFlags.LAST_IN_ENCODE_STEP_1
                    acq.head.flags |= mrd.AcquisitionFlags.LAST_IN_SLICE
                    acq.head.flags |= mrd.AcquisitionFlags.LAST_IN_REPETITION

                # ACS / calibration lines
                n_acs = 24
                center = nPoints[1] // 2
                acs_start = center - n_acs // 2
                acs_end = center + n_acs // 2

                if acs_start <= line < acs_end:
                    acq.head.flags |= mrd.AcquisitionFlags.IS_PARALLEL_CALIBRATION_AND_IMAGING
                    
                acq.head.scan_counter = num + nNoise
               
                acq.head.acquisition_time_stamp_ns = int(num * 2 * 1e9)
                acq.head.physiology_time_stamp_ns = [int(2.5*num * 1e9), 0, 0]
 
                acq.head.channel_order = [0]
 
                acq.head.discard_pre = 0
                acq.head.discard_post = 0
               
                acq.head.center_sample = round(nPoints[0] / 2)
                acq.head.sample_time_ns = int(dwell)
               
                acq.head.idx.kspace_encode_step_1 = line
                acq.head.idx.kspace_encode_step_2 = s
                acq.head.idx.slice = 0
                acq.head.idx.repetition = 0
                acq.head.idx.average = 0
                acq.head.idx.phase = 0
                acq.head.idx.set = 0
                acq.head.idx.contrast = 0
                acq.head.idx.segment = 0
                acq.data[:] = kSpace[:, s, line, :]
                acq.trajectory[0,:] = krd[:, s, line, :]
                acq.trajectory[1,:] = kph[:, s, line, :]
                acq.trajectory[2,:] = ksl[:, s, line, :]
                acq.trajectory[3,:] = rdTimes[:, :, :, :]
                acq.trajectory[4,:] = rd_esp[:, s, line, :]
                acq.trajectory[5,:] = ph_esp[:, s, line, :]
                acq.trajectory[6,:] = sl_esp[:, s, line, :]
               
                yield mrd.StreamItem.Acquisition(acq)
 
    with mrd.BinaryMrdWriter(output) as w:
        w.write_header(h)
        w.write_data(generate_data())

def exportMAT(input, output, out_field, out_field_k):
    images = []

    with mrd.BinaryMrdReader(input) as reader:
        header = reader.read_header()
        assert header is not None, "No header found in reconstructed file"

        for item in reader.read_data():
            if isinstance(item, mrd.StreamItem.ImageFloat):
                images.append(item.value)
            elif isinstance(item, mrd.StreamItem.ImageComplexFloat):
                images.append(item.value)

    print("Number of images:", len(images))

    img_ori = None
    img_den = None

    for i, img in enumerate(images):
        series = getattr(img.head, "image_series_index", None)
        print(i, "image_series_index:", series, "shape:", img.data.shape)

        if series == 100:
            img_ori = img.data[0]
        elif series == 101:
            img_den = img.data[0]

    if img_den is None:
        raise RuntimeError("No denoised image found. Expected image_series_index 101.")

    rawData = sio.loadmat(output)

    rawData[out_field] = img_den
    rawData[out_field_k] = np.fft.fftshift(
        np.fft.fftn(np.fft.fftshift(img_den))
    )

    if img_ori is not None:
        rawData[out_field + "_ori"] = img_ori
        rawData[out_field_k + "_ori"] = np.fft.fftshift(
            np.fft.fftn(np.fft.fftshift(img_ori))
        )

    rawData = {
        k: v for k, v in rawData.items()
        if not k.startswith("__")
    }

    sio.savemat(output, rawData)

    return img_den
 
################################ Plot Data ############################################

def plot_slider_1(img3D,
                       title='Image',
                       vmax_init=1.0):

    nSlice = img3D.shape[0] // 2
    max_val = np.max(img3D)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.subplots_adjust(bottom=0.3)

    im = ax.imshow(img3D[nSlice, :, :],
                   cmap='gray',
                   vmin=0,
                   vmax=vmax_init * max_val)

    ax.axis('off')
    ax.set_title(title)

    # Slider slice
    ax_slice = plt.axes([0.2, 0.12, 0.6, 0.03])
    slider_slice = Slider(ax_slice, 'Slice', 0, img3D.shape[0]-1,
                          valinit=nSlice, valfmt='%d')

    # Slider vmax
    ax_vmax = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider_vmax = Slider(ax_vmax, 'Vmax', 0.01, 1.5,
                         valinit=vmax_init)

    def update(val):
        idx = int(slider_slice.val)
        v = slider_vmax.val

        im.set_data(img3D[idx, :, :])
        im.set_clim(0, v * max_val)

        fig.canvas.draw_idle()

    slider_slice.on_changed(update)
    slider_vmax.on_changed(update)

    plt.show()

def plot_slider_2(img3D_1, img3D_2,
                              titles=('Image 1', 'Image 2'),
                              vmax_init=1.0):

    if img3D_1.shape[0] != img3D_2.shape[0]:
        raise ValueError("Las dos imágenes deben tener el mismo número de slices.")

    nSlice = img3D_1.shape[0] // 2

    max1 = np.max(img3D_1)
    max2 = np.max(img3D_2)

    vmax1 = vmax_init * max1
    vmax2 = vmax_init * max2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.3)

    im1 = ax1.imshow(img3D_1[nSlice, :, :], cmap='gray', vmin=0, vmax=vmax1)
    ax1.axis('off')
    ax1.set_title(titles[0])

    im2 = ax2.imshow(img3D_2[nSlice, :, :], cmap='gray', vmin=0, vmax=vmax2)
    ax2.axis('off')
    ax2.set_title(titles[1])

    # Slider slice
    ax_slice = plt.axes([0.25, 0.12, 0.5, 0.03])
    slider_slice = Slider(ax_slice, 'Slice', 0, img3D_1.shape[0]-1,
                          valinit=nSlice, valfmt='%d')

    # Slider vmax
    ax_vmax = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider_vmax = Slider(ax_vmax, 'Vmax', 0.01, 1.5,
                         valinit=vmax_init)

    def update(val):
        idx = int(slider_slice.val)
        v = slider_vmax.val

        im1.set_data(img3D_1[idx, :, :])
        im2.set_data(img3D_2[idx, :, :])

        im1.set_clim(0, v * max1)
        im2.set_clim(0, v * max2)

        fig.canvas.draw_idle()

    slider_slice.on_changed(update)
    slider_vmax.on_changed(update)

    plt.show()

def plot_slider_4(img1, img2, img3, img4,
                              titles=('Img1', 'Img2', 'Img3', 'Img4'),
                              vmax_init=1.0):

    imgs = [img1, img2, img3, img4]

    # Check slices
    n_slices = [img.shape[0] for img in imgs]
    if len(set(n_slices)) != 1:
        raise ValueError("Todas las imágenes deben tener el mismo número de slices.")

    nSlice = n_slices[0] // 2
    max_vals = [np.max(img) for img in imgs]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    plt.subplots_adjust(bottom=0.3)

    ims = []

    for i, ax in enumerate(axes):
        im = ax.imshow(imgs[i][nSlice, :, :],
                       cmap='gray',
                       vmin=0,
                       vmax=vmax_init * max_vals[i])
        ax.axis('off')
        ax.set_title(titles[i])
        ims.append(im)

    # Slider slice
    ax_slice = plt.axes([0.25, 0.12, 0.5, 0.03])
    slider_slice = Slider(ax_slice, 'Slice', 0, n_slices[0]-1,
                          valinit=nSlice, valfmt='%d')

    # Slider vmax
    ax_vmax = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider_vmax = Slider(ax_vmax, 'Vmax', 0.01, 1.5,
                         valinit=vmax_init)

    def update(val):
        idx = int(slider_slice.val)
        v = slider_vmax.val

        for i in range(4):
            ims[i].set_data(imgs[i][idx, :, :])
            ims[i].set_clim(0, v * max_vals[i])

        fig.canvas.draw_idle()

    slider_slice.on_changed(update)
    slider_vmax.on_changed(update)

    plt.show()

def plot_slider_3(img3D_1, img3D_2, img3D_3,
                  titles=('Image 1', 'Image 2', 'Image 3'),
                  vmax_init=1.0):

    if not (img3D_1.shape[0] == img3D_2.shape[0] == img3D_3.shape[0]):
        raise ValueError("Las tres imágenes deben tener el mismo número de slices.")

    nSlice = img3D_1.shape[0] // 2

    max1 = np.max(img3D_1)
    max2 = np.max(img3D_2)
    max3 = np.max(img3D_3)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.3)

    im1 = ax1.imshow(img3D_1[nSlice], cmap='gray',
                     vmin=0, vmax=vmax_init * max1)
    ax1.axis('off')
    ax1.set_title(titles[0])

    im2 = ax2.imshow(img3D_2[nSlice], cmap='gray',
                     vmin=0, vmax=vmax_init * max2)
    ax2.axis('off')
    ax2.set_title(titles[1])

    im3 = ax3.imshow(img3D_3[nSlice], cmap='gray',
                     vmin=0, vmax=vmax_init * max3)
    ax3.axis('off')
    ax3.set_title(titles[2])

    # Slider slice
    ax_slice = plt.axes([0.25, 0.12, 0.5, 0.03])
    slider_slice = Slider(
        ax_slice, 'Slice',
        0, img3D_1.shape[0] - 1,
        valinit=nSlice,
        valfmt='%d'
    )

    # Slider vmax
    ax_vmax = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider_vmax = Slider(
        ax_vmax, 'Vmax',
        0.01, 1.5,
        valinit=vmax_init
    )

    def update(val):
        idx = int(slider_slice.val)
        v = slider_vmax.val

        im1.set_data(img3D_1[idx])
        im2.set_data(img3D_2[idx])
        im3.set_data(img3D_3[idx])

        im1.set_clim(0, v * max1)
        im2.set_clim(0, v * max2)
        im3.set_clim(0, v * max3)

        fig.canvas.draw_idle()

    slider_slice.on_changed(update)
    slider_vmax.on_changed(update)

    plt.show()