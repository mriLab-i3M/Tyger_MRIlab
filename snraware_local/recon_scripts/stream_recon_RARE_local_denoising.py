import argparse
import os
import sys
import tempfile
import subprocess
import numpy as np
import mrd


def _get_user_double(header, name: str, default=None):
    up = getattr(header, "user_parameters", None)
    doubles = getattr(up, "user_parameter_double", None) if up is not None else None
    if doubles is None:
        return default
    for p in doubles:
        if getattr(p, "name", None) == name:
            return float(getattr(p, "value", default))
    return default


def _read_header_strings(header):
    axesOrientation = None
    dfov = None
    up = getattr(header, "user_parameters", None)
    strings = getattr(up, "user_parameter_string", None) if up is not None else None
    if strings is not None:
        for p in strings:
            if p.name == "axesOrientation":
                axesOrientation = list(map(int, p.value.split(",")))
            if p.name == "dfov":
                dfov = list(map(float, p.value.split(",")))
    return axesOrientation, dfov


def _ifft3_cartesian(kspace_sl_ph_rd: np.ndarray) -> np.ndarray:
    # kspace expected (sl, ph, rd)
    return np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kspace_sl_ph_rd)))


def _run_snraware_inference(td: str, snraware_repo: str, model: str, batch_size: int):
    pts = os.path.join(snraware_repo, model, f"snraware_{model}_model.pts")
    yml = os.path.join(snraware_repo, model, f"snraware_{model}_model.yaml")
    run_py = os.path.join(snraware_repo, "src", "snraware", "projects", "mri", "denoising", "run_inference.py")

    cmd = [
        sys.executable, run_py,
        "--input_dir", td,
        "--output_dir", td,
        "--saved_model_path", pts,
        "--saved_config_path", yml,
        "--batch_size", str(batch_size),
        "--input_fname", "input",
        "--gmap_fname", "gmap",
    ]

    # IMPORTANTÍSIMO: capturamos stdout/stderr para NO contaminar stdout (MRD binario)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.stdout:
        sys.stderr.write(p.stdout)
        sys.stderr.flush()
    if p.returncode != 0:
        raise RuntimeError(f"SNRAware run_inference.py falló con code={p.returncode}")


def _snraware_denoise_complex(img_sl_ph_rd: np.ndarray,
                             noise_std: float,
                             par_fourier_fraction: float,
                             model: str,
                             snraware_repo: str,
                             batch_size: int) -> np.ndarray:
    """
    img_sl_ph_rd: complex (sl, ph, rd)
    Returns: complex (sl, ph, rd) denoised
    """

    # SNRAware espera (H,W,F). Aquí: H=ph, W=rd, F=sl
    img_hwf = np.transpose(img_sl_ph_rd, (1, 2, 0)).astype(np.complex64, copy=False)
    H, W, F = img_hwf.shape

    # FIEL al experto: N_eff = parFourierFraction * prod(image3D.shape)
    N_recon = int(np.prod(img_sl_ph_rd.shape))
    N_eff = float(par_fourier_fraction) * float(N_recon)

    sigma = float(noise_std) / np.sqrt(2.0) / np.sqrt(N_eff)
    if not np.isfinite(sigma) or sigma <= 0:
        raise RuntimeError(f"sigma inválida: {sigma} (noise_std={noise_std}, pF={par_fourier_fraction}, N={N_recon})")

    x_snr = img_hwf / sigma
    gmap = np.ones((H, W), dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="snraware_io_") as td:
        np.save(os.path.join(td, "input_real.npy"), np.real(x_snr).astype(np.float32))
        np.save(os.path.join(td, "input_imag.npy"), np.imag(x_snr).astype(np.float32))
        np.save(os.path.join(td, "gmap.npy"), gmap)

        _run_snraware_inference(td, snraware_repo=snraware_repo, model=model, batch_size=batch_size)

        out_r = np.load(os.path.join(td, "output_real.npy")).astype(np.float32)
        out_i = np.load(os.path.join(td, "output_imag.npy")).astype(np.float32)

        den_hwf = (out_r + 1j * out_i) * sigma
        den_sl_ph_rd = np.transpose(den_hwf, (2, 0, 1))  # back (sl, ph, rd)

    return den_sl_ph_rd.astype(np.complex64, copy=False)


def main():
    ap = argparse.ArgumentParser()

    # Mantener compatibilidad con runPythonFit.py / YAML
    ap.add_argument("--recon", default="fft", choices=["fft"], help="Recon method (only fft supported here)")

    ap.add_argument("-i", "--input", default="-", help="MRD input (default stdin)")
    ap.add_argument("-o", "--output", default="-", help="MRD output (default stdout)")

    ap.add_argument("--denoise", action="store_true", help="Apply SNRAware denoising after recon")
    ap.add_argument("--snraware_repo", required=True)
    ap.add_argument("--snraware_model", default="small", choices=["small", "medium", "large"])
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()

    inp = sys.stdin.buffer if args.input == "-" else open(args.input, "rb")
    out = sys.stdout.buffer if args.output == "-" else open(args.output, "wb")

    with mrd.BinaryMrdReader(inp) as r, mrd.BinaryMrdWriter(out) as w:
        header = r.read_header()

        parFourierFraction = _get_user_double(header, "parFourierFraction", default=1.0)
        noise_std = _get_user_double(header, "noise_std", default=None)
        if noise_std is None:
            raise RuntimeError("No encuentro 'noise_std' en header.user_parameters.user_parameter_double")

        axesOrientation, _dfov = _read_header_strings(header)
        if axesOrientation is None:
            raise RuntimeError("No encuentro 'axesOrientation' en header.user_parameters.user_parameter_string")

        enc0 = header.encoding[0]
        eNx = int(enc0.encoded_space.matrix_size.x)
        eNy = int(enc0.encoded_space.matrix_size.y)
        eNz = int(enc0.encoded_space.matrix_size.z)

        # Derivar n_rd, n_ph, n_sl desde axesOrientation (rd,ph,sl)->(x,y,z)
        n_rd = eNx if axesOrientation[0] == 0 else (eNy if axesOrientation[0] == 1 else eNz)
        n_ph = eNx if axesOrientation[1] == 0 else (eNy if axesOrientation[1] == 1 else eNz)
        n_sl = eNx if axesOrientation[2] == 0 else (eNy if axesOrientation[2] == 1 else eNz)

        kspace = np.zeros((n_sl, n_ph, n_rd), dtype=np.complex64)

        # Leer adquisiciones (en tu writer: idx cuelga de head)
        for item in r.read_data():
            if not isinstance(item, mrd.StreamItem.Acquisition):
                continue
            acq = item.value
            line = int(acq.head.idx.kspace_encode_step_1)
            sl = int(acq.head.idx.kspace_encode_step_2)
            kspace[sl, line, :] = acq.data[0].astype(np.complex64, copy=False)

        # Recon (solo fft)
        img_sl_ph_rd = _ifft3_cartesian(kspace)

        # Denoise
        if args.denoise:
            img_sl_ph_rd = _snraware_denoise_complex(
                img_sl_ph_rd=img_sl_ph_rd,
                noise_std=noise_std,
                par_fourier_fraction=parFourierFraction,
                model=args.snraware_model,
                snraware_repo=args.snraware_repo,
                batch_size=args.batch_size,
            )

        # Escribimos imagen magnitude como ImageFloat 4D: (1, sl, ph, rd)
        img_mag = np.abs(img_sl_ph_rd).astype(np.float32)
        img_mag4 = img_mag[np.newaxis, ...]

        w.write_header(header)

        im_head = mrd.ImageHeader(image_type=mrd.ImageType.MAGNITUDE)
        img_mrd = mrd.Image(head=im_head, data=img_mag4)
        w.write_data([mrd.StreamItem.ImageFloat(img_mrd)])


if __name__ == "__main__":
    main()