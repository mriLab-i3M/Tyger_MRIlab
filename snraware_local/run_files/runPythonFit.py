import io
import time
import yaml
import subprocess
import os
import sys

this_file_path = os.path.abspath(__file__)
project_dir = os.path.abspath(os.path.join(os.path.dirname(this_file_path), ".."))
sys.path.append(project_dir)

from recon_scripts.fromMATtoMRD3D_RARE import matToMRD
from recon_scripts.fromMRDtoMAT3D import export


def main():
    # -----------------------------
    # Paths (WSL)
    # -----------------------------
    raw_mat = "/home/user/Documentos/SNRaware/dev/RarePyPulseq.2026.01.23.15.23.52.059.mat"
    out_mat = "/home/user/Documentos/SNRaware/data/snraware/out/case_output_6_6/RarePyPulseq.2026.01.23.15.23.52.059.mat"
    work_dir = "/home/user/Documentos/SNRaware/data/snraware/work/case_output_6_6"
    yml_path = "/home/user/Documentos/SNRaware/dev/snraware_pipeline/yml_files/python_example.yml"

    os.makedirs(os.path.dirname(out_mat), exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    out_field = "image3D_denoised"
    input_field = ""

    print("Running pipeline (MAT -> MRD -> recon+SNRAware -> MRD -> MAT)")
    start_time = time.time()

    # -----------------------------
    # 1) MAT -> MRD (buffer)
    # -----------------------------
    mrd_buffer = io.BytesIO()
    matToMRD(input=raw_mat, output_file=mrd_buffer, input_field=input_field)
    mrd_buffer.seek(0)
    mrd_in_bytes = mrd_buffer.getvalue()

    # -----------------------------
    # 2) stream_recon_RARE.py (stdin/stdout)
    # -----------------------------
    with open(yml_path, "r") as f:
        config = yaml.safe_load(f)
    args = config["args"]

    stream_script = "/home/user/Documentos/SNRaware/dev/snraware_pipeline/recon_scripts/stream_recon_RARE.py"

    p = subprocess.Popen(
        [sys.executable, stream_script, *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # logs aquí
    )

    mrd_out_bytes, stderr_bytes = p.communicate(input=mrd_in_bytes)

    if stderr_bytes:
        # logs (p.ej. de run_inference.py) -> consola
        sys.stderr.write(stderr_bytes.decode("utf-8", errors="replace"))
        sys.stderr.flush()

    if p.returncode != 0:
        raise RuntimeError(f"stream_recon_RARE.py falló con returncode={p.returncode}")

    # -----------------------------
    # 3) MRD -> MAT (nuevo archivo)
    # -----------------------------
    mrd_out_buffer = io.BytesIO(mrd_out_bytes)
    export(mrd_out_buffer, mat_in_path=raw_mat, mat_out_path=out_mat, out_field=out_field)

    dt = time.time() - start_time
    print(f"Done. Total time: {dt:.2f} s")
    print("Saved:", out_mat)


if __name__ == "__main__":
    main()