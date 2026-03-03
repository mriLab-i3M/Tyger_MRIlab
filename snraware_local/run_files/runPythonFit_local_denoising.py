"Si quiero que se ejecute correctamente, hay que activar el entorno virtual que toca que es: source /home/miguel/dev/snraware_pipeline/run_files/snraware/bin/activate"

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
    raw_mat = "/home/miguel/data/snraware/in/case01/output_6_6.mat"
    out_mat = "/home/miguel/data/snraware/out/case01/output_6_6.mat"
    work_dir = "/home/miguel/data/snraware/work/case01"
    yml_path = "/home/miguel/dev/snraware_pipeline/yml_files/python_example.yml"

    os.makedirs(os.path.dirname(out_mat), exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    out_field = "image3D_denoised"  # campo que se añadirá al MAT nuevo
    input_field = ""                # si se quiere usar un kSpace concreto del MAT, se pone su nombre aquí

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
    # 2) stream_recon_RARE.py (subprocess, stdin/stdout)
    # -----------------------------
    with open(yml_path, "r") as f:
        config = yaml.safe_load(f)
    args = config["args"]

    stream_script = "/home/miguel/dev/snraware_pipeline/recon_scripts/stream_recon_RARE.py"

    p = subprocess.Popen(
        [sys.executable, stream_script, *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    mrd_out_bytes, _ = p.communicate(input=mrd_in_bytes)
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