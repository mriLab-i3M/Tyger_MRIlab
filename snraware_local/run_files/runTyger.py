
import io
import time
import subprocess
import os
import sys
import argparse

this_file_path = os.path.abspath(__file__)
project_dir = os.path.abspath(os.path.join(os.path.dirname(this_file_path), ".."))
sys.path.append(project_dir)

from recon_scripts.fromMATtoMRD3D_RARE import matToMRD
from recon_scripts.fromMRDtoMAT3D import export


def main():
    ap = argparse.ArgumentParser(
        description="Run full pipeline using Tyger (MAT->MRD->Tyger(exec)->MRD->MAT)"
    )
    ap.add_argument("--in_mat", required=True, help="Input MAT path (PC control)")
    ap.add_argument("--out_mat", required=True, help="Output MAT path (PC control)")
    ap.add_argument("--yml", required=True, help="Tyger YAML file (tyger_example.yml)")
    ap.add_argument("--out_field", default="image3D_denoised", help="Field name to store output image in MAT")
    ap.add_argument("--input_field", default="", help="Optional kSpace field in MAT (default uses sampledCartesian)")
    args = ap.parse_args()

    # Crear carpeta destino si hace falta
    out_dir = os.path.dirname(os.path.abspath(args.out_mat))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Comprobaciones rápidas
    if not os.path.exists(args.in_mat):
        raise FileNotFoundError(f"No existe in_mat: {args.in_mat}")
    if not os.path.exists(args.yml):
        raise FileNotFoundError(f"No existe yml: {args.yml}")

    sys.stderr.write("Running Tyger pipeline (MAT -> MRD -> Tyger(exec) -> MRD -> MAT)\n")
    sys.stderr.flush()

    start = time.time()

    # 1) MAT -> MRD (bytes)
    mrd_buffer = io.BytesIO()
    matToMRD(input=args.in_mat, output_file=mrd_buffer, input_field=args.input_field)
    mrd_buffer.seek(0)
    in_bytes = mrd_buffer.getvalue()

    # 2) Tyger exec (stdin=MRD bytes, stdout=MRD bytes)
    p = subprocess.run(
        ["tyger", "run", "exec", "-f", args.yml],
        input=in_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Logs de tyger/contenedor a stderr
    if p.stderr:
        sys.stderr.write(p.stderr.decode("utf-8", errors="replace"))
        sys.stderr.flush()

    if p.returncode != 0:
        raise RuntimeError(f"Tyger falló con returncode={p.returncode}")

    out_bytes = p.stdout

    # 3) MRD -> MAT (nuevo archivo)
    out_buf = io.BytesIO(out_bytes)
    out_buf.seek(0)
    export(out_buf, mat_in_path=args.in_mat, mat_out_path=args.out_mat, out_field=args.out_field)

    dt = time.time() - start
    sys.stderr.write(f"Done. Total time: {dt:.2f} s\nSaved: {args.out_mat}\n")
    sys.stderr.flush()


if __name__ == "__main__":
    main()