import numpy as np
import scipy.io as sio
import mrd


def export(mrd_input, mat_in_path: str, mat_out_path: str, out_field: str = "image3D_denoised"):
    """
    Lee un MRD (stream) producido por stream_recon_RARE.py y escribe un MAT
    con el campo out_field.
    """

    # Cargamos MAT original para copiar todo y añadir el campo de salida
    mat = sio.loadmat(mat_in_path)

    # Abrimos MRD reader (admite file-like, bytes buffer, etc.)
    with mrd.BinaryMrdReader(mrd_input) as r:
        header = r.read_header()

        img_data = None

        # IMPORTANTE: consumir el iterador completo para no disparar ProtocolError
        it = r.read_data()
        for item in it:
            # Nos interesan items de tipo imagen
            if isinstance(item, mrd.StreamItem.ImageFloat):
                img = item.value  # mrd.Image
                img_data = np.asarray(img.data)  # normalmente 4D (1, sl, ph, rd)
                # NO hacemos break: seguimos consumiendo hasta el final
                # para cerrar el reader sin error

        if img_data is None:
            raise RuntimeError("No se encontró ninguna ImageFloat en el MRD de salida.")

    # Convertimos de 4D a 3D si venía con batch=1
    # stream_recon_RARE escribe (1, sl, ph, rd)
    if img_data.ndim == 4 and img_data.shape[0] == 1:
        img_data = img_data[0]  # (sl, ph, rd)

    # Guardamos como float32 por consistencia (es magnitud)
    mat[out_field] = img_data.astype(np.float32, copy=False)

    # Guardamos MAT de salida (nuevo archivo)
    sio.savemat(mat_out_path, mat)
    print(f"Export OK: '{out_field}' guardado en {mat_out_path}")