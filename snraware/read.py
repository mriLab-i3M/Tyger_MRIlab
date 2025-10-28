
import mrd
import numpy as np 
import matplotlib.pyplot as plt

def load_mrd_images(images_file):
    images = []
    with mrd.BinaryMrdReader(images_file) as reader:
        header = reader.read_header()
        assert header is not None, "No header found in reconstructed file"

        for item in reader.read_data():
            if isinstance(item, mrd.StreamItem.ImageFloat):
                images.append(item.value)
    return images

mrd_path = '/home/teresa/marcos_tyger/tyger_repo_may/Tyger_MRIlab/snraware/phantom_ia.mrd'
img = load_mrd_images(mrd_path)
mat = img[0].data
print(mat.shape)

for i in range(36):
    plt.figure(figsize=(6,6))
    plt.imshow(mat[0,i,:,:], cmap='gray')
    plt.axis('off')

    # Guardar la figura
    title = str(i) + "test3.png"
    # plt.savefig(title, bbox_inches='tight', dpi=300)
    plt.show()