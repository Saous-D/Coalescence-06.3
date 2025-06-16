import os, sys
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

# converts input image to ela applied image
from PIL import Image, ImageChops, ImageEnhance
import io

def convert_to_ela_image(path_or_bytes, quality):
    """
    Convertit une image en ELA (Error Level Analysis) sans écrire sur disque.

    path_or_bytes : chemin fichier (str) ou bytes de l'image
    quality : qualité JPEG pour la recompression
    """

    # Charger l'image d'origine
    if isinstance(path_or_bytes, str):
        original_image = Image.open(path_or_bytes).convert("RGB")
    else:
        original_image = Image.open(io.BytesIO(path_or_bytes)).convert("RGB")

    # Sauvegarder l'image en mémoire avec la compression JPEG souhaitée
    compressed_io = io.BytesIO()
    original_image.save(compressed_io, "JPEG", quality=quality)
    compressed_io.seek(0)
    resaved_image = Image.open(compressed_io)

    # Calculer la différence
    ela_image = ImageChops.difference(original_image, resaved_image)

    # Calculer l'échelle
    extrema = ela_image.getextrema()
    max_difference = max([pix[1] for pix in extrema])
    print("Max difference detected:", max_difference)

    if max_difference < 5:
        scale = 40.0
    else:
        scale = 255.0 / max_difference

    # Améliorer luminosité, contraste et netteté
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    ela_image = ImageEnhance.Contrast(ela_image).enhance(1.5)
    ela_image = ImageEnhance.Sharpness(ela_image).enhance(2.0)

    return 255.0, ela_image


#from PIL import Image
import numpy as np

def convert_to_bn_image(image_array, n_pix_h, n_pix_v):
    """
    Convertit un tableau de pixels (1D) en image noir et blanc PIL.
    """

    # S’assurer que le tableau est de type float entre 0 et 1
    image_array = np.clip(image_array, 0.0, 1.0)

    # Mise à l'échelle entre 0 et 255
    segm_image = (image_array * 255).astype(np.uint8)

    try:
        # Tentative de reshape selon les dimensions de l'image initiale
        segm_image = segm_image.reshape((n_pix_h, n_pix_v))
    except ValueError:
        # Si le reshape échoue, on tente de reconstruire une image carrée
        print("[⚠️] Dimensions incompatibles, reshape carré automatique.")
        size = int(np.sqrt(segm_image.shape[0]))
        segm_image = segm_image[:size*size].reshape((size, size))

    # Création d'une image PIL en mode "L" (niveaux de gris)
    img = Image.fromarray(segm_image, mode="L")

    return img

    


if __name__ == "__main__":
    file_path = sys.argv[1]
    quality = int(sys.argv[2])
    convert_to_ela_image(file_path, quality).show()
