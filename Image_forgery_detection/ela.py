import os, sys
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

# converts input image to ela applied image
def convert_to_ela_image(path, quality):

    original_image = Image.open(path).convert("RGB")

    # resaving input image at the desired quality
    resaved_file_name = "resaved_image.jpg"  # predefined filename for resaved image
    original_image.save(resaved_file_name, "JPEG", quality=quality)
    resaved_image = Image.open(resaved_file_name)

    # pixel difference between original and resaved image
    ela_image = ImageChops.difference(original_image, resaved_image)

    # scaling factors are calculated from pixel extremas
    extrema = ela_image.getextrema()
    max_difference = max([pix[1] for pix in extrema])
    if max_difference == 0:
        max_difference = 1
    fact= 255.0
    scale = fact / max_difference #350.0 #255.0 per Shivang Gupta

    # enhancing elaimage to brighten the pixels
    #ela_image = ImageEnhance.Brightness(ela_image).enhance(scale) 
    #scale=1.0 #only for pdfs
    
    #enhance the sharpness of the ela_image in order to highlight the borders and the compression effects
    ela_image = ImageEnhance.Sharpness(ela_image).enhance(scale) #Nuovo

    ela_image.save("ela_image.png")
    return fact,ela_image

#converts the segmented image to an image in black and white
def convert_to_bn_image(segm_image, n_pix_h, n_pix_v):
    n_px=n_pix_h*n_pix_v
    for y in range(n_px):
        segm_image[y]=segm_image[y]*255
    
    segm_image_mat=segm_image.astype(np.uint8).reshape(n_pix_h, n_pix_v, order='C') #astype(np.uint8) necessary to produce a true B/N image
    
    segm_test_image = Image.fromarray(segm_image_mat, mode='L')  # 'L' mode for 8-bit grayscale

    # Save or display the image
    segm_test_image.save("bw_image.png")
    #image.show()
    return segm_test_image
    
    


if __name__ == "__main__":
    file_path = sys.argv[1]
    quality = int(sys.argv[2])
    convert_to_ela_image(file_path, quality).show()
