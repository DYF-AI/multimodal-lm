import os
import fitz
import numpy as np
from tqdm import tqdm
from PIL import Image



def load_image(image_path:str, return_chw:bool=True, size:tuple=None):
    image = Image.open(image_path).convert("RGB")
    if size is not None:
        image = image.resize(size)   # resize image
    image = np.asarray(image)
    image = image[:, :, ::-1]    # flip color channels from RGB to BGR
    w, h = image.shape[1], image.shape[0] # update size after resize
    if return_chw:
        image = image.transpose(2, 0, 1)
    return image, (w, h)

def pdf_to_images(pdf_path:str, output_path:str, high_quality=True) -> bool:
    doc = fitz.open(pdf_path)
    base_name = os.path.basename(pdf_path).split(".")[0]
    try:
        for index, p in enumerate(doc):
            pix = p.get_pixmap()
            if high_quality:
                zoom_x, zoom_y = 1.5, 1.5
                mat = fitz.Matrix(zoom_x, zoom_y)
                pix = p.get_pixmap(matrix=mat)
            save_image_path = os.path.join(output_path, f"{base_name}_{index}.jpg")
            pix.save(save_image_path)
    except Exception as e:
        return False
    return True