import os
from tqdm import tqdm
import cv2
import numpy as np


def shade_of_gray_cc(img, power=6, gamma=None):
    """
    Source: https://www.kaggle.com/code/apacheco/shades-of-gray-color-constancy

    Parameters
    ----------
    img : numpy.ndarray
        The original image with format (H, W, C)
    power : int
        The degree of norm, 6 is used in reference paper
    gamma : float
        the value of gamma correction, 2.2 is used in reference paper

    Returns
    -------
    numpy.ndarray
        Color corrected image
    """
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype("uint8")
        look_up_table = np.ones((256, 1), dtype="uint8") * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i / 255, 1 / gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype("float32")
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0, 1)), 1 / power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    # Andrew Anikin suggestion
    img = np.clip(img, a_min=0, a_max=255)

    return img.astype(img_dtype)


def apply_cc(img_paths, output_folder_path, resize=False):
    """
    Applies the Shades of Gray algorithm and resizes images while maintaining the same aspect ratio
    and ensuring that the smaller dimension is 224

    Parameters
    ----------
    img_paths : list of str
        Paths to the image files to be transformed
    output_folder_path : str
        Path to folder where transformed images will be saved
    resize : bool
        Whether or not images should be resized
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for img_path in tqdm(img_paths):
        img_name = img_path.split("/")[-1]
        img_ = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if resize:
            H, W, _ = img_.shape
            min_dim = np.argmin([H, W])
            scale = img_.shape[min_dim] / 224  # smaller dimension should be 224
            if min_dim == 0:  # H is smaller
                new_size = (
                    max(int(W / scale), 224),
                    224,
                )  # cv2 uses WxH for resize function, max() ensures that size is at least 224x224
            else:
                new_size = (224, max(int(H / scale), 224))
            img_ = cv2.resize(img_, new_size, cv2.INTER_AREA)
        np_img = shade_of_gray_cc(img_)
        cv2.imwrite(
            os.path.join(output_folder_path, img_name.split(".")[0] + ".jpg"), np_img
        )
