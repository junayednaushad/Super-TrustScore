import os
from tqdm import tqdm
from utils import btgraham_processing


if __name__ == "__main__":
    img_dir = "../../data/Messidor_2/IMAGES"
    prep_img_dir = "../../data/Messidor_2/preprocessed_images"
    if not os.path.exists(prep_img_dir):
        os.makedirs(prep_img_dir)
    filenames = os.listdir(img_dir)
    for f in tqdm(filenames):
        btgraham_processing(
            os.path.join(img_dir, f),
            prep_img_dir,
            target_pixels=300,
            crop_to_radius=True,
        )
