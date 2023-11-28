import os
from tqdm import tqdm
from utils import btgraham_processing


if __name__ == "__main__":
    img_dir = "../../data/APTOS_2019/train_images"
    prep_img_dir = "../../data/APTOS_2019/preprocessed_train_images"
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
