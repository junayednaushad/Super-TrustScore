import os
from tqdm import tqdm
from utils import btgraham_processing

if __name__ == "__main__":
    train_dir = "../../data/EyePACS/train"
    train_extract_dir = "../../data/EyePACS/preprocessed_train_images"
    train_files = os.listdir(train_dir)
    for f in tqdm(train_files):
        btgraham_processing(
            os.path.join(train_dir, f),
            train_extract_dir,
            target_pixels=300,
            crop_to_radius=True,
        )

    test_dir = "../../data/EyePACS/test"
    test_extract_dir = "../../data/EyePACS/preprocessed_test_images"
    test_files = os.listdir(test_dir)
    for f in tqdm(test_files):
        btgraham_processing(
            os.path.join(test_dir, f),
            test_extract_dir,
            target_pixels=300,
            crop_to_radius=True,
        )
