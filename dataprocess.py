import scipy
import os
from PIL import Image
import pathlib

DATA_PATH = "/home/kgw/sr-heart/Cardiac_MRI_Segmentation/bicubic/"
DIR_NAME = ["LV", "RV"]
SAVE_PATH = "data"

pathlib.Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

for dir in DIR_NAME:
    image_file_list = [file for file in os.listdir(os.path.join(DATA_PATH, dir)) if file.endswith("image.mat")]
    mask_file_list = [img_file.replace("image.mat", "mask.mat") for img_file in image_file_list]

    for image_file_name, mask_file_name in zip(image_file_list, mask_file_list):
        image_path = os.path.join(os.path.join(DATA_PATH, dir, image_file_name))
        mask_path = os.path.join(os.path.join(DATA_PATH, dir, mask_file_name))


        image = scipy.io.loadmat(image_path)["image"]
        mask = scipy.io.loadmat(mask_path)["mask"]

        for idx in range(image.shape[0]):
            new_image_path = os.path.join(SAVE_PATH, image_file_name.split(".")[0] + f"_{idx}.png")
            new_mask_path = os.path.join(SAVE_PATH, mask_file_name.split(".")[0] + f"_{idx}.png")

            PIL_image = Image.fromarray(image[idx,:,:].astype('uint8'), 'L')
            PIL_image.save(new_image_path)

            """
            PIL_image = Image.fromarray(mask[idx,:, :].astype("uint8"), "L")
            PIL_image.save(new_mask_path)
            """


