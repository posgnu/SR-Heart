import scipy
import os
from PIL import Image
import pathlib
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
import torch

DATA_PATH = "/home/kgw/sr-heart/Cardiac_MRI_Segmentation/bicubic/"
DIR_NAME = ["LV", "RV"]
SAVE_PATH = "data"

pathlib.Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.path.join(SAVE_PATH, DIR_NAME[0])).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.path.join(SAVE_PATH, DIR_NAME[1])).mkdir(parents=True, exist_ok=True)

for dir in DIR_NAME:
    image_file_list = [file for file in os.listdir(os.path.join(DATA_PATH, dir)) if file.endswith("image.mat")]
    mask_file_list = [img_file.replace("image.mat", "mask.mat") for img_file in image_file_list]

    for image_file_name, mask_file_name in zip(image_file_list, mask_file_list):
        image_path = os.path.join(os.path.join(DATA_PATH, dir, image_file_name))
        mask_path = os.path.join(os.path.join(DATA_PATH, dir, mask_file_name))


        image = scipy.io.loadmat(image_path)["image"]
        mask = scipy.io.loadmat(mask_path)["mask"]

        for idx in range(image.shape[0]):
            new_image_path = os.path.join(SAVE_PATH, dir, image_file_name.split(".")[0] + f"_{idx}.png")
            new_mask_path = os.path.join(SAVE_PATH, mask_file_name.split(".")[0] + f"_{idx}.png")

            single_channel_image = image[idx,:,:]
            single_channel_mask = mask[idx,:, :]


            PIL_image = Image.fromarray(single_channel_image.astype('uint8'), 'L')
            PIL_image = PIL_image.convert("RGB")
            tensor_image = pil_to_tensor(PIL_image)
            masked_image = draw_segmentation_masks(tensor_image, torch.from_numpy(single_channel_mask.astype("bool")), alpha=0.5, colors="red")
            np_masked_image = np.array(masked_image.permute(1, 2, 0)).astype('uint8')
            PIL_masked_image = Image.fromarray(np_masked_image, "RGB")

            PIL_masked_image.save(new_image_path)

            """
            PIL_image = Image.fromarray(mask[idx,:, :].astype("uint8"), "L")
            PIL_image.save(new_mask_path)
            """


