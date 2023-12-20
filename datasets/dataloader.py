import os
import random
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from utils import parse_dataset_txt, read_image

###-----[Booster]-----###
rgb_str = "camera_00"
disp_str = "disp_00.npy"
mask_str = "mask_00.png"
mask_c_str = "mask_cat.png"


class Trans10KLoader(Dataset):
    def __init__(self, dataset_dir, dataset_txt, transform):
        self.dataset_dir = dataset_dir
        self.transform = transform
        dataset_dict = parse_dataset_txt(dataset_txt)

        self.images_names = dataset_dict["basenames"]
        self.ground_truth_names = dataset_dict["gt_paths"]


    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.dataset_dir, self.images_names[idx])
        disp_path = os.path.join(self.dataset_dir, self.ground_truth_names[idx])
        
        # Read all the images in the folder and stack them to form the batch.
        rgb_image = read_image(rgb_path) # [0,1] rgb hxwxc image
        ground_truth = np.load(disp_path).astype(np.float32)
        ground_truth = cv2.resize(ground_truth, (rgb_image.shape[1], rgb_image.shape[0]), cv2.INTER_NEAREST)

        transformed_dict = self.transform({"image": rgb_image, "depth": ground_truth})        
        rgb_image = transformed_dict["image"]
        ground_truth = transformed_dict["depth"]
        rgb_image = torch.from_numpy(rgb_image)
        ground_truth = torch.from_numpy(ground_truth)

        return rgb_image, ground_truth, rgb_path

class MSDLoader(Trans10KLoader):
    pass