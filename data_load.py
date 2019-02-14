import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torchvision

class FacialKeypointsDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform = None):
        self.keypoints_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.keypoints_df)
    def __getitem__(self, index):
        image_name = os.path.join(self.root_dir, self.keypoints_df.iloc[index, 0])
        image = plt.imread(image_name)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        keypoints = self.keypoints_df.iloc[index, 1:].values.reshape(-1, 2).astype("float")
        sample = {"image": image, "keypoints": keypoints}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Resize(object):
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, sample):
        image, keypoints = sample["image"], sample["keypoints"]
        h, w = image.shape[:2]
        if h > w:
            new_h, new_w = self.output_size * h / w, self.output_size
        else:
            new_h, new_w = self.output_size, self.output_size * w / h
        new_h, new_w = int(new_h), int(new_w)
        image = cv2.resize(image, (new_w, new_h))
        keypoints = keypoints * [new_w / w, new_h / h]
        return {"image": image, "keypoints": keypoints}
    
class Crop(object):
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, sample):
        image, keypoints = sample["image"], sample["keypoints"]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size, self.output_size
        top, left = np.random.randint(0, h - new_h), np.random.randint(0, w - new_w)
        image = image[top:top + new_h, left:left + new_w]
        keypoints = keypoints - [left, top]
        return {"image": image, "keypoints": keypoints}
    
class Normalize(object):
    def __call__(self, sample):
        image, keypoints = sample["image"], sample["keypoints"]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image / 255.0
        keypoints = (keypoints - 100) / 50.0
        return {"image": image, "keypoints": keypoints}
    
class ToTensor(object):
    def __call__(self, sample):
        image, keypoints = sample["image"], sample["keypoints"]
        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)
        image = image.transpose((2, 0, 1))
        return {"image": torch.from_numpy(image), "keypoints": torch.from_numpy(keypoints)}