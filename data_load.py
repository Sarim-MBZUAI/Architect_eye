from glob import glob
from PIL import Image, ImageDraw
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import json
import random
from itertools import compress
import pandas as pd
from collections import Counter
from skimage.transform import resize
from shapely.geometry import Polygon
import pickle
import os
from tqdm.contrib.concurrent import process_map
import psutil
from datetime import datetime
from joblib import Parallel, delayed
import contextlib
import joblib
from tqdm import tqdm


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()



class Dacl10kPtdataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Custom dataset for loading Dacl10k data from .pt files.

        Args:
        image_dir (str): Directory where the image .pt files are stored.
        mask_dir (str): Directory where the mask .pt files are stored.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Listing the files in the directories
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.pt')]
        self.mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.pt')]

        # Ensuring the lists are sorted to maintain alignment between images and masks
        self.image_files.sort()
        self.mask_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # Load the .pt files
        image = torch.load(image_path)
        mask = torch.load(mask_path)

        if self.transform:
            image = self.transform(image)

        return image, mask

class Dacl10kDatasetTest(Dataset):

    def __init__(self, data_path, resize_img=(512, 512), normalize_img=True):
        """Dataset for dacl10k test dataset.
        
        Args:
            data_path: Path to dataset root
            resize_img: Apply image resize. Default (512, 512).
            normalize_img: Normalize image. Default True.
        """

        # Paths
        self.data_path = data_path
        self.image_path = f"{self.data_path}"
        self.image_files = sorted(glob(self.image_path + "/*.jpg"))
        
        if len(self.image_files) > 0:
            print(f"Found {len(self.image_files)} image_files in folder {self.image_path}")
        else:
            raise Exception(f"No image_files in folder {self.image_path}")

        # Resize and normalize
        self.resize_img = resize_img
        self.normalize_img = normalize_img
        self.normalize_fct = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.n_samples = len(self.image_files)

    def __getitem__(self, index):
        # Load data sample
        img = self._getitem(index)

        # Resize
        if self.resize_img:
            img = resize(img, self.resize_img)

        # Image transforms
        img = transforms.ToTensor()(img)
        img = img.to(dtype=torch.float32)

        if self.normalize_img:
            img = self.normalize_fct(img)

        return img


    def _getitem(self, index, return_name=False):
        """
        Get item with `index` from internal DataFrame (`self.df`).
        Image is loaded using `PIL.Image`.

        """
        image_name = os.path.basename(self.image_files[index])  # Extracting filename from path

        # Image loading and transform
        img = Image.open(self.image_files[index])
        img = np.array(img, dtype=np.uint8)

        return img


class Dacl10kDataset(Dataset):

    TARGET_LIST = ['Crack', 'ACrack', 'Wetspot', 'Efflorescence', 'Rust', 'Rockpocket', 'Hollowareas', 'Cavity',
               'Spalling', 'Graffiti', 'Weathering', 'Restformwork', 'ExposedRebars', 
               'Bearing', 'EJoint', 'Drainage', 'PEquipment', 'JTape', 'WConccor']

    def __init__(self, split, data_path, resize_mask=(512,512), resize_img=(512,512), normalize_img=True):
        """Dataset for dacl10k dataset.
        
        Args:
            split: "train", "valid", "test".
            data_path: Path to dataset root
            resize_mask: Apply mask resize. You need to define the same size as with image. Default (512, 512). 
            resize_img: Apply image resize. You need to define the same size as with mask. Default (512, 512). 
        """
        self.split = split

        # paths
        self.data_path = data_path

        self.image_path = f"{self.data_path}/images/{split}"
        self.image_files = sorted(glob(self.image_path + "/*.pt"))
        if len(self.image_files) > 0:
            print(f"Found {len(self.image_files)} image_files in folder {self.image_path}") 
        else:
            raise Exception(f"No image_files in folder {self.image_path}")

        # resize and normalize
        self.resize_mask = resize_mask
        self.resize_img = resize_img
        self.normalize_img = normalize_img
        self.normalize_fct = transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        # prefeching can only be enabled running function `run_prefetching`
        self.use_prefetched_data = False

        self.df = self._create_df(self.annotation_files)
        self.n_samples = self.__len__()
        self.target_dict = dict(zip(self.TARGET_LIST, range(len(self.TARGET_LIST))))


    def __getitem__(self, index):
        image_name = self.df.iloc[index]["imageName"] 
        img = torch.load("/home/sanjay.manjunath/Downloads/AI_project/dacl10k-toolkit/pt_files_dacl/"+split+"/image/"+image_name.split(".")[0]+".pt")
        target_mask = torch.load("/home/sanjay.manjunath/Downloads/AI_project/dacl10k-toolkit/pt_files_dacl/"+split+"/mask/"+image_name.split(".")[0]+".pt")
        return img, target_mask 

    def __len__(self):
        return self.df.shape[0]

    def _make_mask_per_class(self, data):
        """Creates a 3d mask with as many channels as classes in TARGET_LIST, i.e. with shape (h,w,c)."""
        target_mask = np.zeros((data["imageHeight"], data["imageWidth"], len(self.TARGET_LIST)))
        for index, shape in enumerate(data["shapes"]):
            target_img = Image.new('L', (data["imageWidth"], data["imageHeight"]), 0)
            if shape["label"] in self.TARGET_LIST:
                target_index = self.target_dict[shape["label"]]
                polygon = [(x,y) for x, y in shape["points"]] # list to tuple
                ImageDraw.Draw(target_img).polygon(polygon, outline=1, fill=1)
                target_mask[:,:,target_index] += np.array(target_img)               
        return target_mask.astype(bool).astype(np.uint8)        

    @staticmethod
    def _get_data(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data

    def _create_df(self, annotation_paths):
        """Create DataFrame for easier analysis of dacl5k data."""

        # Empty dataFrame
        df = pd.DataFrame()

        # Empty counter, e.g. Counter({'Rust': 7, 'Spalling': 3, 'Crack': 0, 'Efflorescence': 0, 'ExposedRebars': 0})
        empty_counter = Counter(dict(zip(self.TARGET_LIST, [0] * len(self.TARGET_LIST)))) 

        for filename in tqdm(annotation_paths, desc="Create internal df"):
            # Get data and add new "class" key
            data = self._get_data(filename)
            class_list = [x["label"] for x in data["shapes"]]
            # Count damages
            counts = Counter(class_list)
            counts.update(empty_counter)

            # Count damages raw
            label_list = [x["label"] for x in data["shapes"]]
            counts_raw = Counter(label_list)

            # Remove shapes update counts
            data.pop("shapes")
            data.update(counts)

            # Append current data to df
            data_df = pd.DataFrame([data])
            df = pd.concat([df, data_df], ignore_index=True)

        return df

    @staticmethod
    def get_full_annotation_filename(image_name, annotation_path):
        base_name = image_name.split(".")[0]
        filename = base_name + ".json"
        full_annotation_filename = os.path.join(annotation_path, filename)
        return full_annotation_filename
        
    def __repr__(self):
        return f"Dacl5kDataset(split={str(self.split or 'None')}, data_path={self.data_path}, resize_mask={self.resize_mask}, resize_img={self.resize_img}, normalize_img={self.normalize_img})"
        


if __name__ == "__main__":   
    now = datetime.now()
    print(now)     
    PATH_TO_DATA = "/home/sanjay.manjunath/Downloads/AI_project/dacl10k-toolkit/pt_files_dacl/"

    split = "validation"
    resize_mask = (512, 512) 
    resize_img = (512, 512)

    for split in ["validation"]:
        print("="*80)
        print(split)
        dataset = Dacl10kDataset(split, PATH_TO_DATA, resize_mask=resize_mask, resize_img=resize_img, normalize_img=True)
        dataset[0]
        import pdb; pdb.set_trace()
        print("="*80)
    now = datetime.now()
    print(now)    
    print("Done")       