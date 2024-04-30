from dacl10k.dacl10kdatasetsavept import Dacl10kDataset
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

resize_dimensions = (256, 256)


# Dataset paths
train_data_path = "/home/sanjay.manjunath/Downloads/AI_project/dacl10k/DACL/dacl10k_v2_devphase/"
val_data_path = "/home/sanjay.manjunath/Downloads/AI_project/dacl10k/DACL/dacl10k_v2_devphase/"

# Create Datasets
pt_files = '/home/sanjay.manjunath/Downloads/AI_project/dacl10k/pt_files_dacl'

train_dataset = Dacl10kDataset(split="train", data_path=train_data_path, resize_mask=resize_dimensions, resize_img=resize_dimensions)
val_dataset = Dacl10kDataset(split="validation", data_path=val_data_path, resize_mask=resize_dimensions, resize_img=resize_dimensions)

# DataLoader parameters
batch_size = 4
num_workers = 2  
shuffle = True  

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#Saves Training images and masks as tensors

for batch in tqdm(train_loader):
    images, masks, image_names = batch
    
    for i in range(len(images)):
        image_name = image_names[i].split(".")[0]  # Get the image name without the extension
        image_path = os.path.join("/home/sanjay.manjunath/Downloads/AI_project/pt_files_dacl/train/image", f"{image_name}.pt")
        mask_path = os.path.join("/home/sanjay.manjunath/Downloads/AI_project/pt_files_dacl/train/mask", f"{image_name}.pt")

        # Save the image and mask tensors
        torch.save(images[i], image_path)
        torch.save(masks[i], mask_path)


#Saves Validation images and masks as tensors

for batch in tqdm(val_loader):
    images, masks, image_names = batch
    
    for i in range(len(images)):
        image_name = image_names[i].split(".")[0]  # Get the image name without the extension
        image_path = os.path.join("/home/sanjay.manjunath/Downloads/AI_project/dacl10k/pt_files_dacl/validate/image", f"{image_name}.pt")
        mask_path = os.path.join("/home/sanjay.manjunath/Downloads/AI_project/dacl10k/pt_files_dacl/validate/mask", f"{image_name}.pt")

        # Save the image and mask tensors
        torch.save(images[i], image_path)
        torch.save(masks[i], mask_path)
