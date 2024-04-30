from data_load import Dacl10kPtdataset
from torch.utils.data import DataLoader
import torch
import segmentation_models_pytorch as smp
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import logging
import torch.nn as nn
# Configure logging
log_file_path = "/home/sanjay.manjunath/Downloads/AI_project/Music/new_model.pylogfile.log"  # Set your log file path here
logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')


# Modified Function to calculate IoU for multilabel segmentation
def calculate_iou_multilabel(pred, target, threshold=0.5):
    pred = pred > threshold
    target = target > threshold

    pred = pred.bool()
    target = target.bool()

    # Calculate IoU for each class
    intersection = (pred & target).float().sum((0, 2, 3))
    union = (pred | target).float().sum((0, 2, 3))

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou  # Returns IoU for each class



class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return 1 - dice

# Dataset paths
train_image_dir = "/l/users/sanjay.manjunath/Working_thanks_to_pd/dacl10k-toolkit/pt_files_dacl/train/image/"
train_mask_dir = "/l/users/sanjay.manjunath/Working_thanks_to_pd/dacl10k-toolkit/pt_files_dacl/train/mask/"

val_image_dir = "/l/users/sanjay.manjunath/Working_thanks_to_pd/dacl10k-toolkit/pt_files_dacl/validate/image/"
val_mask_dir = "/l/users/sanjay.manjunath/Working_thanks_to_pd/dacl10k-toolkit/pt_files_dacl/validate/mask/"

# Create Datasets
train_dataset = Dacl10kPtdataset(train_image_dir, train_mask_dir)
val_dataset = Dacl10kPtdataset(val_image_dir, val_mask_dir)

# DataLoader parameters
batch_size = 12
num_workers = 4
shuffle = True

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the FPN model with EfficientNetB4 as the encoder
model = smp.FPN(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=19 # Use sigmoid activation for multilabel
)
model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Use BCEWithLogitsLoss for multilabel segmentation
criterion1 = torch.nn.BCEWithLogitsLoss()
criterion2 = DiceLoss() 

# Training and Validation with Progress Bar
num_epochs = 20
best_mIoU = 0
best_epoch = 0

train_losses, val_losses = [], []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_losses_epoch = []
    with tqdm(train_loader, unit="batch") as tepoch:
        for images, masks in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs} - Train")
            
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss1 = criterion1(output, masks)
            loss2 = criterion2(output,masks)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            train_losses_epoch.append(loss.item())
            tepoch.set_postfix(loss=loss.item())

    # Validation phase
    model.eval()
    val_losses_epoch = []
    val_iou_scores_per_class = []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            output = model(images)
            val_loss = criterion1(output,masks) + criterion2(output, masks)  
            val_losses_epoch.append(val_loss.item())

            iou_per_class = calculate_iou_multilabel(output, masks)
            val_iou_scores_per_class.append(iou_per_class)

    avg_val_iou_per_class = torch.mean(torch.stack(val_iou_scores_per_class), dim=0)
    logging.info(f"Epoch {epoch+1} - Validation mIoU per class: {avg_val_iou_per_class.tolist()}")

    # Save best model
    avg_val_iou = avg_val_iou_per_class.mean()
    if avg_val_iou > best_mIoU:
        best_mIoU = avg_val_iou
        best_epoch = epoch
        best_model_path = os.path.join("/home/sanjay.manjunath/Downloads/AI_project/Music/models", 'fpn_eff_b4_best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mIoU': best_mIoU,
        }, best_model_path)
        logging.info(f"New best model saved with mIoU: {best_mIoU}")

logging.info(f"Training complete. Best model was from epoch {best_epoch+1} with mIoU: {best_mIoU}")
