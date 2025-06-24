from datasets.ImageDataset import ImageDataset
from models.unet3d import SimpleUNet3D
from utils.train import train
from utils.test import test 
from utils.evaluate import evaluate
from utils.visualize import visualize_prediction 
from utils.transform import get_test_tf, get_train_tf 
from utils.kits23_download import get_dataset

import os
import subprocess
import torch
from torch import optim
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from sklearn.model_selection import train_test_split
from monai.data import DataLoader

train_tf=get_train_tf()
test_tf=get_test_tf()

get_dataset()

path='./datasets/kits23/'
all_path=os.listdir(path)
all_path=[os.path.join(path, filename) for filename in all_path if 'case_' in filename]

volumes_path=[os.path.join(filename, 'imaging.nii.gz') for filename in all_path if os.path.exists(os.path.join(filename, 'imaging.nii.gz'))]
masks_path=[os.path.join(filename, 'segmentation.nii.gz') for filename in all_path if os.path.exists(os.path.join(filename, 'segmentation.nii.gz'))]

train_files, temp_files, train_masks, temp_masks=train_test_split(volumes_path, masks_path, test_size=0.2, random_state=42, shuffle=True)
val_files, test_files, val_masks, test_masks=train_test_split(temp_files, temp_masks, test_size=0.5, random_state=42, shuffle=True)

trainset=ImageDataset(train_files, train_masks, train_tf)
valset=ImageDataset(val_files, val_masks, test_tf)
testset=ImageDataset(test_files, test_masks, test_tf)

trainloader=DataLoader(trainset, 4, shuffle=True ,num_workers=4, pin_memory=True)
valloader=DataLoader(valset, 2, shuffle=False ,num_workers=2, pin_memory=True)
testloader=DataLoader(testset, 2, shuffle=True ,num_workers=2, pin_memory=True)


if __name__=="__main__":
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=SimpleUNet3D().to(device)
    criterion=DiceCELoss(sigmoid=True)
    optimizer=optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    dice_metric=DiceMetric(include_background=True, reduction='mean')
    
    for epoch in range(50):
        trainloss=train(model, trainloader, optimizer, criterion, device)
        valloss, dicescore=evaluate(model, valloader, criterion, dice_metric, device)
        print(f'Epoch: {epoch+1}, Train Loss: {trainloss:.4f}, Val Loss: {valloss:.4f}, Dice Score: {dicescore*100:.2f}%')
    
    test(model, testloader, dice_metric, device)
    visualize_prediction(model, testloader, 3, [32, 64, 92], device)