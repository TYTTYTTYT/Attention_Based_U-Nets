# -*- coding: utf-8 -*-
"""
This is the base training program to train the specified U-Net.
You can specify the hyper parameters below.
Each time you run this program would train the model once.
"""

from nets import SE_UNet
from nets import Res_SE_UNet
from nets import Full_SE_UNet
from losses import DiceLoss, FocalLoss, Dice_Focal
from metrics import IoU_Score
from utils import load_data
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import numpy as np
from os.path import isfile


# You can specify the hyper parameters here
batch_size = 16
learning_rate = 0.0001
num_epoch = 50
# Speficy the path to store the trained model
best_model_path = '00010best_Res_SE_UNet_Dice_Focal_basic_modified.model'
final_model_path = '00010final_Res_SE_UNet_Dice_Focal_basic_modified.model'
# Specify the model and the seeds
model = Res_SE_UNet
pretrained = True
# Setting the reproducibility. If you reuiqres the reproducibility,
# the training would be slower.
deterministic = False
seed = 30

# Initialize the metrics
best_eval_loss = 999999.99
best_eval_iou = -1.0

if __name__ == "__main__":
    # Check existency of trained models
    assert not isfile(best_model_path), f"{best_model_path} already exists!"
    assert not isfile(final_model_path), f"{final_model_path} already exists!"
    
    # Set seed for reproducibility
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Check GPU availibility, use GPU in advanvce
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on ", device)
    
    # Load the dataset and split into 4:1
    dataset = load_data()
    line = int(len(dataset) * 0.8)
    train_set = dataset[:line]
    eval_set = dataset[line:]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=16, shuffle=False)

    # Initialize the neural net, optimizer, and the loss function
    net = model()
    net.to(device)
    
    if pretrained:
        optimizer = Adam(net.parameters(), lr=learning_rate)
        
        for p in list(net.encoder_parameters()):
            p.requires_grad = False
    else:
        optimizer = Adam(net.parameters(), lr=learning_rate)
        
    # set scheduler to regulate learning rate by monitoring validation IoU
    scheduler = ReduceLROnPlateau(optimizer, factor = 0.2, verbose = True)
    criterion = Dice_Focal().to(device)
    iou_cal = IoU_Score()

    # Initialize the loss log
    net.training_losses = []
    net.eval_losses = []
    net.training_iou = []
    net.eval_iou = []
    
    # Loop training epoches
    for epoch in tqdm(range(num_epoch)):
        # Add the ResNet parameters into the optimizer to train after 5 epoches
        if epoch == 4 and pretrained:
            for p in list(net.encoder_parameters()):
                p.requires_grad = True

        net.train()
        running_loss = 0.0
        iou_score = 0.0
        for idx, data in enumerate(train_loader):
            # Move data to GPU
            X, Y = data
            X = X.to(device)
            Y = Y.to(device)

            # Train and update the net for one mini-batch
            optimizer.zero_grad()

            pred = net(X)
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()
            iou = iou_cal(pred.round(), Y)
#             iou = iou_cal(pred.where(x >= threshold, 1.0, 0.0), Y)
            
            iou_score += iou.item()
            running_loss += loss.item()

        # Record the training loss
        epoch_loss = running_loss / (idx + 1)
        net.training_losses.append(epoch_loss)
        
        epoch_iou = iou_score / (idx + 1)
        net.training_iou.append(epoch_iou)
        
        print(f"\nTraining \nloss：{epoch_loss}, IoU: {epoch_iou}")
        # Enter evaluation mode
        net.eval()
        running_loss = 0.0
        iou_score = 0.0
        # Calculate the loss and IoU score on evaluation set
        with torch.no_grad():
            for idx, data in enumerate(eval_loader):

                X, Y = data
                X = X.to(device)
                Y = Y.to(device)

                pred = net(X)
                loss = criterion(pred, Y)
                iou = iou_cal(pred.round(), Y)

                running_loss += loss.item()
                iou_score += iou.item()
        # Record the evaluation loss
        eval_loss = running_loss / (idx + 1)
        net.eval_losses.append(eval_loss)
        
        eval_iou = iou_score / (idx + 1)
        net.eval_iou.append(eval_iou)
    
        # Store the best model
        if epoch >= 3:
            # save model based on IoU score instead
            if eval_iou > best_eval_iou:
                torch.save(net, best_model_path)
                best_eval_iou = eval_iou
                
        print(f"Evaluation \nloss：{eval_loss}, IoU: {eval_iou}, best IoU: {best_eval_iou}")
                
        # update leraning rate to a factor if validation IoU did not improve for 10 consecutive epochs 
        scheduler.step(eval_iou)
    
    # Store the final model
    torch.save(net, final_model_path)
