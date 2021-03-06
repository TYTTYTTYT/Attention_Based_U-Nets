from nets import SE_UNet
from nets import Res_SE_UNet
from nets import Full_SE_UNet
from nets import SE_UNet
from nets import UNet
from losses import DiceLoss, FocalLoss, Dice_Focal
from metrics import IoU_Score
from utils import load_data
from utils import validate_path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import numpy as np
from os.path import isfile
from argparse import ArgumentParser
import nets
import losses as loss

parser = ArgumentParser(description="Train U-Net modes in one run")
parser.add_argument(
    '-b', '--batch', nargs=1, default=[16], help='Mini batch size', type=int
)
parser.add_argument(
    '-l', '--learningrate', nargs=1, default=[0.0001], help='The initial learning reate', type=float
)
parser.add_argument(
    '-e', '--epoch', nargs=1, default=[50], help='Number of epochs', type=int
)
parser.add_argument(
    '-t', '--train', nargs=1, default=[5], help='Number of training', type=int
)
parser.add_argument(
    '-m', '--models', nargs='+', help='Models to train', 
    choices=['SE_UNet', 'Res_SE_UNet', 'Full_SE_UNet', 'UNet', 'CBAM_UNet', 'Res_CBAM_UNet', 'Full_CBAM_UNet']
)
parser.add_argument(
    '-p', '--pretrain', nargs='+', help='Is the model pretrained', type=bool
)
parser.add_argument(
    '-s', '--loss', nargs='+', help='Loss functions for each model', 
    choices=['DiceLoss', 'FocalLoss', 'Dice_Focal']
)
args = parser.parse_args()

print(args.batch)
print(args.learningrate)
print(args)

# Training parameters
batch_size = args.batch[0]
learning_rate = args.learningrate[0]
num_epoch = args.epoch[0]
num_train = args.train[0]
if args.models is None or args.pretrain is None or args.loss is None:
    parser.print_help()
    print('Incifficient parameters!')
    exit(0)
if len(args.train) != len(args.pretrain):
    parser.print_help()
    print('The number of models and pretrain must match!')
    exit(0)
# What models to train, the pretraineds match to each model
models = tuple(map(lambda m: nets.__dict__[m], args.models))
pretraineds = args.pretrain
# What loss functions to use
losses = tuple(map(lambda l: loss.__dict__[l], args.loss))


# Train function: train the specified model with specified parameters once
def train(model, pretrained, loss, seed, batch_size, learning_rate, num_epoch, best_model_path, final_model_path):
    # Check existency of trained models
    assert not isfile(best_model_path), f"{best_model_path} already exists!"
    assert not isfile(final_model_path), f"{final_model_path} already exists!"
    
    # Init the evaluation metrics
    best_eval_loss = 999999.99
    best_eval_iou = -1.0
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
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
    criterion = loss().to(device)
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
        
        print(f"\nTraining \nloss???{epoch_loss}, IoU: {epoch_iou}")
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
                
        print(f"Evaluation \nloss???{eval_loss}, IoU: {eval_iou}, best IoU: {best_eval_iou}")
                
        # update leraning rate to a factor if validation IoU did not improve for 10 consecutive epochs 
        scheduler.step(eval_iou)
    
    # Store the final model
    torch.save(net, final_model_path)

    return best_eval_iou

if __name__ == "__main__":
    num_loss = len(losses)
    num_model = len(models)
    assert num_model == len(pretraineds)
    
    num_set = list(range(10000))
    global_best = 0
    global_best_model_path = ''

    for idx_model in range(num_model):
        # Set seeds, model and pretrained for each model
        seeds = random.sample(num_set, num_train)
        model = models[idx_model]
        pretrained = pretraineds[idx_model]

        for idx_train in range(1, num_train + 1):
            # Set the seed
            seed = seeds[idx_train - 1]

            # Train for each loss
            for loss in losses:
                
                # Set the path to store the trained model
                best_model_path = f'0000best_{str(model)}_{str(loss)}_{seed}.model' 
                final_model_path = f'0000final_{str(model)}_{str(loss)}_{seed}.model'
                best_model_path = validate_path(best_model_path)
                final_model_path = validate_path(final_model_path)
                
                print(f'Models: {str(models)}, losses: {str(losses)}')
                print(f'Train {str(model)} in turn {idx_train}/{num_train} with {str(loss)}.')
                
                best = train(model, pretrained, loss, seed, batch_size, learning_rate, num_epoch, best_model_path, final_model_path)
                
                if best > global_best:
                    global_best = best
                    global_best_model_path = best_model_path
                    
                print(f'Current best eval iou for all: {global_best}')
                print(f'Achieved by {global_best_model_path}.')
