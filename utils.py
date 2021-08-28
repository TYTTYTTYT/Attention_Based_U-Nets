import pickle
from config import DATA_SET
import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import torch
import random

DataLoader = torch.utils.data.DataLoader

def load_data():
    '''
    Load the entire Brain MRI dataset as numpy arrays.
    
    param:
    
    return:
        list[tuple(image, mask)]

        image is a (3, 256, 256) numpy float32 array, 
        all numbers are within range (0, 1)

        mask is a (1, 256, 256) numpy float32 array,
        all numbers are within range (0, 1)
    '''
    with open(DATA_SET, 'rb') as f:
        data_set = pickle.load(f)
    
    return data_set

def find_tumor_samples(dataset, n):
    '''
    Find samples with tumor from the dataset.
    Each sample is a image, mask pair.
    
    param:
        dataset: the dataset
        n: number of data samples to return
        
    return:
        list[tuple(image, mask)]

        image is a (3, 256, 256) numpy float32 array, 
        all numbers are within range (0, 1)

        mask is a (1, 256, 256) numpy float32 array,
        all numbers are within range (0, 1)
    '''
    tumor_samples = []
    
    while len(tumor_samples) < n:
        idx = random.randrange(len(dataset))
        mask = dataset[idx][1]
        
        if np.sum(mask) > 0.0:
            tumor_samples.append(dataset[idx])
    
    return tumor_samples

def plot_contour(imgs, masks, preds, H, W, thresh=0.5):
    '''
    Plot the contour and annotated masks on brain MRI images.
    The ground truth masks will be ploted as a green area
    The predicted tumor will be ploted as a red contour
    
    param:
        imgs: a list of 3-channel graphs
            each image should be (3, 256, 256) ndarray
        masks: a list of 1-channel graphs
            each mask should be (1, 256, 256) ndarray
        preds: a list of 1-channel graphs
            each pred shoud be (1, 256, 256) ndarray
        H: number of rows
        W: number of columns
        
    return:
    '''
    assert len(imgs) == H * W == len(masks) == len(preds), "Number of images don't match the size"
    
    pred = concat_graph(preds, H, W)
    img = concat_graph(imgs, H, W)
    mask = concat_graph(masks, H, W)
    
    masked_img = img
    masked_img[1, :, :] += mask.squeeze()
    masked_img = np.clip(masked_img, 0, 1)
    contours = measure.find_contours(pred.squeeze(), thresh)
    
    fig, ax = plt.subplots(figsize=(H * 3, W * 3))
    ax.imshow(np.moveaxis(masked_img, 0, -1))
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
        ax.axis('off')
    
    return

def get_imgs_from(samples):
    imgs = []
    
    for s in samples:
        imgs.append(s[0])
        
    return imgs

def get_masks_from(samples):
    masks = []
    
    for s in samples:
        masks.append(s[1])
        
    return masks

def concat_graph(graphs, H, W):
    n = len(graphs)
    assert H * W == n
    
    rows = []
    
    for i in range(H):
        row = np.concatenate(graphs[i*H: i*H+W], axis=2)
        rows.append(row)
        
    return np.concatenate(rows, axis=1)

def concat_samples(samples, H, W):
    imgs = get_imgs_from(samples)
    masks = get_masks_from(samples)
    
    img = concat_graph(imgs, H, W)
    mask = concat_graph(masks, H, W)
    
    return img, mask
