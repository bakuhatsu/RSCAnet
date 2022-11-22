
path = "/home/sven/data/ctw1500/"
subfolder = "train"
file = "trainval.txt"
file_path = f"{path}{subfolder}/{file}"

# Open file
img_paths_file = open(file_path, "r")

# Read data
data = img_paths_file.read()

# Convert to list with one item per line (and strip whitespace)
img_paths = data.split("\n")
# Make sure any whitespace around path is stripped
img_paths = [item.strip() for item in img_paths]

print(img_paths[0:20])

def load_paths(subfolder, file, path="/home/sven/data/ctw1500/", prepend="../../"):
    file_path = f"{path}{subfolder}/{file}"
    # Open file
    img_paths_file = open(file_path, "r")

    # Read data
    data = img_paths_file.read()

    # Convert to list with one item per line (and strip whitespace)
    img_paths = data.split("\n")
    # Make sure any whitespace around path is stripped
    img_paths = [item.strip() for item in img_paths]
    # Prepend relative path to data folder so that paths work
    img_paths = [prepend + item for item in img_paths]

    return img_paths




# train_images = load_paths(subfolder="train", file="trainval.txt")
# train_curvebbox = load_paths(subfolder="train", file="trainval_label_curve.txt")

# test_images = load_paths(subfolder="test", file="test.txt")
# test_curvebbox = load_paths(subfolder="test", file="test_label_curve.txt")

train_images = load_paths(subfolder="train", file="img_paths.txt")
train_curvebbox = load_paths(subfolder="train", file="label_curve_paths.txt")

test_images = load_paths(subfolder="test", file="img_paths.txt")
test_curvebbox = load_paths(subfolder="test", file="label_curve_paths.txt")

print(train_images[0:10])
print(train_curvebbox[0:10])
print(test_images[0:10])
print(test_curvebbox[0:10])

#print(["../../" + item for item in train_images[0:10]])


import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

img_HW = Image.open(train_images[0]).convert("RGB")
print(img_HW.size)

display(img_HW)

polygons = open(train_curvebbox[0], "r")

# Read data
polygon_data = polygons.read()

# Convert to list with one item per line 
polygon_list = polygon_data.split("\n")

print(polygon_list[0].split(","))

if (polygon_list[len(polygon_list)-1] == ""):
    polygon_list = polygon_list[:-1]

for polygon in polygon_list:
    polygon = polygon.split(",")
    # Make sure any whitespace around path is stripped
    polygon = [item.strip() for item in polygon]
    polygon = [int(num) for num in polygon]

print(eval(polygon_list[0]))
type(eval(polygon_list[0]))
print(polygon_list)

polygn = zip(*[iter(eval(polygon_list[0]))]*2)
pg = list(polygn)
print(pg)

width = img_HW.size[0]
height = img_HW.size[1]
img = Image.new("L", (width, height), 0)
ImageDraw.Draw(img).polygon(pg, outline=1, fill=1)  ## Must be a list of tuples
mask = np.array(img)

plt.imshow(mask, cmap="gray")
plt.show()

display(img_HW)

## Import png images (transparency will be ignored, which I think is ok)
from PIL import Image

path_png = "../../data/totaltext/Text_Region_Mask/Train/img101.png"
msk_png = Image.open(path_png).convert("L")

display(msk_png)


import argparse
import torch
from torch.utils.data import DataLoader, Dataset, random_split
#from torch.optim import Adam
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms as tfm
from torchvision.models import resnet18 as resnet18  # Use this for resnet18 model
# Useful link to list of models: https://pytorch.org/vision/stable/models.html
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import QuantizationAwareTraining, ModelCheckpoint #, ModelPruning, 
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import copy
# Possible additional imports (delete unneeded later)
# import torchvision as tv
# import torchvision.transforms as transforms
# import torchvision.models as models
# import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pretrainedmodels
import numpy as np

## Predictions from trained model
import torch

from training import RSCANet

model = RSCANet.load_from_checkpoint("../RSCAnet_checkpoints/lightning_logs/version_17/checkpoints/epoch=49-step=11950.ckpt")

model.eval()

# Load image to predict mask for
img_path = "../../data/totaltext/Images/Train/img101.jpg"

transform = tfm.Compose([
    tfm.CenterCrop(640),
    tfm.ToTensor()
])

img_HW = Image.open(img_path).convert("RGB")
img_CHW = transform(img_HW)

display(img_HW)

with torch.no_grad():
    pred = model(img_CHW.unsqueeze(0))

# pred = torch.sigmoid(pred)
print(pred.max())
print(pred.min())

display(pred)

pred2 = torch.where(pred > 0.001, 1, 0)
display(pred2)
pred2.shape

#plt.imshow(pred2.squeeze(0).permute(1, 2, 0))

plt.imshow(pred2.squeeze(0).permute(1, 2, 0), cmap="gray")
plt.show()

def plot_predictions(image="img101.jpg", cutoff=0.001):
    model = RSCANet.load_from_checkpoint("../RSCAnet_checkpoints/lightning_logs/version_17/checkpoints/epoch=49-step=11950.ckpt")

    model.eval()

    # Load image to predict mask for
    img_path = f"../../data/totaltext/Images/Train/{image}"

    transform = tfm.Compose([
        tfm.CenterCrop(640),
        tfm.ToTensor()
    ])

    img_HW = Image.open(img_path).convert("RGB")
    img_CHW = transform(img_HW)

    # display(img_HW)

    with torch.no_grad():
        pred = model(img_CHW.unsqueeze(0))

    # pred = torch.sigmoid(pred)
    print("Min: ", pred.min())
    print("Max: ", pred.max())

    # display(pred)

    pred2 = torch.where(pred > cutoff, 1, 0)
    #display(pred2)
    pred2.shape

    #plt.imshow(pred2.squeeze(0).permute(1, 2, 0))
    display(img_HW)

    plt.imshow(pred2.squeeze(0).permute(1, 2, 0), cmap="gray")
    plt.show()


plot_predictions()

plot_predictions(image="img19.jpg", cutoff=0.0005)
plot_predictions(image="img21.jpg", cutoff=0.0001)

# Something wrong with my prediction code. 