###############################################################################################
# Sven Nelson
# ECE 570: Artificial Intelligence
# Student ID: nelso531
# Date: 10/1/2022
#
# Reimplementing RSCA network based on the paper:
# Li, Jiachen, Yuan Lin, Rongrong Liu, Chiu Man Ho, and Humphrey Shi. “RSCA: Real-Time 
# Segmentation-Based Context-Aware Scene Text Detection.” In 2021 IEEE/CVF Conference 
# on Computer Vision and Pattern Recognition Workshops (CVPRW), 2349–58. Nashville, TN, 
# USA: IEEE, 2021. https://doi.org/10.1109/CVPRW53098.2021.00267.
###############################################################################################

# Imports
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


def parameters():
    parser = argparse.ArgumentParser(description="RSCAnet Trainer")
    # Run quick-test mode or full training
    parser.add_argument("--quick_test", help="Runs only a single batch of train, val, and test for testing code,", action="store_true")
    args, args_other = parser.parse_known_args()
    return (args, args_other)


# Define Dataset for dataloader
class TotalTextDataset(Dataset):
    def __init__(self, traintest, data_dir="../../data/totaltext/", transform=None):
        # Create empty list for paths to images
        self.images = []
        # Create empty list for paths to text region masks
        self.masks = []

        ## First, get list of images
        # Define path to original images folder 
        path = os.path.join(data_dir, f"Images/{traintest}/")
        # Get a list of all jpeg files in the folder
        image_list = glob(f"{path}/*.jpg")
        # For each image file
        for img in image_list:
            # Append image path to the images list
            self.images.append(img)
        
        ## Next, get list of text region masks that correspon with the images
        # Define path to original images folder 
        path = os.path.join(data_dir, f"Text_Region_Mask/{traintest}/")
        # Get a list of all jpeg files in the folder
        mask_list = glob(f"{path}/*.jpg")
        # For each image file
        for msk in mask_list:
            # Append image path to the images list
            self.masks.append(msk)

        # Set the transform equal to the passed transform
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Import image (in height x width format)
        img_HW = Image.open(self.images[idx]).convert("RGB")
        # If any images in the dataset are grayscale, above forces them to import as 3-channel images
        msk_HW = Image.open(self.masks[idx]).convert("L")
        
        # Rearrange to CxHxW format and apply necessary transforms
        img_CHW = self.transform(img_HW)
        msk_CHW = self.transform(msk_HW)

        # Return processed image and label
        return (img_CHW, msk_CHW)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class LCAU_Block(nn.Module):
  def __init__(self, in_channels, out_channels, l=5):
    super(LCAU_Block, self).__init__()
    ## Fill in the rest here
    self.r = 2 # Set scale factor here
    self.l = l # sqrt of 25 = 5
    self.conv1 = nn.Conv2d(in_channels, l**2, kernel_size = 3, stride = 1, padding = 1)
    self.nn_upsample = nn.Upsample(scale_factor=self.r, mode="nearest")
    #self.batch_size = batch_size
    self.conv2 = nn.Conv2d(4*in_channels, out_channels, kernel_size = 1)

  def forward(self, x):
    original = x
    # Get dimensions for the passed tensor to use in steps below
    batch_dim, channel_dim, H, W = x.size()
    print("lcau1: ", x.shape)
    out = self.conv1(x)
    out = self.nn_upsample(out)
    print("lcau2: ", out.shape)
    # custom kernel for each pixel (values are pixel values for each channel l^2)
    # Apply softmax to kernel (should apply independently for each kernel for best results, I think)
    out = F.softmax(out, dim=1) # Should this get moved later? 

    # Need to generate weights from out (reshape, view, and/or stack)
    out = out.unfold(2, self.r, step=self.r)
    out = out.unfold(3, self.r, step=self.r)
    #out = out.reshape(batch_num, kernel_size ** 2, H, W, self.r ** 2) ## NEED TO FIX (variables don't exist)
    out = out.reshape(batch_dim, self.l ** 2, H, W, self.r ** 2)
    out = out.permute(0, 2, 3, 1, 4)
    print("lcau3: ", out.shape)

    # Reassembly
    original = F.pad(original, pad=(self.l//2, self.l//2, self.l//2, self.l//2), 
                     mode="constant", value=0)
    # Put original tensor into right shape to multiply with weights tensor generated above
    original = original.unfold(2, self.l, step=1)
    original = original.unfold(3, self.l, step=1)
    original = original.reshape(batch_dim, channel_dim, H, W, -1)
    original = original.permute(0,2,3,1,4)
    print("lcau4: ", out.shape)

    # Perform multiplication to apply wieghts kernel to each pixel of original
    out = torch.matmul(original, out)
    out = out.reshape(batch_dim, H, W, -1)
    out = out.permute(0, 3, 1, 2)
    print("lcau5: ", out.shape)
    out = self.nn_upsample(out)
    print("lcau6: ", out.shape) # lcau6:  torch.Size([1, 256, 112, 112])
    out = self.conv2(out)
    print("lcau7: ", out.shape)

    ############################
    # Apply that kernel to each pixel of original in an upsampling operation
    # original = self.nn_upsample(original) # This is equivalent to multiplying each pixel by all values of the kernel to upsample
    # out = F.conv2d(original, weights, stride=1, padding=2, groups=self.batch_size)
    # out = out.view(N, 15, 24, 24) # Not sure if I need to reshape this here?  
    ############################
    # Can upsample by 2, use nearest neighbor
    # Return output (now upsampled by r (in this case 2x)) from original 
    return out


class RSCA_Resnet(nn.Module):
    def __init__(self, block, layers, num_classes = 1000):
        super(RSCA_Resnet, self).__init__()
        self.model =  pretrainedmodels.__dict__['resnet18'](pretrained='imagenet')
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        # LCAU_Block takes in_channels, out_channels
        self.lcau1 = LCAU_Block(512, 256)  # 512/2 = 256
        self.lcau2 = LCAU_Block(256, 128)  # 256/2 = 128
        self.lcau3 = LCAU_Block(128, 64)   # 128/2 = 64
        self.lcau4 = LCAU_Block(64, 32)    # 64/2 = 32
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor): #-> List[torch.Tensor]:
        # See note [TorchScript super()]
        x = self.conv1(x)
        #x = self.bn1(x)
        #x0 = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        #return [x1,x2,x3,x4] # returns features with channels [64,64,128,256,512]

        ## Feature Pyramid with LCAU blocks
        # print("step1: ", x4.shape)
        x = self.lcau1(x4)
        # print("step2: ", x.shape)
        # print("shape of x3: ", x3.shape)
        # Sum x and x3
        x += x3
        # print("step3: ", x.shape)
        x = self.lcau2(x)
        # print("step4: ", x.shape)
        x += x2
        # print("step5: ", x.shape)
        x = self.lcau3(x)
        # print("step6: ", x.shape)
        x += x1
        # print("step7: ", x.shape)
        x = self.lcau4(x)
        # print("step8: ", x.shape)
        # do LCAU steps take 2 inputs or are there additions (I think multiplications?) between steps?
        # Nope, just addition as above (I think) 

        # Make output pixels between 0 and 1
        x = torch.sigmoid(x)

        return x


# For pytorch lightning, everything goes inside of the network class
class RSCANet(LightningModule):
    def __init__(self, bs=10, lr=0.007, workers=4, epochs=7, data_dir="../../data/totaltext/", seed=42):
        super().__init__()
        # Set random seed
        pl.seed_everything(seed)
        # Load RSCA model with Resnet18 backbone
        rsca_model = RSCA_Resnet(ResidualBlock, [3, 4, 6, 3])
        # Use rsca_model as network for training
        self.net = rsca_model
        self.data_dir = data_dir
        # Set batch size based on value passed
        self.bs = bs
        # Set learning rate based on value passed
        self.lr = lr
        # Set current epoch and total epochs
        self.current_epoch = 0
        self.max_epoch = epochs
        # Set number of workers based on value passed
        self.workers = workers
        # Set criterion for loss to Binary Cross Entropy Loss with weighted positives
        #self.criterion = nn.BCELoss()
        self.criterion = nn.BCEWithLogits(pos_weights=3)  # From paper: loss ration pos:neg 1:3
        # For plotting training vs validation loss
        self.running_train_loss = []
        self.running_val_loss = []
        # Accuracy metrics
        self.correct = 0
        self.pixels = 0
        # self.dice_score = 0

    def forward(self,x):
        output = self.net(x)
        return output

    def prepare_data(self):   
        # Transform to match expected input image format 
        # Mini-batches of 3-channel RGB images of shape (3 x H x W)
        # 1) Images are randomly horizontally flipped and rotated in range [-10, 10] (degrees)
        # 2) Images are randomly reshaped with ratio [0.5, 3.0] and then cropped by 640 x 640

        transform = tfm.Compose([
            tfm.RandomHorizontalFlip(p=0.5),
            tfm.RandomRotation(degrees=(-10,10)),
            #tfm.RandomAffine(translate=(0.5, 3.0)),  ## Unclear what is meant by "reshaped with ratio"
            tfm.CenterCrop(640),
            tfm.ToTensor(),
            tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Define TotalText train and test datasets
        train_dataset = TotalTextDataset(traintest="Train", data_dir=self.data_dir, transform=transform)
        test_dataset = TotalTextDataset(traintest="Test", data_dir=self.data_dir, transform=transform)

        self.test_data = test_dataset
        
        # Define split for training into train/val: 80:20
        split = [int(0.80*len(train_dataset)), int(0.20*len(test_dataset))]
        # Account for rounding down causing images to get dropped
        split[0] += len(train_dataset) - sum(split)  # Should not be a problem, since 1000 images
        
        # Split into training and validation datasets
        self.train_data, self.val_data = random_split(train_dataset, split)

    def train_dataloader(self):
        train_data_loader = DataLoader(
            self.train_data, 
            batch_size=self.bs,
            shuffle=True, 
            num_workers=self.workers)
        return train_data_loader
    
    def val_dataloader(self):
        val_data_loader = DataLoader(
            self.val_data, 
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.workers)
        return val_data_loader

    def test_dataloader(self):
        test_data_loader = DataLoader(
            self.test_data, 
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.workers)
        return test_data_loader
    
    def configure_optimizers(self):
        # Use SGD for optimizer (use momentum 0.9, weight decay 0.0001)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)
        # Add poly learning rate so that learning rate decays over epochs
        lambda1 = lambda epoch, max_epoch: (1 - (epoch/max_epoch))**0.9
        scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda1)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}] 
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        # Changing learning rate based on epoch so that learning rate decays over epochs
        scheduler.step(epoch=self.current_epoch, max_epoch=self.max_epoch)
    
    def training_step(self, batch, batch_idx):
        self.epoch = batch_idx
        inputs, masks = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, masks)
        self.log("train_loss", loss)
        return loss
    
    def get_accuracy(self, pred_masks, gt_masks):
        '''Get an accuracy by comparing accurate pixels and also a dice score accuracy'''
        pred_masks = (pred_masks > 0.5).float()

        # Calculate dice score (may be more accurate for images with a lot of background)
        # self.dice_score += (2 * (pred_masks * gt_masks).sum())/((pred_masks + gt_masks).sum() + 1e-10)
        # Leave out dice score since it wasn't used in the paper

        # Get number of correct pixels
        self.correct += (pred_masks == gt_masks).sum()
        # Count total pixels
        self.pixels += torch.numel(pred_masks)
        # Compute accuracy
        pixel_accuracy = 100*(self.correct/self.pixels)
        
        return pixel_accuracy, self.dice_score

    def validation_step(self, batch, batch_idx):
        inputs, masks = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, masks)
        accuracy = self.get_accuracy(outputs, masks)

        # pred_indices = outputs.argmax(dim=1, keepdim=True)
        # matches = pred_indices.eq(masks.view_as(pred_indices)).sum().item()
        # accuracy = matches / (len(masks) * 1.0)
        self.log("val_loss", loss)   # more recent versions of lighting use this method to log
        #self.log("val_correct", matches) # more recent versions of lighting use this method to log
        self.log("val_acc", accuracy) # more recent versions of lighting use this method to log
        total = len(masks)
        # return {"val_loss": loss, "correct": matches, "total": total}  # Don't return logs in recent versions of lightning
        return {"val_loss": loss}
        #return loss
    
    def test_step(self, batch, batch_idx):
        inputs, masks = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, masks)
        accuracy = self.get_accuracy(outputs, masks)
        # pred_indices = outputs.argmax(dim=1, keepdim=True)
        # matches = pred_indices.eq(masks.view_as(pred_indices)).sum().item()
        # accuracy = matches / (len(masks) * 1.0)
        self.log("test_loss", loss)   # more recent versions of lighting use this method to log
        #self.log("test_correct", matches) # more recent versions of lighting use this method to log
        self.log("test_acc", accuracy) # more recent versions of lighting use this method to log
        return loss
    
    
    def training_epoch_end(self, outputs):
        average_training_loss = torch.stack([x["loss"] for x in outputs]).mean()

        # To get plotting per epoch, need to add loss and accuracy as scalars
        self.logger.experiment.add_scalar("Train_Loss", average_training_loss, self.current_epoch)

        # For plot comparing training and validation loss
        self.running_train_loss.append(average_training_loss)
    
    
    def validation_epoch_end(self, outputs):
        average_validation_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        #average_validation_loss = torch.stack(outputs).mean()
        
        # Calculate accuracy from correct / total predictions
        # correct = sum([x["correct"] for x in outputs])
        # total = sum([x["total"] for x in outputs])
        # accuracy = correct / total

        # Compute accuracy
        accuracy = 100*(self.correct/self.pixels)
        # Reset values for test dataset
        self.correct = 0
        self.pixels = 0

        # Create tensorboard dictionary for log info
        tensorboard_logs = {"loss": average_validation_loss, "accuracy": accuracy}
        # Log loss and accuracy and tensorboard info
        self.log("avg_val_loss", average_validation_loss) 
        self.log("accuracy", accuracy)
        self.log("log", tensorboard_logs)

        # To get plotting per epoch, need to add loss and accuracy as scalars
        self.logger.experiment.add_scalar("Val_Loss", average_validation_loss, self.current_epoch)

        self.logger.experiment.add_scalar("Val_Accuracy", accuracy, self.current_epoch)

        # For plot comparing training and validation loss
        self.running_val_loss.append(average_validation_loss)
        
        # For tensorboard loss by epoch, need to create and epoch dictionary
        epoch_dict = {
            # This is required, apparently
            "loss": average_validation_loss
        }
        return epoch_dict
    

def compare_loss_plots(train_loss, val_loss, title="Training vs Validation Loss", save_file=False, outfile="train-val_loss_plot.jpg"):
    """A function to plot training and validation running losses on the same plot for a side-by-side comparison"""
    plt.figure()
    plt.title(title)     
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend(loc="upper right")

    if save_file:
        # Save image to disk
        plt.savefig(outfile)
    else:
        plt.show()


def main():
    #### Set parameters for training and validation ####
    bs = 16  # Same as used in the paper 
    workers = 4
    epochs = 7 
    learning_rate = 0.007  # Initial learning rate
    # Set path to data directory (where folders of images have been downloaded to)
    data = "../../data/totaltext/"
    # Set random seed for consistency in experimentation
    randomseed = 42
    ####################################################

    # Get arguments passed in command line call
    args, args_other = parameters()

    # Instantiate the network
    model = RSCANet(bs=bs, lr=learning_rate, workers=workers, epochs=epochs, data_dir=data, seed=randomseed)

    if args.quick_test:
        # For quick testing of a single batch run below instead: 
        trainer = Trainer(fast_dev_run=True, gpus=1, default_root_dir="../RSCAnet_checkpoints/")
    else:
        # Train (all batches)
         trainer = Trainer(max_epochs=epochs, gpus=1, default_root_dir="../RSCAnet_checkpoints/")

    # Train and validate
    trainer.fit(model)
    # Same as: trainer.fit(model, model.train_dataloader, model.val_dataloader)

    # Plot training loss and validation loss on the same plot for side-by-side comparison
    version = trainer.logger.version
    outpath = f"lightning_logs/version_{version}/train-val_loss_plot.jpg"
    compare_loss_plots(train_loss=model.running_train_loss, val_loss=model.running_val_loss, save_file=True, outfile=outpath)
    
    # Run tests: Will use the best checkpoint automatically (best training)
    # Checkpoints are where model weights are stored for pytorch lighting
    # path_to_checkpoint = glob(f"../RSCAnet_checkpoints/lightning_logs/version_{version}/checkpoints/*.ckpt")[0]
    # loaded_model = RSCANet.load_from_checkpoint(path_to_checkpoint)
    ## Uncomment 2 lines above only if needed to load from existing model 
    ## (otherwise trained model already exists from above)
    trainer = Trainer(default_root_dir="../RSCAnet_checkpoints/")
    trainer.test(model)

    # # Now move model to CPU
    # model.to("cpu")
    # test_data = model.test_data
    # # test_dataloader.to("cpu") # no attribute "to"

    # torch.jit.save(model.to_torchscript(), f"lightning_logs/version_{version}/model_trained.pt")
    # # Save out the test_data_loader so that we can use the unseen data for testing on the pi
    # torch.save(test_data, f"lightning_logs/version_{version}/model_test_data.pt")


if __name__ == "__main__":
    main()