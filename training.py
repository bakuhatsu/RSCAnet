###############################################################################################
# Sven Nelson
# ECE 570: Artificial Intelligence
# Student ID: nelso531
# Date: 10/1/2022
#
# Reimplementing SCPNET based on the paper:
# Xie, Enze, Yuhang Zang, Shuai Shao, Gang Yu, Cong Yao, and Guangyao Li. “Scene Text Detection with Supervised Pyramid Context Network.” Proceedings of the AAAI Conference on Artificial Intelligence 33, no. 1 (April 3, 2019): 9038–45. https://doi.org/10.1609/aaai.v33i01.33019038.
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
#from torchvision.models.quantization.resnet import resnet18  # Use this for resnet18 model
#from resnet import resnet18
# Useful link to list of models: https://pytorch.org/vision/stable/models.html
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import QuantizationAwareTraining, ModelCheckpoint #, ModelPruning, 
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import copy


def parameters():
    parser = argparse.ArgumentParser(description="NutriVision Trainer")
    # Run quick-test mode or full training
    parser.add_argument("--quick_test", help="Runs only a single batch of train, val, and test for testing code,", action="store_true")
    # For full precision (normal) training
    parser.add_argument("--quantize", help="Default is to train with quantization, this option will train without quantization", action="store_true") 
    args, args_other = parser.parse_known_args()
    return (args, args_other)


# Define DataLoader for fruit-262 dataset
class ProduceDataset(Dataset):
    def __init__(self, data_dir="../data/", transform=None):
        # Create an empty list for image file paths
        self.images = []
        # Create an empty list for labels 0 or 1
        self.labels = []
        # Get list of folders in data directory, these are the classes
        self.classes = os.listdir(data_dir)
        # Loop across folders names, which are the labels (will work for any number of classes)
        for label, image_class in enumerate(self.classes):
            # Define path to image folder (aka folder for a class)
            path = os.path.join(data_dir, image_class)
            # Get a list of all jpeg files in the folder
            img_list = glob(f"{path}/*.jpg")
            # For each image file
            for img in img_list:
                # Define image path
                #img_path = os.path.join(path, img)  # glob step already gets path
                # Append image path to the images list
                self.images.append(img)
                # Append the image class label (0:apple, 1:orange, ...) to the labels list
                self.labels.append(label)
        # Set the transform equal to the passed transform
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Import image (in height x width format)
        img_HW = Image.open(self.images[idx]).convert("RGB")
        # Some images in the dataset were grayscale, so the above forces them to import as 3-channel images
        
        # Rearrange to CxHxW format and normalize values from 0-255 to 0-1
        # and apply necessary transformation to change 0-1 into range from -1-1
        img_CHW = self.transform(img_HW)

        # Do not one-hot encode labels for CrossEntropyLoss()
        # CrossEntropyLoss(input, target) takes LongTensor targets and FloatTensor inputs 
        # label as integer value
        label = self.labels[idx]

        # Return processed image and label
        return (img_CHW, label)


# For pytorch lightning, everything goes inside of the network class
class NutriVisionNet(LightningModule):
    def __init__(self, num_classes=262, bs=10, lr=1e-3, workers=4, data_dir="../data/"):
        super().__init__()
        # Load resnet18 model (not pretrained)
        resnet_model = resnet18(pretrained=False)
        # Get number of features
        num_ftrs = resnet_model.fc.in_features
        # Change last layer such that it will have the desired number of classes
        resnet_model.fc = nn.Linear(num_ftrs, num_classes)
        # if model is not None:
        #     self.net = resnet_model
        # else:
        #     self.net = model
        self.net = resnet_model
        self.data_dir = data_dir
        # Set batch size based on value passed
        self.bs = bs
        # Set learning rate based on value passed
        self.lr = lr
        # Set number of workers based on value passed
        self.workers = workers
        # Set criterion for loss to CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss()
        # For plotting training vs validation loss
        self.running_train_loss = []
        self.running_val_loss = []

    def forward(self,x):
        output = self.net(x)
        return output

    def prepare_data(self):   
        # Transform to match expected input image format for resnet
        # Comment below from: https://pytorch.org/hub/pytorch_vision_resnet/
        # All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        transform = tfm.Compose([
            tfm.Resize(256),
            tfm.CenterCrop(224),
            tfm.ToTensor(),
            tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Define produce dataset
        prod_dataset = ProduceDataset(self.data_dir, transform)
        # Define split: 70:20:10
        split = [int(0.7*len(prod_dataset)), int(0.2*len(prod_dataset)), int(0.1*len(prod_dataset))]
        # Account for rounding down causing images to get dropped
        split[0] += len(prod_dataset) - sum(split)
        
        # Split into training and validation datasets
        self.train_data, self.val_data, self.test_data = random_split(prod_dataset, split)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer 
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)
        pred_indices = outputs.argmax(dim=1, keepdim=True)
        matches = pred_indices.eq(labels.view_as(pred_indices)).sum().item()
        accuracy = matches / (len(labels) * 1.0)
        self.log("val_loss", loss)   # more recent versions of lighting use this method to log
        #self.log("val_correct", matches) # more recent versions of lighting use this method to log
        self.log("val_acc", accuracy) # more recent versions of lighting use this method to log
        total = len(labels)
        return {"val_loss": loss, "correct": matches, "total": total}  # Don't return logs in recent versions of lightning
        #return loss
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)
        pred_indices = outputs.argmax(dim=1, keepdim=True)
        matches = pred_indices.eq(labels.view_as(pred_indices)).sum().item()
        accuracy = matches / (len(labels) * 1.0)
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
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])
        accuracy = correct / total

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
    #### For reproducibility to use same seed #########
    seed = 0
    # random.seed(seed) # I did not use random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmarks=False
    os.environ['PYTHONHASHSEED'] = str(seed)
    ####################################################

    #### Set parameters for training and validation ####
    bs = 64  # Previously 10
    workers = 4
    epochs = 7 #7  # Trained for 20, but getting overfitting.  Lowest validation loss occured at 7 epochs.
    learning_rate = 1e-3
    # Set path to data directory (where folders of images have been downloaded to)
    data = "../data/"
    #data = "../data_temp/"
    # Define number of classes based on number of folders in the data directory (each folder is a class)
    number_of_classes = len(os.listdir(data))
    ####################################################

    # Get arguments passed in command line call
    args, args_other = parameters()

    # Instantiate the network
    nutrinet = NutriVisionNet(num_classes=number_of_classes, bs=bs, lr=learning_rate, workers=workers, data_dir=data)

    if args.quick_test:
        # For quick testing of a single batch run below instead: 
        trainer = Trainer(fast_dev_run=True, gpus=1)
    else:
        # Else, train with quantization
        if args.quantize:  # This option did not produce good results, requires further testing
            # Instantiate the trainer
            trainer = Trainer(max_epochs=epochs, gpus=1, callbacks=QuantizationAwareTraining())
        # Train with full precision
        else: 
            trainer = Trainer(max_epochs=epochs, gpus=1)

    # Train and validate
    trainer.fit(nutrinet)
    # This should be the same as: trainer.fit(nutrinet, nutrinet.train_dataloader, nutrinet.val_dataloader)

    # Plot training loss and validation loss on the same plot for side-by-side comparison
    version = trainer.logger.version
    outpath = f"lightning_logs/version_{version}/train-val_loss_plot.jpg"
    compare_loss_plots(train_loss=nutrinet.running_train_loss, val_loss=nutrinet.running_val_loss, save_file=True, outfile=outpath)
    
    # Run tests: Will use the best checkpoint automatically (best training)
    # Checkpoints are where model weights are stored for pytorch lighting
    #path_to_checkpoint = glob(f"lightning_logs/version_{version}/checkpoints/*.ckpt")[0]
    #loaded_model = NutriVisionNet.load_from_checkpoint(path_to_checkpoint)
    #loaded_model = copy.deepcopy(nutrinet)
    trainer = Trainer()
    trainer.test(nutrinet)

    # For Quantization
    #nutrinet.to("cpu")
    # quant_nutrinet = copy.deepcopy(nutrinet.net)
    # # quant_nutrinet.train()  # Need to switch to train mode to fuse layers

    # # # Perform the steps to fuse model layers for resnet18 architecture
    # # quant_nutrinet = torch.quantization.fuse_modules(quant_nutrinet, [["conv1", "bn1", "relu"]], inplace=True)
    # # for module_name, module in quant_nutrinet.named_children():
    # #     if "layer" in module_name:
    # #         for basic_block_name, basic_block in module.named_children():
    # #             torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
    # #             for sub_block_name, sub_block in basic_block.named_children():
    # #                 if sub_block_name == "downsample":
    # #                     torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)
    
    # quant_nutrinet.eval()  # Switch to eval mode

    # # Use the fused model to generate the quantized model
    # nutrinet_q = QuantNutriVisionNet(
    #     num_classes=number_of_classes, 
    #     bs=bs, 
    #     lr=learning_rate, 
    #     workers=workers, 
    #     data_dir=data, 
    #     model=quant_nutrinet)
    
    # # Use fbgemm quantization scheme
    # quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    # nutrinet_q.qconfig = quantization_config
    # torch.quantization.prepare_qat(nutrinet_q, inplace=True)

    # # Put model in train mode
    # nutrinet_q.train()

    # # Now move model to CPU
    nutrinet.to("cpu")
    test_data = nutrinet.test_data
    # test_dataloader.to("cpu") # no attribute "to"

    # # And prepare for export to raspberry pi
    # nutrinet_q = torch.quantization.convert(nutrinet_q, inplace=True)
    # # Put into eval mode
    # nutrinet_q.eval()

    # Save out to run tests on pi
    #save_torchscript_model(model=quantized_model, model_dir=model_dir, model_filename=quantized_model_filename)
    #torch.jit.save(torch.jit.script(model), model_filepath)

    torch.jit.save(nutrinet.to_torchscript(), f"lightning_logs/version_{version}/nutrinet_trained.pt")
    # Save out the test_data_loader so that we can use the unseen data for testing on the pi
    torch.save(test_data, f"lightning_logs/version_{version}/nutrinet_test_data.pt")


if __name__ == "__main__":
    main()