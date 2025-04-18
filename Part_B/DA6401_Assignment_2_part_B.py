# -*- coding: utf-8 -*-
"""dl-assignment-2-part-b (1).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HDGvcNIj7AWoXtHbEg7i2mLyAAQZAdxi
"""

# Standard library imports
import math
import argparse
import os
from torch.utils.data import DataLoader, Subset

# Third-party library imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

# Torchvision imports
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder

# !pip install --upgrade wandb
import wandb
# # import socket
# # socket.setdefaulttimeout(30)
# wandb.login(key='your key')

# from google.colab import drive
# drive.mount('/content/drive')

# train_directory='/kaggle/input/dataset2/inaturalist_12K/train'
# test_directory='/kaggle/input/dataset2/inaturalist_12K/val'



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def load_datasets(train_directory, test_directory):
    # Basic transformations
    transform_basic = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to fixed dimensions
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalization
    ])
    
    # Augmentation transformations
    transform_augmented = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_directory, transform=transform_basic)
    train_dataset_aug = datasets.ImageFolder(root=train_directory, transform=transform_augmented)
    test_dataset = datasets.ImageFolder(root=test_directory, transform=transform_basic)
    
    # Split into training and validation sets (80/20)
    training_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [8000, 1999])
    training_dataset_aug, validation_dataset_aug = torch.utils.data.random_split(train_dataset_aug, [8000, 1999])
    
    return (training_dataset, validation_dataset, training_dataset_aug, validation_dataset_aug, test_dataset)

def data_loader_creator(augmentation_flag,batch_size,train_directory,test_directory): # function to return the data loaders depending on augumentation.
    (training_dataset, validation_dataset, training_dataset_aug, validation_dataset_aug, test_dataset) = load_datasets(train_directory, test_directory)
    if(augmentation_flag == 'no'):
        train_loader = torch.utils.data.DataLoader(training_dataset,batch_size =batch_size,shuffle = True,num_workers=2,pin_memory=True)
        val_loader = torch.utils.data.DataLoader(validation_dataset,batch_size =batch_size,shuffle = True,num_workers=2,pin_memory=True)
        return train_loader,val_loader
    else:
        train_loader_aug = torch.utils.data.DataLoader(training_dataset_aug,batch_size =batch_size,shuffle = True,num_workers=4,pin_memory=True)
        val_loader_aug = torch.utils.data.DataLoader(validation_dataset_aug,batch_size =batch_size,shuffle = True,num_workers=4,pin_memory=True)
        return train_loader_aug,val_loader_aug

# Initializes a pretrained ResNet50 model and modifies it for transfer learning.
def RESNET50(NUM_OF_CLASSES):
    # Load pretrained ResNet50 model
    model = models.resnet50(pretrained=True)

    # Modify the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_OF_CLASSES)

    # Freeze all layers except the final one
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Unfreeze the final layer
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model

# Initializes a pretrained ResNet50 model and freezes first k layers for transfer learning.
def RESNET50_1(k, NUM_OF_CLASSES):
    # Load pretrained ResNet50 model
    model = models.resnet50(pretrained=True)

    # Freeze first k layers
    model_parameters = list(model.parameters())
    for parameter in model_parameters[:k]:
        parameter.requires_grad = False

    # Modify the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_OF_CLASSES)

    return model

# Initializes pretrained ResNet50 with added dense layer and selective freezing.
def RESNET50_2(neurons_dense, NUM_OF_CLASSES):
    # Load pretrained model
    model = models.resnet50(pretrained=True)

    # Freeze all original parameters
    for param in model.parameters():
        param.requires_grad = False

    # Modify final layers
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, neurons_dense),  # Additional dense layer
        nn.ReLU(),                              # Activation
        nn.Dropout(0.4),                        # Fixed dropout
        nn.Linear(neurons_dense, NUM_OF_CLASSES) # Final output layer
    )

    # Unfreeze only the new sequential layers
    for param in model.fc.parameters():
        param.requires_grad = True

    return model

def Accuracy_calculator(loader, model, criterion, batch_size):
    """Computes model accuracy and average loss on a given dataset loader"""
    correct_predictions = 0
    total_samples = 0
    accumulated_loss = 0.0
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in loader:
            # Move data to the appropriate device
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Update metrics
            accumulated_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += predicted.size(0)
    model.train()  # Restore model to training mode
    accuracy = (correct_predictions / total_samples) * 100
    average_loss = accumulated_loss / total_samples
    return accuracy, average_loss

def train_the_model(batch_size,no_of_epochs,learning_rate,augmentation_flag,strategy_flag,NUM_OF_CLASSES,train_directory,test_directory):

    train_loader,val_loader = data_loader_creator(augmentation_flag,batch_size,train_directory,test_directory)  # getting dataloaders.

    # Uncomment the below line for test data loader
    # test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=True)


    no_of_input_channels=3
    no_of_classes=10

    if(strategy_flag == 0):
        model = RESNET50(NUM_OF_CLASSES).to(device)
    elif(strategy_flag == 1):
        model = RESNET50_1(10,NUM_OF_CLASSES).to(device)
    else:
        model = RESNET50_2(256,NUM_OF_CLASSES).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss() # since it is classification problem corss entropy loss is used.

    for epoch in range(no_of_epochs):
        for batchId, (input_images, target_classes) in enumerate(tqdm(train_loader)):
            input_images = input_images.to(device=device)
            target_classes = target_classes.to(device=device)
            # forward
            scores = model(input_images)  # give the last layer pre-activation values.
            loss = criterion(scores, target_classes)  # gets the overll cross entropy loss for each batch

            optimizer.zero_grad()  # gradients are made to zero for each batch.
            loss.backward()  # calculaing the gradients
            optimizer.step()  # updates the parameters

        training_accuracy, training_loss = Accuracy_calculator(train_loader, model, criterion, batch_size)
        validation_accuracy, validation_loss = Accuracy_calculator(val_loader, model, criterion, batch_size)

        # Uncomment the below lines for test data evaluation
        # test_accuracy, test_loss = Accuracy_calculator(test_loader, model, criterion, batch_size)
        # print(f"test_accuracy:{test_accuracy:.4f},test_loss:{test_loss:.4f}")
        # wandb.log({'test_accuracy': test_accuracy})
        # wandb.log({'test_loss': test_loss})

        print(f"training_accuracy:{training_accuracy:.4f},training_loss:{training_loss:.4f}")
        print(f"validation_accuracy:{validation_accuracy:.4f},validation_loss:{validation_loss:.4f}")
        wandb.log({'training_accuracy': training_accuracy})  # plotting the data in wandb
        wandb.log({'training_loss': training_loss})
        wandb.log({'validation_accuracy': validation_accuracy})
        wandb.log({'validation_loss': validation_loss})

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training_Parameters')

    parser.add_argument('-wp', '--wandb_project', type=str, default='DA6401_Assignment_2',
                      help='Project name used to track experiments in Weights & Biases dashboard')
    
    parser.add_argument('-rd', '--root_dir', type=str,default= '/kaggle/input/dataset2/inaturalist_12K' ,
                      help='Root directory containing train/val folders')

    parser.add_argument('-bS', '--batch_size', type=int, default=32,
                      choices=[32, 64, 128], help='Batch size for training')

    parser.add_argument('-nE', '--no_of_epochs', type=int, default=10,
                       help='Number of training epochs')

    parser.add_argument('-lR', '--learning_rate', type=float, default=0.001,
                       help='Learning rate')

    parser.add_argument('-ag', '--augmentation_flag', type=str, default='no',
                      choices=['yes','no'], help='Whether to use data augmentation')

    parser.add_argument('-st', '--strategy_flag', type=int, default=2,
                      choices=[0,1,2], help='Training strategy to use')

    return parser.parse_args()

args = parse_arguments()
wandb.init(project=args.wandb_project)
train_directory = os.path.join(args.root_dir, 'train')  # root_dir/train
test_directory = os.path.join(args.root_dir, 'val')     # root_dir/val

wandb.run.name = (
    f"Batch_size: {args.batch_size}, "
    f"No_of_epochs: {args.no_of_epochs}, "
    f"Learning_Rate: {args.learning_rate}, "
    f"Augmentation_flag: {args.augmentation_flag}, "
    f"Strategy: {args.strategy_flag}"
)

train_the_model(
    args.batch_size,
    args.no_of_epochs,
    args.learning_rate,
    args.augmentation_flag,
    args.strategy_flag,
    10,
    train_directory,
    test_directory
)