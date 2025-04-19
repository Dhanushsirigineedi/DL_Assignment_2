# -*- coding: utf-8 -*-
"""dl-assignment-2-part-a-ipynb (5).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1iCesBxnk7WBr7qCi87ch876ETVgv03W1
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
from torchvision import datasets
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

# !pip install --upgrade wandb
import wandb
# import socket
# socket.setdefaulttimeout(30)
# wandb.login(key='1d2423ec9b728fe6cc1e2c0b9a2af0e67a45183c')

# from google.colab import drive
# drive.mount('/content/drive')

# path in kaggle for the datasets
# train_directory='/kaggle/input/dataset2/inaturalist_12K/train'
# test_directory='/kaggle/input/dataset2/inaturalist_12K/val'

class CNN(nn.Module):
    def __init__(self, no_of_input_channels=3, no_of_classes=10, no_of_filters=[32,32,32,32,32], size_of_filter=[3,3,3,3,3],
                 no_of_neurons=128, activation_function='sigmoid',dropout_probability=0.0, batch_normalization='no'):
        super(CNN, self).__init__()
        self.activation_function_name = activation_function
        self.batch_normalization = batch_normalization

        width = height = 256.0 # Initialize width and height for feature map calculations

        # Create convolutional, batch norm, and pooling layers dynamically
        for i in range(len(no_of_filters)):
            # Conv layer
            conv_layer = nn.Conv2d(in_channels=no_of_input_channels if i == 0 else no_of_filters[i-1],
                                   out_channels=no_of_filters[i],kernel_size=size_of_filter[i],stride=1)
            setattr(self, f'conv_layer{i+1}', conv_layer)

            width = height = (width - size_of_filter[i]) + 1  # Update feature map dimensions after convolution
            # Batch norm layer
            if batch_normalization == 'yes':
                batch_norm = nn.BatchNorm2d(no_of_filters[i])
                setattr(self, f'batch_norm{i+1}', batch_norm)
            # Pooling layer
            pool_layer = nn.MaxPool2d(kernel_size=size_of_filter[i], stride=2)
            setattr(self, f'pool_layer{i+1}', pool_layer)
            width = height = math.floor((width - size_of_filter[i]) / 2) + 1 # Update feature map dimensions after pooling
        # Fully connected layers
        self.dropout = nn.Dropout(p=dropout_probability)
        self.full_connected1 = nn.Linear(no_of_filters[-1] * int(width) * int(height), no_of_neurons)
        if batch_normalization == 'yes':
            self.batch_norm6 = nn.BatchNorm1d(no_of_neurons)
        self.full_connected2 = nn.Linear(no_of_neurons, no_of_classes)

    def forward(self, x):
      # Set activation function
      if(self.activation_function_name == 'relu'):
            activation_function = F.relu
      elif(self.activation_function_name == 'gelu'):
          activation_function = F.gelu
      elif(self.activation_function_name == 'silu'):
          activation_function = F.silu
      else:
            activation_function = F.mish

      # Process through 5 convolutional blocks
      for i in range(1, 6):
          conv_layer = getattr(self, f'conv_layer{i}')
          if self.batch_normalization == 'yes':
              batch_norm = getattr(self, f'batch_norm{i}')
              x = activation_function(batch_norm(conv_layer(x)))
          else:
              x = activation_function(conv_layer(x))
          pool_layer = getattr(self, f'pool_layer{i}')
          x = pool_layer(x)
      # Flatten the output
      x = x.reshape(x.shape[0], -1)
      # First fully connected layer
      if self.batch_normalization == 'yes':
          x = activation_function(self.batch_norm6(self.full_connected1(x)))
      else:
          x = activation_function(self.full_connected1(x))
      x = self.dropout(x)
      x = self.full_connected2(x)
      return x

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

def train_the_model(no_of_neurons, no_of_filters, size_of_filter, activation_function_name, optimizer_name, batch_size,
                   dropout_probability, no_of_epochs, learning_rate, batch_normalization, augmentation_flag,train_directory,test_directory):
    no_of_input_channels = 3
    no_of_classes = 10

    train_loader, val_loader = data_loader_creator(augmentation_flag, batch_size,train_directory,test_directory)  # getting dataloaders
    # Uncomment the below line for test data loader
    # test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=True)

    model = CNN(no_of_input_channels, no_of_classes, no_of_filters, size_of_filter, no_of_neurons,
                activation_function_name, dropout_probability, batch_normalization).to(device)
    # model=nn.DataParallel(model)
    # model=model.to(device)

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.NAdam(model.parameters(), lr=learning_rate)  # optimzers selection
    criterion = nn.CrossEntropyLoss()  # since it is classification problem corss entropy loss is used.

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

    parser.add_argument('-n', '--no_of_neurons', type=int, default=128,
                      choices=[128, 256, 512], help='Number of neurons in dense layer')

    parser.add_argument('-nF', '--no_of_filters', type=str, default='32,64,128,256,512',
                      help='Number of filters per layer as comma-separated values')

    parser.add_argument('-sF', '--size_of_filter', type=str, default='3,3,3,3,3',
                      help='Filter sizes per layer as comma-separated values')

    parser.add_argument('-aF', '--activation_function_name', type=str, default='gelu',
                      choices=['relu','gelu','silu','mish'], help='Activation function type')

    parser.add_argument('-opt', '--optimizer_name', type=str, default='nadam',
                      choices=['adam','nadam'], help='Optimizer type')

    parser.add_argument('-bS', '--batch_size', type=int, default=32,
                      choices=[32, 64, 128], help='Batch size for training')

    parser.add_argument('-d', '--dropout_probability', type=float, default=0.4,
                       help='Dropout probability')

    parser.add_argument('-nE', '--no_of_epochs', type=int, default=10,
                       help='Number of training epochs')

    parser.add_argument('-lR', '--learning_rate', type=float, default=0.001,
                       help='Learning rate')

    parser.add_argument('-bN', '--batch_normalization', type=str, default='yes',
                      choices=['yes','no'], help='Whether to use batch normalization')

    parser.add_argument('-ag', '--augmentation_flag', type=str, default='no',
                      choices=['yes','no'], help='Whether to use data augmentation')

    return parser.parse_args()

args = parse_arguments()
args.no_of_filters = [int(x) for x in args.no_of_filters.split(',')]
args.size_of_filter = [int(x) for x in args.size_of_filter.split(',')]
train_directory = os.path.join(args.root_dir, 'train')  # root_dir/train
test_directory = os.path.join(args.root_dir, 'val')     # root_dir/val
wandb.init(project=args.wandb_project)

wandb.run.name = (
    f"No_of_neurons: {args.no_of_neurons}, "
    f"No_of_filters: {args.no_of_filters}, "
    f"Size_of_filter: {args.size_of_filter}, "
    f"Activation_function: {args.activation_function_name}, "
    f"Optimizer: {args.optimizer_name}, "
    f"Batch_size: {args.batch_size}, "
    f"Dropout: {args.dropout_probability}, "
    f"No_of_epochs: {args.no_of_epochs}, "
    f"Learning_Rate: {args.learning_rate}, "
    f"Batch_normalization: {args.batch_normalization}, "
    f"Augmentation_flag: {args.augmentation_flag}"
)

train_the_model(
    args.no_of_neurons,
    args.no_of_filters,
    args.size_of_filter,
    args.activation_function_name,
    args.optimizer_name,
    args.batch_size,
    args.dropout_probability,
    args.no_of_epochs,
    args.learning_rate,
    args.batch_normalization,
    args.augmentation_flag,
    train_directory,
    test_directory
)