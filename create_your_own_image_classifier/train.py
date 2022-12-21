import argparse
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict






parser = argparse.ArgumentParser()

parser.add_argument('data_dir', type = str,
                    help = 'data directory')
parser.add_argument('--save_dir', type = str, default = './',
                    help = 'directory to save')
parser.add_argument('--arch', type = str, default = 'densenet121',
                    help = 'densenet121 or vgg13')
parser.add_argument('--learning_rate', type = float, default = 0.001,
                    help = 'Learning rate')
parser.add_argument('--hidden_units', type = int, default = 512,
                    help = 'no. of hidden units')
parser.add_argument('--epochs', type = int, default = 3,
                    help = 'epochs')
parser.add_argument('--gpu', action='store_true',
                    help = "to use GPU")


args_in = parser.parse_args()





device = torch.device("cuda" if args_in.gpu else "cpu")






data_dir = args_in.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)





import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)




if args_in.arch == 'densenet121':

    model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(1024, args_in.hidden_units)),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(p=0.5)),
                            ('fc2', nn.Linear(args_in.hidden_units, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))


else:

    model = models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, args_in.hidden_units)),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(p=0.5)),
                            ('fc2', nn.Linear(args_in.hidden_units, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=args_in.learning_rate)

model.to(device);




epochs = args_in.epochs
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(validloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()





test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        test_loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Test loss: {test_loss/len(testloader):.3f}.. "
      f"Test accuracy: {accuracy/len(testloader):.3f}")





model.class_to_idx = train_data.class_to_idx

checkpoint = {'input_size': 1024,
              'output_size': 102,
              'arch': args_in.arch,
              'learning_rate': args_in.learning_rate,
              'classifier' : model.classifier,
              'epochs': args_in.epochs,
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict()
             }

save_path = args_in.save_dir + 'checkpoint.pth'
torch.save(checkpoint, save_path)
