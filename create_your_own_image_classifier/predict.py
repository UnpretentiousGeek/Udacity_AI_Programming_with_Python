import argparse
import numpy as np


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

from PIL import Image



parser = argparse.ArgumentParser()

parser.add_argument('img_path', type = str, 
                    help = 'path to image')
parser.add_argument('check_path', type = str, 
                    help = 'path to checkpoint file')

parser.add_argument('--category_names', type = str,
                    help = 'mapping of classes to index')
parser.add_argument('--top_k', type = int, default = 5,
                    help = 'top K prob.')

parser.add_argument('--gpu', action='store_true',
                    help = "to use gpu")

args_in = parser.parse_args()




device = torch.device("cuda" if args_in.gpu else "cpu")







def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        model = models.vgg19(pretrained = True)
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

model = load_checkpoint(args_in.check_path)
model.to(device);




def process_image(img):

    
    size = []
    
    if img.size[0] > img.size[1]:
        size = [img.size[0], 256]
        
    else:
        size = [256, img.size[1]]
        
    img.thumbnail(size)
    
    width, height = img.size
    
    left = (width - 224)/2
    upper = (height - 224)/2
    right = left + 224
    lower = upper + 224
    
    img = img.crop((left, upper, right, lower))
    np_image = np.array(img)
    
    np_image = np_image/255
    
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image-means)/stds
    
    return np.transpose(np_image, (2, 0, 1))




def predict(image_path, model, topk=5):
    
    model.eval()
    img = Image.open(image_path)
    img = process_image(img)
    img = torch.from_numpy(img)
    img = img.unsqueeze_(0)
    img = img.float()
    
    with torch.no_grad():
        output = model.forward(img.cuda())
        
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk)
    top_p = np.array(top_p)[0]
    top_class = np.array(top_class)[0]
    
    class_index_dict = {}
    for i, j in model.class_to_idx.items():
        class_index_dict[j] = class_index_dict.get(j, i)
    
    top_class2 = [class_index_dict[i] for i in top_class]
    
    return top_p, top_class2



top_p, top_class = predict(args_in.img_path, model, topk=args_in.top_k)



if args_in.category_names:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    names = [cat_to_name[i] for i in top_class]
    print(names)

print(top_class)
print(top_p)