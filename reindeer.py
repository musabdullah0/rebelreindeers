
'''
1. feed images through yolo and get outputs
2. feed images through restnet18 to get features out
3. concat output from above two steps
4. train with neural part of cspn
5. test cspn with images
'''

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io, transform
import pandas as pd

# cspn stuff
from rat_cspn import CSPN
from region_graph import RegionGraph

class KittiDataset(Dataset):
    def __init__(self, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_dir = './dataset/training/label_2/'
        self.images_dir = './dataset/training/image_2/'
        self.transform = transform
        self.df = pd.read_csv('yolo-outputs.csv')
        self.types = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2, 'Tram': 0, 'Person_sitting': 1, 'Misc': 3, 'DontCare': 3, 'Van': 0, 'Truck': 0}


    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, idx):
        row = self.df.iloc[idx].to_numpy() # get row out of big csv
        
        # getting image
        img_name = row[0]
        image = io.imread(img_name)

        # getting ground truth label
        index = row[1]
        label_name = img_name.replace('image_2', 'label_2').replace('png', 'txt')
        label = ''
        with open(label_name, 'r') as f:
            lines = f.readlines()
            if index >= len(lines):
                # bad label bc yolo found more objects than reality
                label = "Pedestrian 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00"
            else:
                label = lines[index]
        label = label.split()
        label[0] = self.types[label[0]]
        label = [float(x) for x in label]
        label = torch.Tensor(label)

        # getting yolo outputs
        yolo_outputs = row[2]
        yolo_outputs = yolo_outputs.split(',')
        yolo_outputs = [float(x) for x in yolo_outputs]
        yolo_outputs = yolo_outputs[:-1]
        yolo_outputs = torch.Tensor(yolo_outputs)
        
        if self.transform:
            image = transform(image)

        return image, yolo_outputs, label


device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
resnet_model = models.resnet18()
transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize(size=(375, 1250)),
                                    transforms.ToTensor()
                                ])
dataset = KittiDataset(transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
loss = nn.NLLLoss()

input_size = 1000 + 15 # input_size = resnet outputs size + yolo outputs size
num_RVs = 15 # num of RVs

rg = RegionGraph(range(num_RVs)) 

for _ in range(8):
    rg.random_split(2, 2)

cspn = CSPN(rg, input_size) 

generic_nn = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

cspn.make_cspn(generic_nn, 64)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = resnet_model
        self.cspn = cspn

    def forward(self, image, yolo_outputs, label):
        res_output = self.resnet(image)
        label = label.to(device)
        conditional = torch.cat((res_output, yolo_outputs), dim=1).to(device)
        log_prob = self.cspn.forward(inputs=label, conditional=conditional)
        return log_prob

model = Net()
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

def train():
    for image, yolo_outputs, label in dataloader:
        
        log_prob = model.forward(image, yolo_outputs, label)
        log_prob = torch.unsqueeze(log_prob, 2) # idk if this is correct
        # print('log_prob', log_prob, log_prob.shape, log_prob.dtype)
        
        target = torch.tensor([[0], [0], [0], [0]]).to(device, torch.long)
        # print('target', target, target.shape, target.dtype)
        
        diff = loss(log_prob, target)
        print('loss', diff)

        diff.backward()
        optimizer.step()

train()

        

'''
command to run yolo and save 100 images
python yolo-pytorch/src/test.py --gpu_idx 0 --pretrained_path complex_yolov4_mse_loss.pth --cfgfile yolo-pytorch/src/config/cfg/complex_yolov4.cfg --save_test_output --num_samples 100


Questions
- what if objects are labeled in different order than in yolo outputs?
- why is loss not going down?
- how are we using NLLLoss?
- what is the region graph random split for?


forward: P (input | conditional)
what is conditional and what is input?
P (yolo outputs | image features)
P (label | concat of yolo output and image features)

use features from yolo instead of resnet


for every yolo output
    P (yolo output | image features)

NLL loss

only have to train on ground truth, once, don't have to go thru wrong ones

'''