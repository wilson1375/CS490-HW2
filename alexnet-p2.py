#!/usr/bin/env python
# coding: utf-8

# In[15]:


import torch

# In[16]:


torch.cuda.is_available()

# In[3]:


torch.cuda.get_device_name(0)

# In[18]:


torch.cuda.mem_get_info()

# In[17]:


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if device.type=='cuda':
    print(torch.cuda.get_device_name(0))

# In[19]:


CLASSES=10
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# In[6]:


import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import numpy as np

# In[7]:

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()

        # L1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.local_response1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # L2
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1, padding=2, stride=1)
        self.local_response2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # L3
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1, padding=1, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout3 = nn.Dropout(0.5)

        # L4
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=9, padding=1, stride=4)

        # L5
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=1, stride=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout5 = nn.Dropout(0.5)

        # L6
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)

        # L7
        self.conv7 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1, stride=1)

        # L8
        self.conv8 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1, stride=1)
        self.dropout8 = nn.Dropout(0.5)

        self.a_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # L9
        self.fc1 = nn.Linear(in_features=384, out_features=4096)

        # L10
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)

        # L11
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.local_response1(self.conv1(x))))
        x = self.pool2(F.relu(self.local_response2(self.conv2(x))))
        x = self.dropout3(self.pool3(F.relu(self.conv3(x))))
        x = F.relu(self.conv4(x))
        x = self.dropout5(self.pool5(F.relu(self.conv5(x))))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.dropout8(F.relu(self.conv8(x)))
        x = self.a_pool(x)
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc3(x)

        return x

# In[8]:


transform_conf=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

# In[9]:


BATCH_SIZE=16
train_dataset=datasets.MNIST('./data_MNIST/',train=True,download=True,transform=transform_conf,)
test_dataset=datasets.MNIST('./data_MNIST/',train=False,download=True,transform=transform_conf)

# In[10]:


train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)

# In[11]:


model=Net().to(device)
optimizer=optim.Adam(params=model.parameters(),lr=0.0001)
loss_fn = nn.CrossEntropyLoss()

# In[12]:


def train(model,device,train_loader,optimizer,epochs):
    print("inside train")
    model.train()
    for batch_ids, (img, classes) in enumerate(train_loader):
        classes=classes.type(torch.LongTensor)
        img,classes=img.to(device),classes.to(device)
        torch.autograd.set_detect_anomaly(True)     
        optimizer.zero_grad()
        output=model(img)
        loss = loss_fn(output,classes)                
        
        loss.backward()
        optimizer.step()
    if(batch_ids +1) % 2 == 0:
        print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
            epochs, batch_ids* len(img), len(train_loader.dataset),
            100.*batch_ids / len(train_loader),loss.item()))

# In[13]:


def test(model, device, test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for img,classes in test_loader:
            img,classes=img.to(device), classes.to(device)
            y_hat=model(img)
            test_loss+=F.nll_loss(y_hat,classes,reduction='sum').item()
            _,y_pred=torch.max(y_hat,1)
            correct+=(y_pred==classes).sum().item()
        test_loss/=len(test_dataset)
        print("\n Test set: Avarage loss: {:.0f},Accuracy:{}/{} ({:.0f}%)\n".format(
            test_loss,correct,len(test_dataset),100.*correct/len(test_dataset)))
        print('='*30)

# In[14]:


if __name__=='__main__':
    seed=42
    EPOCHS=1
    
    for epoch in range(1,EPOCHS+1):
        train(model,device,train_loader,optimizer,epoch)
        test(model,device,test_loader)

# In[ ]:



