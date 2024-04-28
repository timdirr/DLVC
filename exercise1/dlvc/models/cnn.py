from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch
 
class YourCNN(torch.nn.Module):

    def __init__(self):
        super(YourCNN, self).__init__()
        self.cs          = 8 
        
        self.layer_1     = torch.nn.Conv2d(3,self.cs,(5,5), padding = "same") # input size (B, 3, 32, 32)
        self.bn_1        = torch.nn.BatchNorm2d(self.cs)
        
        self.layer_2     = torch.nn.Conv2d(self.cs,self.cs,(5,5), padding = "same") # input size (B, 16, 32, 32)
        self.bn_2        = torch.nn.BatchNorm2d(self.cs)
        
        self.maxp        = torch.nn.MaxPool2d((2,2),stride = (2,2))
        
        self.layer_3     = torch.nn.Conv2d(self.cs,self.cs*2,(3,3), padding = "same") # input size (B, 16, 16, 16)
        self.bn_3        = torch.nn.BatchNorm2d(self.cs*2)
        
        self.layer_4     = torch.nn.Conv2d(self.cs*2,self.cs*2,(3,3), padding = "same") # input size (B, 32, 16, 16)
        self.bn_4        = torch.nn.BatchNorm2d(self.cs*2)
        
        self.maxp2       = torch.nn.MaxPool2d((2,2),stride = (2,2))
        
        self.layer_5 = torch.nn.Conv2d(self.cs*2,self.cs*2,(3,3), padding = "same") # input size (B, 32, 8, 8)
        self.bn_5    = torch.nn.BatchNorm2d(self.cs*2)
        
        self.layer_6     = torch.nn.Linear(self.cs*2*8*8, 128) # input 8192
        self.bn_6        = torch.nn.BatchNorm1d(128)
        self.drop_6      = torch.nn.Dropout(0.25)
        
        self.layer_7     = torch.nn.Linear(128, 10) # input 128   
        
        
    def forward(self, x):
        
        x = self.layer_1(x) # Conv Layer
        x = self.bn_1(x)
        x = torch.nn.functional.relu(x)
        
        x = self.layer_2(x) # Conv Layer
        x = self.bn_2(x)
        x = torch.nn.functional.relu(x)
        
        x = self.maxp(x)
        
        x = self.layer_3(x) # Conv Layer
        x = self.bn_3(x)
        x = torch.nn.functional.relu(x)
        
        x = self.layer_4(x)
        x = self.bn_4(x)
        x = torch.nn.functional.relu(x)
        
        x = self.maxp2(x)
        
        x = self.layer_5(x) # Conv Layer
        x = self.bn_5(x)
        x = torch.nn.functional.relu(x)

        x = torch.flatten(x, 1)
        
        x = self.layer_6(x) # Linear Layer
        x = self.bn_6(x)
        x = self.drop_6(x)
        x = torch.nn.functional.relu(x)
        
        output = self.layer_7(x)
            
        return  output

   