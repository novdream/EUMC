import torch
import torch.nn as nn
class Homo(nn.Module):
    def __init__(self, feature_dim,hidden_dim,device,dropout=0.5):
        super(Homo, self).__init__()
        # self.layers = nn.Linear(feature_dim,feature_dim).to(device)
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        ).to(device) 
        self.device = device
        
    def forward(self, x):
        x = x.to(self.device)
        return self.layers(x)
    
    