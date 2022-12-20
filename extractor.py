import torch
import timm

class Extractor(torch.nn.Module):
    def __init__(self, model_name):
        model = timm.create_model(model_name, pretrained=True)
        model.eval()
        model.head = torch.nn.Identity()
    
    def forward(self, x):
        return None