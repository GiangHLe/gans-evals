import torch
import timm

MODEL_POOL = {'inception': ['inception_v3', 299],
              'vgg': ['vgg16', 224]}

class Extractor(torch.nn.Module):
    def __init__(self, model_name) -> None:
        model = timm.create_model(model_name, pretrained=True)
        model.eval()
        model.head = torch.nn.Identity()
    
    def forward(self, x):
        
        
299  
224 


if __name__=='__main__':
    sada