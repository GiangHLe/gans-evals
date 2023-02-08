import torch
import timm
from gans_eval.inception import InceptionV3

BLOCK_INDEX_BY_DIM = {
    64: 0,   # First max pooling features
    192: 1,  # Second max pooling featurs
    768: 2,  # Pre-aux classifier features
    2048: 3  # Final average pooling features
}

class Inception(torch.nn.Module):
    def __init__(self, dims=2048) -> None:
        super().__init__()
        index = BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([index], resize_input=True, normalize_input=True)
        temp = self.model.model.fc.bias.clone()
        self.model.model.fc.bias = torch.nn.Parameter(torch.zeros(temp.shape, dtype=temp.dtype))
        
    def forward(self, x):
        prob, features = self.model(x)
        return prob, features

    @staticmethod
    def get_activation(name, activation):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
class VGG(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = timm.create_model('vgg16', pretrained=True)
        self.model.head = torch.nn.Identity()
    
    def forward(self, x):
        return [None, self.model(x)]

class Extractor(torch.nn.Module):
    def __init__(self, model_name, dims=2048):
        super().__init__()
        if model_name=='inception':
            self.model = Inception(dims=dims)
        else:
            self.model = VGG()
        self.model.eval()
        self.name = model_name
        
    @torch.no_grad()
    def forward(self, x):
        prob, features = self.model(x)
        if len(features.shape) != 2:
            features = torch.nn.functional.adaptive_avg_pool2d(features, output_size=(1,1)).squeeze(-1).squeeze(-1)
        return [prob, features]
