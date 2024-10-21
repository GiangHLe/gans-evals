import torch
import pickle

def load_pickle(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def test_float16(device: str):
    var = torch.randn(15, 30).half().to(device)
    float16_available = True
    try:
        _ = torch.cdist(var, var)
    except:
        float16_available=False
    return float16_available