import torch
import time

if torch.backends.mps.is_available():
    mps_device=torch.device("mps")
    x=torch.ones(1,device=mps_device)
    print(x)
else:
    print("Not found")