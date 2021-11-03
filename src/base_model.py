import numpy as np
import torch

class BaseModel:
    def __init__(self):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
        
    def inference_loop(self):
        pass
    
    def training_loop(self):
        pass
    
    def tensorboard_logging(self):
        pass
    
    def load(self):
        pass
    
    def save(self):
        pass
    
    def epoch_reset(self):
        pass