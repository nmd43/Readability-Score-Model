import os
import torch, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from embedders import doc2vec 

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class TrainSet(Dataset): 
    
    def __init__(self, transform=None): 
        
        self.embedding = doc2vec()[10000:,:].astype("float32")
        self.transform = transform 
        
        with open(f"src/DeepPipe/labels", "rb") as fp: 
            labels = pickle.load(fp) 
            
        self.labels = [elem[0] for elem in labels][10000:]
        
    
    def __len__(self): 
        return self.embedding.shape[0]
    
    def __getitem__(self, idx): 

        return (self.embedding[idx], self.labels[idx])
    
class TestSet(Dataset): 
    
    def __init__(self, transform=None): 
        
        self.embedding = doc2vec()[:10000,:].astype("float32")
        self.transform = transform 
        
        with open(f"src/DeepPipe/labels", "rb") as fp: 
            labels = pickle.load(fp) 
            
        self.labels = [elem[0] for elem in labels][:10000]
        
    
    def __len__(self): 
        return self.embedding.shape[0]
    
    def __getitem__(self, idx): 
            
        return (self.embedding[idx], self.labels[idx])

        

def dataloader(training_data, test_data, batch_size): 
    
    train_dataloader = DataLoader(training_data,    # our dataset
                                    batch_size=batch_size,    # batch size
                                    shuffle=True      # shuffling the data
                                    )
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader