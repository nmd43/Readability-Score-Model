import torch 
import torch.nn as nn 
from device import select_device 
from testtrain import train, test
from mlp import MLP
from dataset import TrainSet, TestSet, dataloader
import matplotlib.pyplot as plt
import numpy as np

device = select_device()

trainSet, testSet = TrainSet(), TestSet() 

train_dataloader, test_dataloader = dataloader(trainSet, testSet, batch_size=1024)
    

model = MLP(
    data_shape=(300,), 
    hidden_size=300, 
    scale_factor=1, 
    num_layers=3, 
    activation="leaky relu", 
    target_size=1
).to(device)


loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(
    model.parameters(),     # which parameters to optimize
    lr=1e-7,                 # learning rate 
)


epochs = 500
train_losses = []
test_losses = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_losses.append(np.mean(train(train_dataloader, model, loss_fn, optimizer, device)))
    test_losses.append(test(test_dataloader, model, loss_fn, device))
    
plt.plot(train_losses, c="b")
plt.xlabel("Epoch")
plt.ylabel("Loss on Training Set")
plt.show() 
plt.close()

plt.plot(test_losses, c="r")
plt.xlabel("Epoch")
plt.ylabel("Loss of Testing Set")
plt.show() 
plt.close()