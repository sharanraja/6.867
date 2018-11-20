# Neural network to approximate state vector.
# Loss - MSE Loss
# Use different networks for each input
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

import gym
import random
import numpy as np

env = gym.make("CartPole-v0")

# Neural network to approximate state vectors
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(5, 64)
        #self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 16)
        #self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x



loss_fn = torch.nn.MSELoss(reduction="sum")
η = 1e-3
model_0 = Net()
model_1 = Net()
model_2 = Net()
model_3 = Net()
opt_0 = Adam(model_0.parameters(), lr = η)
opt_1 = Adam(model_1.parameters(), lr = η)
opt_2 = Adam(model_2.parameters(), lr = η)
opt_3 = Adam(model_3.parameters(), lr = η)



# Generate training data
l = []
replay_array = []
for epoch in range(200):

    curr = env.reset()
    for i in range(200):

        # Generate a random step
        st = random.randint(0,1)
        nex, rew, done, info = env.step(st)
        replay_array.append((curr, st, nex, rew, done))
        if done:
            break

        curr = nex




# Train the network
l = []
replay_size = len(replay_array)
for epoch in range(22000):
    i = random.randint(0, replay_size-1)
    curr = replay_array[i][0]
    st = replay_array[i][1]
    nex_0 = replay_array[i][2][0]
    nex_0 = [nex_0]
    nex_0 = np.array(nex_0)
    nex_0 = torch.from_numpy(nex_0).float()
    nex_1 = replay_array[i][2][1]
    nex_1 = [nex_1]
    nex_1 = np.array(nex_1)
    nex_1 = torch.from_numpy(nex_1).float()
    nex_2 = replay_array[i][2][2]
    nex_2 = [nex_2]
    nex_2 = np.array(nex_2)
    nex_2 = torch.from_numpy(nex_2).float()
    nex_3 = replay_array[i][2][3]
    nex_3 = [nex_3]
    nex_3 = np.array(nex_3)
    nex_3 = torch.from_numpy(nex_3).float()
    # Create input for our network and generate prediction
    input = torch.from_numpy(np.append(curr,st)).float()
    nex_pred_0 = model_0(input)
    nex_pred_1 = model_1(input)
    nex_pred_2 = model_2(input)
    nex_pred_3 = model_3(input)
    # Calculate loss
    loss_0 = loss_fn(nex_pred_0, nex_0)
    loss_1 = loss_fn(nex_pred_1, nex_1)
    loss_2 = loss_fn(nex_pred_2, nex_2)
    loss_3 = loss_fn(nex_pred_3, nex_3)
    # Backprop
    opt_0.zero_grad()
    opt_1.zero_grad()
    opt_2.zero_grad()
    opt_3.zero_grad()
    loss_0.backward()
    loss_1.backward()
    loss_2.backward()
    loss_3.backward()
    opt_0.step()
    opt_1.step()
    opt_2.step()
    opt_3.step()
    curr = nex
    nex_pred = [nex_pred_0.item(),nex_pred_1.item(), nex_pred_2.item(), nex_pred_3.item()]
    nex_pred = np.array(nex_pred)
    nex_pred = torch.from_numpy(nex_pred).float()
    nex      = replay_array[i][2]
    nex      = torch.from_numpy(nex).float()
    loss     = loss_fn(nex,nex_pred)
    l.append(loss.item())


plt.plot(l)
torch.save(model_0.state_dict(), "model_0.torch")
torch.save(model_1.state_dict(), "model_1.torch")
torch.save(model_2.state_dict(), "model_2.torch")
torch.save(model_3.state_dict(), "model_3.torch")


# Test the network with random state vectors
# from the environment
t = []
for epoch in range(500):

    curr = env.reset()
    for i in range(200):

        # Generate a random step
        st = random.randint(0,1)

        # Get simulated result from the environment
        nex, rew, done, info = env.step(st)
        nex = torch.from_numpy(nex).float()

        # Check if done and then break
        if done:
            break

        # Create input for our network and generate prediction
        input = torch.from_numpy(np.append(curr,st)).float()
        nex_pred_0 = model_0(input)
        nex_pred_1 = model_1(input)
        nex_pred_2 = model_2(input)
        nex_pred_3 = model_3(input)
        nex_pred_0 = nex_pred_0.item()
        nex_pred_1 = nex_pred_1.item()
        nex_pred_2 = nex_pred_2.item()
        nex_pred_3 = nex_pred_3.item()
        nex_pred = [nex_pred_0,nex_pred_1,nex_pred_2,nex_pred_3]
        nex_pred = np.array(nex_pred)
        nex_pred = torch.from_numpy(nex_pred).float()
        # Calculate loss
        loss = loss_fn(nex_pred, nex)

        if i %100 == 0:
            print("nex = ", nex)
            print("nex_pred = ", nex_pred)
            #print("loss = ", loss.item())


        # Backprop
        # opt.zero_grad()
        # loss.backward()
        # opt.step()

        curr = nex

    t.append(loss.item())

    # epoch % 1000 == 0 and print("Epoch %d done" % epoch)

plt.plot(t)
