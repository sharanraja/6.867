# Solution of Open AI gym environment "Cartpole-v0" (https://gym.openai.com/envs/CartPole-v0) using DQN and Pytorch.
# It is is slightly modified version of Pytorch DQN tutorial from
# http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html.
# The main difference is that it does not take rendered screen as input but it simply uses observation values from the \
# environment.

import gym
from gym import wrappers
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam


# hyper parameters
EPISODES = 1000  # number of episodes
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
GAMMA = 0.8  # Q-learning discount factor
LR = 0.0001  # NN optimizer learning rate
HIDDEN_LAYER_1 = 128  # NN hidden layer size
Hidden_layer_2 = 128
BATCH_SIZE = 64  # Q-learning batch size
TARGET_UPDATE = 10

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, HIDDEN_LAYER_1)
        self.l2 = nn.Linear(HIDDEN_LAYER_1, Hidden_layer_2)
        self.l3 = nn.Linear(Hidden_layer_2,2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

# Establish a baseline generative model. Exploits the structure/physics. 
class Net_1(nn.Module):
    
    def __init__(self):
        super(Net_1, self).__init__()
        self.fc1 = nn.Linear(2,1)
    
    def forward(self, x):
        x = self.fc1(x)
        return x
        
class Net_2(nn.Module):
    
    def __init__(self):
        super(Net_2, self).__init__()
        self.fc1 = nn.Linear(3,1)
    
    def forward(self, x):
        x = self.fc1(x)
        return x

class Net_3(nn.Module):
    
    def __init__(self):
        super(Net_3, self).__init__()
        self.fc1 = nn.Linear(4,1)
    
    def forward(self, x):
        x = self.fc1(x)
        return x


env = gym.make('CartPole-v0')

# Setting random seed
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
env.seed(0)

model = Network().to(device)
target = Network().to(device)
target.load_state_dict(model.state_dict())
target.eval()
memory = ReplayMemory(10000)
optimizer = optim.Adam(model.parameters(), LR)
steps_done = 0
# episode_durations = []
model_0 = Net_1()
model_1 = Net_2()
model_2 = Net_1()
model_3 = Net_3()

class SampleCounter: 
    def __init__(self):
        self.counter = 0
    def increment(self): 
        self.counter += 1
    def count(self):
        return self.counter


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return model(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)

def run_episode(e, environment, episode_durations, sample_counter, switch, test, debug):
    state = environment.reset()
    steps = 0
    meanie = 0.
    while True:
        # environment.render()
        action = select_action(FloatTensor([state]))

        st = action.item()
        if not switch:
            next_state, reward, done, _ = environment.step(st)
            samples.append((state, st, next_state, reward, done))
        else:
            _next , reward, done, _ = environment.step(st)
            ip = torch.from_numpy(np.append(state, st)).float()

            input = torch.from_numpy(np.append(ip[0],ip[1])).float()
            nex_pred_0 = model_0(input)

            input = torch.from_numpy(np.array([ip[1],ip[2],st])).float()
            nex_pred_1 = model_1(input)

            input = torch.from_numpy(np.append(ip[2],ip[3])).float()
            nex_pred_2 = model_2(input)

            input = torch.from_numpy(np.array([ip[1],ip[2],ip[3],st])).float()
            nex_pred_3 = model_3(input)

            next_state = np.array([nex_pred_0.item(),nex_pred_1.item(), nex_pred_2.item(), nex_pred_3.item()])


        if debug: 
            print("state = ", next_state)
            print("_next = ", _next)

        # negative reward when attempt ends
        if done:
            reward = -1

        memory.push((FloatTensor([state]),
                     action,  # action is already a tensor
                     FloatTensor([next_state]),
                     FloatTensor([reward])))

        if not test: 
            learn()
        else: 
            pass

        if not switch: 
            state = next_state
        else:
            state = next_state
            env.env.state = next_state

        steps += 1
        sample_counter.increment()

        if done:
            print("{2} Episode {0} finished after {1} steps"
                  .format(e, steps, '\033[92m' if steps >= 195 else '\033[99m'))
            episode_durations.append(steps)
            meanie = plot_durations(episode_durations)
            break
    
    return meanie


def learn():
    if len(memory) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))

    # current Q values are estimated by NN for all actions
    current_q_values = model(batch_state).gather(1, batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = target(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    loss = F.mse_loss(current_q_values.squeeze(1), expected_q_values)

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(),10)
    optimizer.step()


def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    meanie = 0.
    # take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        print("mean over last 100 episodes = ", means[-1])
        meanie = (means[-1]).item()
        plt.plot(means.numpy())
    plt.savefig('reward_vs_episode_DDQN.png')

    plt.pause(0.001)  # pause a bit so that plots are updated

    return meanie


x = SampleCounter()
samples = []

# Trains for a number of episodes
def train(episode_count, switch = False, test = False, reset_episode_durations = False):
    episode_durations = []

    for e in range(episode_count):
        if len(episode_durations) > 0 and episode_durations[-1] == 200:
            debug = False
        else:
            debug = False
        meanie = run_episode(e, env, episode_durations, x, switch, test, debug)
        if meanie >= 195 and not test:
            print("CONVERGED!!!!")
            break
        if e % TARGET_UPDATE == 0:
            target.load_state_dict(model.state_dict())

def train_nn():

    loss_fn = torch.nn.MSELoss(reduction="sum")
    η = 1e-2

    opt_0 = Adam(model_0.parameters(), lr = η)
    opt_1 = Adam(model_1.parameters(), lr = η)
    opt_2 = Adam(model_2.parameters(), lr = η)
    opt_3 = Adam(model_3.parameters(), lr = η)

    l = []
    replay_array = []
    for epoch in range(5): 
        
        curr = env.reset()
        for i in range(200):

            # Generate a random step 
            st = random.randint(0,1)
            nex, rew, done, info = env.step(st)
            replay_array.append((curr, st, nex, rew, done, i))
            if done:
                break        
            curr = nex


    l = []
    replay_size = len(replay_array)
    for epoch in range(70000): 
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
        input = torch.from_numpy(np.append(curr[0],curr[1])).float()
        nex_pred_0 = model_0(input)
        inter = [curr[1],curr[2],st]
        inter = np.array(inter)
        input = torch.from_numpy(inter).float()
        nex_pred_1 = model_1(input)
        input = torch.from_numpy(np.append(curr[2],curr[3])).float()
        nex_pred_2 = model_2(input)
        inter = [curr[1],curr[2],curr[3],st]
        inter = np.array(inter)
        input = torch.from_numpy(inter).float()
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
        
        nex_pred = [nex_pred_0.item(),nex_pred_1.item(), nex_pred_2.item(), nex_pred_3.item()]
        nex_pred = np.array(nex_pred)
        nex_pred = torch.from_numpy(nex_pred).float()
        nex      = replay_array[i][2]
        nex      = torch.from_numpy(nex).float()
        loss     = loss_fn(nex,nex_pred)
        l.append(loss.item())

    plt.plot(l[1000:-1])
    plt.xlabel('epochs')
    plt.ylabel('training_error')

train(50)
samples_size = len(samples)
print("Training NN now...")
train_nn()
print("Done Training NN.")

# Train DQN with NN doing inference

# model = Network().to(device)
# target = Network().to(device)
# target.load_state_dict(model.state_dict())
# target.eval()
# optimizer = optim.Adam(model.parameters(), LR)

# Train with our neural network
train(1000, switch = True, test = False, reset_episode_durations = True)

# Now test
train(101, switch = False, test = True, reset_episode_durations = True)

print('Complete')

env.close()
plt.ioff()
plt.show()
