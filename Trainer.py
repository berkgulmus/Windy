
from timeit import default_timer as timer
import time
import sys
from GameEnvironment import GameEnv
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch.optim as optim
import os
from collections import deque
import numpy as np
import tqdm
import pickle


START_LEARNING_RATE = 0.0002
LEARNING_RATE_DECAY = 0.9997
MINIMUM_LEARNING_RATE = 0.000001

MODEL_PATH = "Models/"
TEST_VISUALS_PATH = "Test Visuals/"
BUNDLED_TEST_VISUALS_PATH = 'Bundled Test Visuals/'
TEST_DATA_PATH = 'Test Data/'
TEST_FULL_DATA_PATH = 'Test Full Data/'



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 64)
        self.fc9 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        #return F.log_softmax(x, dim=1)
        return x

class SoloQN():




    def __init__(self,model_name,test_times = 100):


        self.start_lr = START_LEARNING_RATE
        self.lr_decay = LEARNING_RATE_DECAY
        self.min_lr = MINIMUM_LEARNING_RATE
        self.update_lr_periodically = True
        self.test_times = test_times

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
            print("Running on the GPU")
        else:
            self.device = torch.device("cpu")
            print("Running on the CPU")

        self.model = Net().to(self.device)
        self.model_name = model_name
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.start_lr)
        self.lossfunc = nn.MSELoss()
        self.test_scores = []
        self.load_model()
        self.gameEnv=GameEnv(sizeX=400,sizeY=600,
                    ballRad=25,boxSize=60,
                    boxSurviveAfterArriving=9,
                    ballElasticity=0.7,mxspd=15)
        self.DISCOUNT = 0.9
        self.REPLAY_MEMORY_SIZE = 300_000  # How many last steps to keep for model training
        self.REPLAY_MEMORY = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.MIN_REPLAY_MEMORY_SIZE = 10_000  # Minimum number of steps in a memory to start training
        self.MINIBATCH_SIZE = 1024  # How many steps (samples) to use for training
        self.MODEL_SAVE_EVERY = 200
        self.TEST_EVERY = 200

        # Exploration settings
        self.epsilon = 0.6  # not a constant, going to be decayed
        self.EPSILON_DECAY = 0.9999
        self.MIN_EPSILON = 0.2
        self.MAX_SCORE_BEFORE_RESET = 1500



        random.seed(time.time())
        np.random.seed(int(time.time()))

        self.CreateDirectory()




    def model_action(self,inpt):
        inpt = inpt.to(self.device).view(-1,8)
        with torch.no_grad():
            return torch.argmax(self.model(inpt),dim=1)

    def CreateDirectory(self):
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        if not os.path.exists(TEST_VISUALS_PATH):
            os.makedirs(TEST_VISUALS_PATH)
        if not os.path.exists(BUNDLED_TEST_VISUALS_PATH):
            os.makedirs(BUNDLED_TEST_VISUALS_PATH)
        if not os.path.exists(TEST_DATA_PATH):
            os.makedirs(TEST_DATA_PATH)
        if not os.path.exists(TEST_FULL_DATA_PATH):
            os.makedirs(TEST_FULL_DATA_PATH)

    def test_model(self,times=100,file_name = None):
        self.model.eval()
        test_scores = []
        for i in range(times):
            testgame = GameEnv(sizeX=400,sizeY=600,
                    ballRad=25,boxSize=60,
                    boxSurviveAfterArriving=9,
                    ballElasticity=0.7,mxspd=15)
            cur_state, point, done = testgame.Reset()
            score = 0
            while not done:
                model_prediction = self.model_action(torch.Tensor([cur_state]))
                cur_state, point, done = testgame.Step(model_prediction[0])
                score+=1
            test_scores.append(score)

        self.test_scores.append(test_scores)
        self.save_test_scores(test_scores,file_name)

        return

    def save_test_scores(self,ts,file_name = None):
        fig = plt.figure(figsize=(6, 3), dpi=150)
        plt.plot(ts)
        if file_name is None:
            str_time = str(int(time.time()))
        else:
            str_time = file_name
        average_score = str(int(sum(ts)/len(ts)))
        fig.savefig(TEST_VISUALS_PATH+self.model_name + "_" + str_time+"_"+average_score +".png")
        with open(TEST_DATA_PATH+self.model_name + "_" + str_time+"_"+average_score, 'wb') as f:
            pickle.dump(ts,f)
        plt.close(fig)
        plt.clf()

        self.show_all_test_results(file_name=str_time)


    def save_model(self,file_name = None):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'test_scores': self.test_scores
        }
        if file_name == None:
            str_time = str(int(time.time()))
        else:
            str_time = file_name
        torch.save(state, MODEL_PATH+self.model_name+"_"+str_time)

    def load_model(self):
        a= os.listdir(MODEL_PATH)
        a.sort(reverse=True)
        for mdl in a:
            if self.model_name in mdl:
                found_state  = torch.load(MODEL_PATH+mdl)
                self.model.load_state_dict(found_state['state_dict'])
                self.optimizer.load_state_dict(found_state['optimizer'])
                self.test_scores=found_state['test_scores']
                print(found_state['test_scores'])
                print(self.test_scores)
                print("model found and read at "+mdl)
                return
        print("no model found with this name")


    def play_train(self,ep_count = 1000):

        for i in tqdm.tqdm(range(ep_count)):
            cur_state, reward, done = self.gameEnv.Reset()
            episode_reward = 0
            step = 1
            while not done:
                self.model.eval()
                if np.random.random() > self.epsilon:
                    act_tbt = self.model_action(torch.Tensor(cur_state))
                else:
                    act_tbt = np.random.random() > 0.5
                new_state, reward, done = self.gameEnv.Step(act_tbt)
                step += 1
                self.REPLAY_MEMORY.append([cur_state, act_tbt, reward, new_state, done])
                cur_state = new_state
                self.train_from_mem()
                if step == self.MAX_SCORE_BEFORE_RESET:
                    done = True
            if i% self.TEST_EVERY == self.TEST_EVERY-1:
                self.test_model()
            if i% self.MODEL_SAVE_EVERY == self.MODEL_SAVE_EVERY-1:
                self.save_model()
            if (self.epsilon > self.MIN_EPSILON):
                self.epsilon *= self.EPSILON_DECAY

    def train(self,hmt = 1000,bs = 64):

        #hmt = how many times
        #bs = batch size

        overhead = 0

        while len(self.REPLAY_MEMORY) < self.MIN_REPLAY_MEMORY_SIZE:
            self.play()

        for i in tqdm.tqdm(range(hmt)):
            #training
            self.train_from_mem(bs)

            #creating memory if we used up more than we created
            overhead -= bs
            while overhead < 0:
                overhead += self.play()

            #updating learning rate
            self.update_lr(self.optimizer)

            #testing & saving
            cur_time = str(int(time.time()))
            if i% self.MODEL_SAVE_EVERY == self.MODEL_SAVE_EVERY-1:
                self.save_model(file_name=cur_time)
            if i% self.TEST_EVERY == self.TEST_EVERY-1:
                self.test_model(file_name=cur_time,times=self.test_times)




    def play(self):
        cur_state, reward, done = self.gameEnv.Reset()
        step = 1
        self.model.eval()
        while not done:
            random_movement = False
            if np.random.random() > self.epsilon:
                act_tbt = self.model_action(torch.Tensor(cur_state))
            else:
                random_movement = True
                act_tbt = np.random.random() > 0.5

            new_state, reward, done = self.gameEnv.Step(act_tbt)
            step += 1
            self.REPLAY_MEMORY.append([cur_state, act_tbt, reward, new_state, done])
            cur_state = new_state
            if (self.epsilon > self.MIN_EPSILON):
                self.epsilon *= self.EPSILON_DECAY
        return step

    def show_all_test_results_v0(self,file_name= None): #obsolete
        a = os.listdir("test scores/")
        a.sort()
        fig = plt.figure(figsize=(6, 3), dpi=800)
        ctr = 0
        for f_name in a:
            if self.model_name in f_name:
                with open("test scores/"+f_name, 'rb') as f:
                    test_score = pickle.load(f)
                    plt.scatter(x=[ctr ] * len(test_score),y=test_score,marker = '4',linewidths=0.4,s=15)
                    ctr+=1

        if file_name is None:
            str_time = str(int(time.time()))
        else:
            str_time = file_name
        fig.savefig("bundled test visuals/" + self.model_name + "_" + str_time + ".png")
        plt.close(fig)
        #plt.show()

    def show_all_test_results(self,file_name= None):

        fig = plt.figure(figsize=(6, 3), dpi=800)

        lineplotx=[]
        lineploty=[]

        for order,i in enumerate(self.test_scores):
            plt.scatter(x=[order] * len(i), y=i, marker='4', linewidths=0.4, s=15)
            lineploty.append(sum(i)/len(i))
            lineplotx.append(order)

        plt.plot(lineplotx,lineploty,linewidth=1,color='silver')

        if file_name is None:
            str_time = str(int(time.time()))
        else:
            str_time = file_name

        plt.ylim(bottom=0)

        fig.savefig(BUNDLED_TEST_VISUALS_PATH + self.model_name + "_" + str_time + ".png")
        plt.close(fig)
        plt.clf()


    def train_from_mem(self,batch_size = None):
        if batch_size == None:
            batch_size= self.MINIBATCH_SIZE

        if len(self.REPLAY_MEMORY) < self.MIN_REPLAY_MEMORY_SIZE:
            return
        self.model.train()
        minibatch = random.sample(self.REPLAY_MEMORY, batch_size)

        current_states = torch.Tensor([transition[0] for transition in minibatch])
        current_states = current_states.to(self.device)
        current_qs_list = self.model(current_states.view(-1, 8))

        new_current_states = torch.Tensor([transition[3] for transition in minibatch])
        new_current_states = new_current_states.to(self.device)
        future_qs_list = self.model(new_current_states.view(-1, 8))

        X=[]
        y=[]

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = torch.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index].tolist()
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)
        #print(X)
        #print(y)
        X = torch.Tensor(X).to(self.device)
        y = torch.Tensor(y).to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(X)
        loss = self.lossfunc(outputs, y)
        loss.backward()
        self.optimizer.step()
        return



    def update_lr(self,optim):
        if(self.update_lr_periodically):
            for g in optim.param_groups:
                if (g['lr'] > self.min_lr):
                    g['lr'] = g['lr'] * self.lr_decay
                if (g['lr'] < self.min_lr):
                    g['lr'] = self.min_lr

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            if (g['lr'] > self.min_lr):
                g['lr'] = lr

myqn = SoloQN("ModelK00005")
myqn.update_lr_periodically = False
myqn.set_lr(0.000000001)
myqn.train(15000)   