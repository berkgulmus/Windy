
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


MODEL_PATH = "Models/"
TEST_VISUALS_PATH = "Test Visuals/"
BUNDLED_TEST_VISUALS_PATH = 'Bundled Test Visuals/'
TEST_DATA_PATH = 'Test Data/'
TEST_FULL_DATA_PATH = 'Test Full Data/'



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        #return F.log_softmax(x, dim=1)
        return x

class Tester():




    def __init__(self,model_name,test_times = 100):


        self.CreateDirectory()
        self.test_times = test_times

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
            print("Running on the GPU")
        else:
            self.device = torch.device("cpu")
            print("Running on the CPU")

        self.model = Net().to(self.device)
        self.model_name = model_name
        self.test_scores = []
        self.model_names = []
        self.gameEnv=GameEnv(sizeX=400,sizeY=600,
                    ballRad=25,boxSize=60,
                    boxSurviveAfterArriving=9,
                    ballElasticity=0.7,mxspd=15)

        random.seed(time.time())
        np.random.seed(int(time.time()))





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

        average_score = str(int(sum(ts)/len(ts)))
        str_time = str(int(time.time()))

        fig.savefig(TEST_VISUALS_PATH+file_name+"_Avg="+average_score +"_TestedAt_"+str_time+".png")
        with open(TEST_DATA_PATH+file_name+"_Avg="+average_score +"_TestedAt_"+str_time, 'wb') as f:
            pickle.dump(ts,f)
        plt.close(fig)
        plt.clf()






    def load_model_names(self):
        a = os.listdir(MODEL_PATH)
        a.sort(reverse=False)
        for mdl in a:
            if self.model_name in mdl:
                self.model_names.append(mdl)

    def load_model(self,mdl_name):
        found_state = torch.load(MODEL_PATH+mdl_name)
        self.model.load_state_dict(found_state['state_dict'])

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

    def Test(self):
        self.load_model_names()
        for model in tqdm.tqdm(self.model_names):
            self.load_model(model)
            str_time = str(int(time.time()))
            self.test_model(times=self.test_times,file_name=model )
        self.show_all_test_results(file_name=self.model_name + "_TestedWithTS="+ str(self.test_times))

Tstr = Tester('ModelC00006',test_times=25)
Tstr.Test()