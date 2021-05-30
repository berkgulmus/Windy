import pygame
from timeit import default_timer as timer
import time
import sys
from GameEnvironment import GameEnv
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

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

class ModelVisualizer:
    def __init__(self, SPS, WT, GH, BackC, BallC, BoxC, GroundC, WallC, sizeX, sizeY, ballRad, boxSize,
                 boxSurviveAfterArriving, ballElasticity, mxspd,model_name):
        self.stepPerSecond = SPS
        self.wallThickness = WT
        self.groundHeight = GH
        self.backgroundColour = BackC
        self.ballColour = BallC
        self.boxColour = BoxC
        self.groundColour = GroundC
        self.wallColour = WallC

        pygame.init()
        self.screen = pygame.display.set_mode([sizeX + 2 * WT, sizeY + GH])
        self.game = GameEnv(sizeX, sizeY, ballRad, boxSize, boxSurviveAfterArriving, ballElasticity, mxspd)
        self.running = True
        self.ai = Net()
        self.ai.load_state_dict(torch.load(model_name)['state_dict'])
        self.ai.eval()
        self.scores = []
    def Start(self):
        last_time = timer()
        delta_time = 0
        cur_input = False
        self.screen.fill(self.backgroundColour)
        cur_state, reward, done = self.game.Reset()
        score = 0
        while self.running:

            cur_time = timer()
            elapsed_time = cur_time - last_time
            last_time = cur_time
            delta_time += elapsed_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            if (delta_time >= 1 / self.stepPerSecond):
                #---------
                with torch.no_grad():
                    z = self.ai(torch.Tensor(cur_state).view(-1,8))
                    print(z)
                    ai_output = torch.argmax(z,dim=1)
                    #print(ai_output)

                    #print(torch.argmax(ai_output[0]))
                #----------

                cur_state, p, go = self.game.Step(ai_output[0])
                cur_input = False
                delta_time -= 1 / self.stepPerSecond
                score +=1
                self.Draw()
                if (self.game.gameOver):
                    self.game.Reset()
                    self.scores.append(score)
                    score = 0

        pygame.quit()

    def Draw(self):
        self.screen.fill(self.backgroundColour)
        ball = pygame.draw.circle(self.screen,
                                  pygame.Color(self.ballColour),
                                  (self.game.ball.x+self.wallThickness, self.game.ball.y),
                                  self.game.ball.radius,
                                  0)


        boxR = pygame.draw.rect(self.screen,
                                     pygame.Color(self.game.boxR.col),
                                     (self.game.boxR.x+self.wallThickness, self.game.boxR.y,
                                      self.game.boxSize, self.game.boxSize),
                                     0)
        if self.game.boxF!=None:
            boxF = pygame.draw.rect(self.screen,
                                    pygame.Color(self.game.boxF.col),
                                    (self.game.boxF.x + self.wallThickness, self.game.boxF.y,
                                     self.game.boxSize, self.game.boxSize),
                                    0)
        left_wall = pygame.draw.rect(self.screen,
                                     pygame.Color(self.wallColour),
                                     (0, 0, self.wallThickness, self.game.sizeY ),
                                     0)

        right_wall = pygame.draw.rect(self.screen,
                                     pygame.Color(self.wallColour),
                                     (self.game.sizeX+self.wallThickness, 0, self.wallThickness, self.game.sizeY ),
                                     0)

        ground = pygame.draw.rect(self.screen,
                                  pygame.Color(self.groundColour),
                                  (0, self.game.sizeY , self.game.sizeX+2*self.wallThickness,self.groundHeight),
                                  0)

        pygame.display.update()



my_env = ModelVisualizer(SPS=30,WT=15,GH=20,
                    BackC="#faf3e0",
                    BallC="#b7657b",
                    BoxC="#cdc733",
                    GroundC="#b68973",
                    WallC="#eabf9f",
                    sizeX=400,sizeY=600,
                    ballRad=25,boxSize=60,
                    boxSurviveAfterArriving=9,
                    ballElasticity=0.7,mxspd=15,model_name='models/Elite6_ModelC00006_1620049652')

my_env.Start()

import matplotlib.pyplot as plt
plt.plot(my_env.scores)
plt.show()
