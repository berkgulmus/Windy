import pygame
from timeit import default_timer as timer
import time
from GameEnvironment import GameEnv

class WrapperEnv:
    def __init__(self, SPS, WT, GH, BackC, BallC, BoxC, GroundC, WallC, sizeX, sizeY, ballRad, boxSize,
                 boxSurviveAfterArriving, ballElasticity, mxspd,restartAfterDeath = True):
        self.stepPerSecond = SPS
        self.wallThickness = WT
        self.groundHeight = GH
        self.backgroundColour = BackC
        self.ballColour = BallC
        self.boxColour = BoxC
        self.groundColour = GroundC
        self.wallColour = WallC
        self.restartAfterDeath = restartAfterDeath
        pygame.init()
        self.screen = pygame.display.set_mode([sizeX + 2 * WT, sizeY + GH])
        self.game = GameEnv(sizeX, sizeY, ballRad, boxSize, boxSurviveAfterArriving, ballElasticity, mxspd)
        self.running = True

    def Start(self):
        last_time = timer()
        delta_time = 0
        cur_input = False
        self.screen.fill(self.backgroundColour)
        while self.running:

            cur_time = timer()
            elapsed_time = cur_time - last_time
            last_time = cur_time
            delta_time += elapsed_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            keys = pygame.key.get_pressed()

            if keys[pygame.K_SPACE]:
                cur_input = True
            if (delta_time >= 1 / self.stepPerSecond):
                self.game.Step(cur_input)
                cur_input = False
                delta_time -= 1 / self.stepPerSecond

                self.Draw()
                if (self.game.gameOver):
                    if self.restartAfterDeath:
                        self.game.Reset()
                    else:
                        time.sleep(1)
                        self.running=False

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



