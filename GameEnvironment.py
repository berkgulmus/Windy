import math, random


class Ball:

    def __init__(self, rad):
        self.radius = rad
        self.x = 0
        self.y = 0
        self.speed = 0

    def set_pos(self, px, py):
        self.x = px
        self.y = py

    def set_speed(self, spd):
        self.speed = spd


class Box:

    def __init__(self, sz, saa):
        self.size = sz
        self.remainingLife = saa  # box will survive x turns after arriving to the ground
        self.arrived = False
        self.x = 0
        self.y = 0
        self.speed = 0
        self.col = (random.uniform(0,255),random.uniform(0,255),random.uniform(0,255))

    def set_pos(self, px, py):
        self.x = px
        self.y = py

    def set_speed(self, spd):
        self.speed = spd


class GameEnv:

    def __init__(self, sizeX, sizeY, ballRad, boxSize, boxSurviveAfterArriving, ballElasticity, mxspd):

        self.ball = Ball(ballRad)
        self.ball.set_pos(sizeX / 2, sizeY - ballRad)
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.ballRad = ballRad
        self.boxSize = boxSize
        self.boxSurviveAfterArriving = boxSurviveAfterArriving
        self.ballElasticity = ballElasticity
        self.maxSpeed = mxspd
        self.gameOver = False
        # until this threshold, there's no need to check the box-ball collision because the box is too far up
        self.box_check_threshold = self.sizeY-self.ballRad-self.boxSize-1
        # at this step, follower box will be created for the first time
        self.createBoxF = 23
        self.score=0
        self.boxR = None
        self.boxF = None
        self.DropRandomBox()


        # these are gonna be used for scaling ball position
        self.ballXmin = self.ballRad
        self.ballXmax = self.sizeX - self.ballRad
        self.ballXrange= self.ballXmax - self.ballXmin

        # these are gonna be used for scaling box position
        self.boxYmin = -self.boxSize
        self.boxYmax = self.sizeY - self.boxSize
        self.boxYrange = self.boxYmax - self.boxYmin

        self.boxXmin = 5
        self.boxXmax = self.sizeX - self.boxSize - 5
        self.boxXrange = self.boxXmax - self.boxXmin



    def Step(self, action):

        # until this threshold, there's no need to check the box-ball collision because the box is too far up
        col_found = False
        if self.boxR.y > self.box_check_threshold :
            col_found = col_found or self.CollisionCheckR()
        if self.boxF!= None:
            if self.boxF.y > self.box_check_threshold:
                col_found = col_found or self.CollisionCheckF()
        self.UpdateSpeeds(action)
        self.BoxRecreation()
        self.Move()

        if col_found:
            self.gameOver=True
        else:
            self.score += 1
            if self.score == self.createBoxF:
                self.DropFollowerBox()

        point = 2
        if self.gameOver:
            point = -5

        return (self.observation(),point,self.gameOver)

    def CollisionCheckR(self):
        return self.HalfColCheckR() or self.FullColCheckR()
    def CollisionCheckF(self):
        return self.HalfColCheckF() or self.FullColCheckF()

    def Reset(self):

        self.ball.set_pos(self.sizeX / 2, self.sizeY - self.ballRad)
        self.ball.speed=0
        self.score = 0
        self.boxR = None
        self.boxF = None
        self.DropRandomBox()
        self.gameOver = False

        point = 0
        return (self.observation(), point, self.gameOver)

    def Move(self):

        self.ball.x += self.ball.speed
        if (self.ball.x < self.ballRad):
            self.ball.speed = math.floor(self.ball.speed * -self.ballElasticity)  # ball gets pushed back from the wall
            self.ball.x = self.ballRad

        elif (self.ball.x > (self.sizeX - self.ballRad)):
            self.ball.speed = -1 * math.floor(
                self.ball.speed * self.ballElasticity)  # ball gets pushed back from the wall
            self.ball.x = self.sizeX - self.ballRad


        self.boxR.y += self.boxR.speed
        if(self.boxR.y> self.sizeY-self.boxSize):
            self.boxR.y = self.sizeY-self.boxSize
            self.boxR.arrived=True
            self.boxR.speed = 0

        if( self.boxF != None):
            self.boxF.y += self.boxF.speed
            if (self.boxF.y > self.sizeY - self.boxSize):
                self.boxF.y = self.sizeY - self.boxSize
                self.boxF.arrived = True
                self.boxF.speed = 0

        return

    def UpdateSpeeds(self, action):
        if (action):
            self.ball.speed = min(self.maxSpeed, self.ball.speed + 1)
        else:
            self.ball.speed = max(-self.maxSpeed, self.ball.speed - 1)

        if(not self.boxR.arrived):
            self.boxR.speed = min(25, self.boxR.speed + 1)

        if (self.boxF != None):
            if (not self.boxF.arrived):
                self.boxF.speed = min(25, self.boxF.speed + 1)


        return

    def DropRandomBox(self):
        # box' pivot point is top left corner
        # this box spawns randomly
        self.boxR = Box(self.boxSize, self.boxSurviveAfterArriving)
        self.boxR.set_pos(random.randint(5, self.sizeX - self.boxSize - 5), -self.boxSize)

    def DropFollowerBox(self):
        # box' pivot point is top left corner
        # this box spawns at the top of the ball
        self.boxF = Box(self.boxSize, self.boxSurviveAfterArriving)
        self.boxF.col = (0,0,0)
        self.boxF.set_pos(self.MinMax(5,self.ball.x-(self.boxSize/2),self.sizeX - self.boxSize - 5), -self.boxSize)

    def MinMax(self, mn, inpt, mx):
        return min(max(mn, inpt), mx)

    def HalfColCheckR(self):

        halfPosBall = self.ball.x + (self.ball.speed/2)
        halfPosBoxR  = self.boxR.y + (self.boxR.speed/2)

        return self.CheckBoxBallCollision(halfPosBall,self.ball.y,self.boxR.x,halfPosBoxR)

    def HalfColCheckF(self):

        halfPosBall = self.ball.x + (self.ball.speed / 2)
        BoxFCheck = False

        if self.boxF != None:
            halfPosBoxF = self.boxF.y + (self.boxF.speed / 2)
            BoxFCheck = self.CheckBoxBallCollision(halfPosBall, self.ball.y, self.boxF.x, halfPosBoxF)

        return BoxFCheck

    def FullColCheckR(self):

        PosBall = self.ball.x + self.ball.speed
        PosBoxR = self.boxR.y + self.boxR.speed


        BoxFCheck = False
        if self.boxF != None:
            PosBoxF = self.boxF.y + self.boxF.speed
            BoxFCheck = self.CheckBoxBallCollision(PosBall, self.ball.y, self.boxF.x, PosBoxF)

        return self.CheckBoxBallCollision(PosBall, self.ball.y, self.boxR.x, PosBoxR) or BoxFCheck

    def FullColCheckF(self):

        PosBall = self.ball.x + self.ball.speed
        BoxFCheck = False
        if self.boxF != None:
            PosBoxF = self.boxF.y + self.boxF.speed
            BoxFCheck = self.CheckBoxBallCollision(PosBall, self.ball.y, self.boxF.x, PosBoxF)

        return BoxFCheck


    def BoxRecreation(self):
        if(self.boxR.arrived):
            self.boxR.remainingLife-=1
            if(self.boxR.remainingLife==0):
                #print(f"box destroyed at {self.score}. step")
                self.DropRandomBox()

        if (self.boxF != None):
            if (self.boxF.arrived):
                self.boxF.remainingLife -= 1
                if (self.boxF.remainingLife == 0):
                    #print(f"box destroyed at {self.score}. step")
                    self.DropFollowerBox()

    def CheckBoxBallCollision(self,ballposx,ballposy,boxposx,boxposy):
        closestX = self.MinMax(ballposx, boxposx, boxposx + self.boxSize)
        closestY = self.MinMax(ballposy, boxposy, boxposy + self.boxSize)

        distanceX = ballposx - closestX
        distanceY = ballposy - closestY

        distanceSquared = (distanceX * distanceX) + (distanceY * distanceY)

        return distanceSquared < (self.ballRad * self.ballRad)


    def observation_v0(self): #obsolete
        BallPosXScaled = (self.ball.x - self.ballXmin) / (self.ballXrange)
        BallSpeedScaled = self.ball.speed / self.maxSpeed

        BoxRPosXScaled = (self.boxR.x - self.boxXmin) / self.boxXrange
        BoxRPosYScaled = (self.boxR.y - self.boxYmin) / self.boxYrange
        BoxRSpeedScaled = self.boxR.speed / self.maxSpeed

        if self.boxF != None:
            BoxFPosXScaled = (self.boxF.x - self.boxXmin) / self.boxXrange
            BoxFPosYScaled = (self.boxF.y - self.boxYmin) / self.boxYrange
            BoxFSpeedScaled = self.boxF.speed / self.maxSpeed

        else:
            BoxFPosXScaled = BoxRPosXScaled
            BoxFPosYScaled = BoxRPosYScaled
            BoxFSpeedScaled = BoxRSpeedScaled

        return (BallPosXScaled,BallSpeedScaled,BoxRPosXScaled,BoxRPosYScaled,BoxRSpeedScaled,BoxFPosXScaled,BoxFPosYScaled,BoxFSpeedScaled)

    def observation_v1(self): #obsolete
        BallPosXScaled = (self.ball.x - self.ballXmin) / (self.ballXrange)
        BallSpeedScaled = self.ball.speed / self.maxSpeed

        BoxRPosXScaled = (self.boxR.x - self.boxXmin) / self.boxXrange
        BoxRPosYScaled = (self.boxR.y - self.boxYmin) / self.boxYrange
        BoxRRemLifeScaled = self.boxR.remainingLife / self.boxSurviveAfterArriving

        if self.boxF != None:
            BoxFPosXScaled = (self.boxF.x - self.boxXmin) / self.boxXrange
            BoxFPosYScaled = (self.boxF.y - self.boxYmin) / self.boxYrange
            BoxFRemLifeScaled = self.boxF.remainingLife / self.boxSurviveAfterArriving

        else:
            BoxFPosXScaled = BoxRPosXScaled
            BoxFPosYScaled = BoxRPosYScaled
            BoxFRemLifeScaled = BoxRRemLifeScaled

        return [BallPosXScaled, BallSpeedScaled, BoxRPosXScaled, BoxRPosYScaled, BoxRRemLifeScaled, BoxFPosXScaled,
                BoxFPosYScaled, BoxFRemLifeScaled]

    def observation(self):
        BallPosXScaled = (self.ball.x - self.ballXmin) / (self.ballXrange)
        BallSpeedScaled = self.ball.speed / self.maxSpeed

        BoxRPosXScaled = (self.boxR.x - self.boxXmin) / self.boxXrange
        BoxRPosYScaled = (self.boxR.y - self.boxYmin) / self.boxYrange
        BoxRRemLifeScaled = self.boxR.remainingLife / self.boxSurviveAfterArriving

        if self.boxF != None:
            BoxFPosXScaled = (self.boxF.x - self.boxXmin) / self.boxXrange
            BoxFPosYScaled = (self.boxF.y - self.boxYmin) / self.boxYrange
            BoxFRemLifeScaled = self.boxF.remainingLife / self.boxSurviveAfterArriving

        else:
            BoxFPosXScaled = BoxRPosXScaled
            BoxFPosYScaled = BoxRPosYScaled
            BoxFRemLifeScaled = BoxRRemLifeScaled

        #this will ensure that whichever box is clossest to the ground will be in the first place
        if BoxRPosYScaled > BoxFPosYScaled:
            temp0 = BoxRPosXScaled
            temp1 = BoxRPosYScaled
            temp2 = BoxRRemLifeScaled

            BoxRPosXScaled = BoxFPosXScaled
            BoxRPosYScaled = BoxFPosXScaled
            BoxRRemLifeScaled = BoxFRemLifeScaled

            BoxFPosXScaled = temp0
            BoxFPosYScaled = temp1
            BoxFRemLifeScaled = temp2



        return [BallPosXScaled, BallSpeedScaled, BoxRPosXScaled, BoxRPosYScaled, BoxRRemLifeScaled, BoxFPosXScaled,
                BoxFPosYScaled, BoxFRemLifeScaled]