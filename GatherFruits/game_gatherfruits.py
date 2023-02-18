import sys
sys.path.append("..\\")
from delega import GameCore
import pygame, sys
import random
import numpy as np

class Fruit:
    def __init__(self, x, y, v):
        self.x = x
        self.y = y
        self.v = v
    
    def update(self, dt):
        self.y += self.v * dt


class Game(GameCore):
    def __init__(self, with_graphics = True):
        # 0 -> nothing, 1 -> Left, 2->Right
        super().__init__(3, 1, )

        self.with_graphics = with_graphics
        self.width = 200
        self.height = 200

        self.playerSpeed = 0.0
        self.playerAcceleration = 250.0
        self.playerMaxSpeed = 200
        self.maxNumFruits = 1

        self.playerWidth = 30
        self.playerHeight = 6
        self.playerWidthHalf = self.playerWidth/2
        self.playerHeightHalf = self.playerHeight/2
        self.playerYpos = self.height - self.playerHeightHalf - 10

        if (with_graphics):
            self.surface = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Fruits Gathering")
            self.textFont = pygame.font.SysFont('Calibri', 30)
            self.scoreSurface = self.textFont.render("0", False, (255, 255,255))

        self.reset()

    def spawnFruit(self):
        self.ripefruits.append(Fruit(random.randint(0, self.width), 0, 150))

    def update(self, dt):
        for fruit in self.ripefruits:
            fruit.update(dt)

        if (self.playerActions[0] == 1):
            self.playerSpeed -= self.playerAcceleration * dt
        elif (self.playerActions[0] == 2):
            self.playerSpeed += self.playerAcceleration * dt
        else:
            self.playerSpeed *= 0.99

        self.playerXpos += self.playerSpeed * dt
        if (self.playerXpos < self.playerWidthHalf) : 
            self.playerXpos = self.playerWidthHalf
            self.playerSpeed *= -1
        if (self.playerXpos > self.width - self.playerWidthHalf) : 
            self.playerXpos = self.width - self.playerWidthHalf
            self.playerSpeed *= -1

        if (self.playerSpeed > self.playerMaxSpeed) : self.playerSpeed = self.playerMaxSpeed
        if (self.playerSpeed < -self.playerMaxSpeed) : self.playerSpeed = -self.playerMaxSpeed

        for fruit in self.ripefruits:
            if (fruit.y > self.height):
                self.ripefruits.remove(fruit)
            else:
                if (fruit.x > self.playerXpos - self.playerWidthHalf) and (fruit.x < self.playerXpos + self.playerWidthHalf):
                    pMin =  self.playerYpos - self.playerHeightHalf
                    pMax =  self.playerYpos + self.playerHeightHalf
                    if (fruit.y > pMin) and (fruit.y < pMax):
                        self.ripefruits.remove(fruit)
                        self.scores[0] += 1
                        self.updateScoreText()

        if (len(self.ripefruits) < self.maxNumFruits):
                self.spawnFruit()

    def updateScoreText(self):
        if not self.with_graphics:
            return
        self.scoreSurface = self.textFont.render(str(self.scores[0]),  False, (255, 255,255))

    def render(self):
        if not self.with_graphics:
            return

        self.surface.fill((0,0,0))
        rect = pygame.Rect(self.playerXpos - self.playerWidthHalf, self.playerYpos - self.playerHeightHalf, self.playerWidth, self.playerHeight)
        pygame.draw.rect(self.surface, (0, 0, 255), rect)
        for fruit in self.ripefruits:
            pygame.draw.circle(self.surface, (0, 255, 0), (fruit.x, fruit.y), 5)

        self.surface.blit(self.scoreSurface, (5,5))
        pygame.display.flip()

    # For Traning:
    def get_state(self):
        # Player x pos, player speed, fruit x, fruit y
        s = np.zeros(4)
        s[0] = self.playerXpos / self.width
        s[1] = self.playerSpeed / self.playerMaxSpeed
        if (len(self.ripefruits) > 0):
            s[2] = self.ripefruits[0].x / self.width
            s[3] = self.ripefruits[0].x / self.height
        return s

    def get_state_size(self):
        return [4]
    
    def get_number_actions(self):
        return 3

    def reset(self):
        self.ripefruits = []
        self.playerSpeed = 0.0
        self.playerXpos = self.width/2
        self.scores[0] = 0

    def get_score(self):
        return self.scores[0]
