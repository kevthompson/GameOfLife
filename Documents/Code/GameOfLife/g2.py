# Python code to implement Conway's Game Of Life
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

# setting up the values for the grid
ON = 255
OFF = 0
vals = [ON, OFF]

class Game:
    def __init__(self, repeat=True):
        # set grid size
        self.gridWidth = 70
        self.initWidth = 10
        self.repeat = repeat

        # set animation update interval
        self.length = 150
        self.grid = np.zeros((self.gridWidth, self.gridWidth))
        self.randomGrid()

        # average number of live tiles
        self.fitness = 0

    def display(self):
        # set up animation
        fig, ax = plt.subplots()
        img = ax.imshow(self.grid, interpolation='nearest')
        ani = animation.FuncAnimation(fig, self.update, fargs=(img,),
                                      frames = self.length,
                                      interval=30,
                                      save_count=50,
                                      repeat=self.repeat)
        plt.show() 

    def randomGrid(self):
        self.grid = np.zeros((self.gridWidth, self.gridWidth))
        self.xStart = np.random.randint(self.gridWidth-self.initWidth)
        self.yStart = np.random.randint(self.gridWidth-self.initWidth)
        self.grid[self.yStart:(self.yStart+self.initWidth)%self.gridWidth, self.xStart:(self.xStart+self.initWidth)%self.gridWidth] = np.random.choice(vals, self.initWidth*self.initWidth, p=[0.5, 0.5]).reshape(self.initWidth, self.initWidth)

    def update(self, frameNum, img, ):
        if frameNum == 0:
            self.randomGrid()

        # update data
        self.compute()
        img.set_data(self.grid)     
        return img,

    def compute(self):
        # copy grid since we require 8 neighbors 
        # for calculation and we go line by line 
        tempGrid = self.grid.copy()
        for i in range(self.gridWidth):
            for j in range(self.gridWidth):
                # compute 8-neghbor sum
                # using toroidal boundary conditions - x and y wrap around 
                # so that the simulaton takes place on a toroidal surface.
                total = int((self.grid[i, (j-1)%self.gridWidth] + self.grid[i, (j+1)%self.gridWidth] +
                             self.grid[(i-1)%self.gridWidth, j] + self.grid[(i+1)%self.gridWidth, j] +
                             self.grid[(i-1)%self.gridWidth, (j-1)%self.gridWidth] + self.grid[(i-1)%self.gridWidth, (j+1)%self.gridWidth] +
                             self.grid[(i+1)%self.gridWidth, (j-1)%self.gridWidth] + self.grid[(i+1)%self.gridWidth, (j+1)%self.gridWidth])/255)
     
                # apply Conway's rules
                if self.grid[i, j]  == ON:
                    if (total < 2) or (total > 3):
                        tempGrid[i, j] = OFF
                else:
                    if total == 3:
                        tempGrid[i, j] = ON

        self.grid[:] = tempGrid[:]


# call main
if __name__ == '__main__':
    game = Game()
    game.display()
