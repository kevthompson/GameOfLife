# Python code to implement Conway's Game Of Life
import numba
from numba import guvectorize, int64
from numba import cuda
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

# setting up the values for the grid
ON = 1
OFF = 0
vals = [ON, OFF]
I_W = 5

# testGrid = np.array([[1, 1, 0, 0, 0, 0],[1, 1, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0]], np.int32)
testGrid = np.zeros((I_W, I_W))
class player:
    def __init__(self, gridWidth=30, initWidth=5):
        # set grid size
        self.gridWidth = gridWidth
        self.initWidth = initWidth
        self.xStart = (self.gridWidth-self.initWidth) // 2
        self.yStart = (self.gridWidth-self.initWidth) // 2

        # set animation update interval
        self.length = 50
        # self.setGrid(testGrid)
        self.tempGrid = np.zeros((self.gridWidth, self.gridWidth)).astype(int)
        self.randomGrid()

        # average number of live tiles
        self.fitness = 0

    def setGrid(self, newGrid):
        self.grid = np.zeros((self.gridWidth, self.gridWidth)).astype(int)
        self.initGrid = newGrid.astype(int)
        self.grid[self.yStart:(self.yStart+self.initWidth)%self.gridWidth, self.xStart:(self.xStart+self.initWidth)%self.gridWidth] = self.initGrid

    def resetGrid(self):
        self.grid = np.zeros((self.gridWidth, self.gridWidth)).astype(int)
        self.grid[self.yStart:(self.yStart+self.initWidth)%self.gridWidth, self.xStart:(self.xStart+self.initWidth)%self.gridWidth] = self.initGrid

    def randomGrid(self):
        self.grid = np.zeros((self.gridWidth, self.gridWidth)).astype(int)
        self.initGrid = np.random.choice(vals, self.initWidth*self.initWidth, p=[0.5, 0.5]).reshape(self.initWidth, self.initWidth).astype(int)
        self.grid[self.yStart:(self.yStart+self.initWidth)%self.gridWidth, self.xStart:(self.xStart+self.initWidth)%self.gridWidth] = self.initGrid


    def update(self, frameNum, img, fig,):
        if frameNum == 0:
            self.fitness = 0

        # update data
        self.compute()
        img.set_data(self.grid)   

        if frameNum == self.length-1:
            plt.close(fig)
            # print(self.fitness)
            # self.setGrid(testGrid)
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
                             self.grid[(i+1)%self.gridWidth, (j-1)%self.gridWidth] + self.grid[(i+1)%self.gridWidth, (j+1)%self.gridWidth])/ON)
                
                # apply Conway's rules
                if self.grid[i, j]  == ON:
                    if (total < 2) or (total > 3):
                        tempGrid[i, j] = OFF
                else:
                    if total == 3:
                        tempGrid[i, j] = ON

        self.grid[:] = tempGrid[:]
        self.fitness += sum(sum(self.grid))
    
    def display(self):
        # set up animation
        fig, ax = plt.subplots()
        img = ax.imshow(self.grid, interpolation='nearest')
        ani = animation.FuncAnimation(fig, self.update, fargs=(img, fig,),
                                      frames = self.length,
                                      interval=30,
                                      save_count=50)
        plt.show() 

    def run(self):
        for frameNum in range(self.length):
            self.compute()
        # print(self.fitness)
        self.setGrid(testGrid)
        self.fitness = 0


class population:
    def __init__(self, size, grid, init):
        self.pool = [ player(grid, init) for x in range(size) ]
        self.initWidth = init
        self.gridWidth = grid
        self.size = size
        self.xStart = (grid-init)//2
        self.yStart = (grid-init)//2

    def update(self, count=1):
        for ind in self.pool:
            for i in range(count):
                ind.compute()              

    def parallel_update(self, count=1):
        for i in range(count):
            for individual in self.pool:
                parallel_compute(individual.grid, individual.gridWidth, individual.tempGrid)
                individual.grid[:] = individual.tempGrid[:].astype(int)
                individual.fitness += sum(sum(individual.grid))
                # print(individual.grid, individual.fitness)
                individual.tempGrid = np.ndarray((self.gridWidth, self.gridWidth))

    def reset(self):
        for ind in self.pool:
            ind.resetGrid()

    def fitness(self):
        summed = sum([x.fitness for x in self.pool])
        return summed / (self.size * 1.0)

    def generateGrid(self, index1, index2):
        parent1 = self.pool[index1]
        parent2 = self.pool[index2]
        childGrid = np.zeros((self.initWidth, self.initWidth)).astype(int)
        #Crossover
        for i in range(self.initWidth):
            splitPoint = np.random.randint(self.initWidth)
            temp1 = parent1.initGrid[i][:splitPoint]
            temp2 = parent2.initGrid[i][splitPoint:]
            childGrid[i] = np.append(temp1, temp2)

        return childGrid

    def evolve(self, retain=0.2, random_select=0.05, mutate=0.01):
        self.reset()
        graded = [ (x.fitness, x) for x in self.pool]
        graded = [ x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]
        retain_length = int(len(graded)*retain)
        parents = graded[:retain_length]     

        # randomly add other individuals to promote genetic diversity
        for ind in graded[retain_length:]:
            if random_select > np.random.random():
                parents.append(ind)
        
        #mutation
        for ind in parents:
            if mutate > np.random.random():
                x_modify = np.random.randint(0,ind.initWidth-2)+ind.xStart
                y_modify = np.random.randint(0,ind.initWidth-2)+ind.yStart
                ind.grid[x_modify:x_modify+2, y_modify:y_modify+2] = np.random.choice(vals, 4, p=[0.5, 0.5]).reshape(2, 2).astype(int)
        
        parents_length = len(parents)
        desired_length = self.size - parents_length
        children = []
        while len(children) < desired_length:
            male = np.random.randint(0, parents_length-1)
            female = np.random.randint(0, parents_length-1)
            if male != female:
                child = player(self.gridWidth, self.initWidth)
                child.setGrid(self.generateGrid(male, female))
                children.append(child)

        parents.extend(children)
        self.pool = parents

    def save(self):
        f = open("data.txt", "w+")
        f.write("Size: %d\r\nInitWidth: %d\r\nGridWidth: %d\r\n" %(self.size, self.initWidth, self.gridWidth));
        for individual in self.pool:
            f.write("%s,%d\n" %(individual.initGrid, individual.fitness))

# Run GOL algorithm
@guvectorize([(int64[:,:], int64, int64[:,:])], '(n,n),()->(n,n)')
def parallel_compute(grid, width, tempGrid):

    # compute 8-neghbor sum
    # using toroidal boundary conditions - x and y wrap around 
    # so that the simulaton takes place on a toroidal surface.
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            total = int((grid[i, (j-1)%width] + grid[i, (j+1)%width] +
                         grid[(i-1)%width, j] + grid[(i+1)%width, j] +
                         grid[(i-1)%width, (j-1)%width] + grid[(i-1)%width, (j+1)%width] +
                         grid[(i+1)%width, (j-1)%width] + grid[(i+1)%width, (j+1)%width])/ON)
            
            # apply Conway's rules
            if grid[i, j]  == ON:
                if (total < 2) or (total > 3):
                    tempGrid[i, j] = OFF
            else:
                if total == 3:
                    tempGrid[i, j] = ON

# call main
if __name__ == '__main__':
    cuda.select_device(0)
    p_count = 16   # less than 32
    i_width = 8 * 3 # multiple of 8, 2 should be optimal unless multidimensional blocks are added
    i_init = 8
    lifespan = 100
    generationCount = 20
    epochCount = 5
    p = population(p_count, i_width, i_init)
    fitness_history = [p.fitness(),]
    for epoch in range(epochCount):
        for generation in range(generationCount):
            p.parallel_update(lifespan)
            p.evolve()
            p.save()
            fitness_history.append(p.fitness())

        # for datum in fitness_history:
        #    print(datum)
        p.pool[0].display()