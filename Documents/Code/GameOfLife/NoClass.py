# Python code to implement Conway's Game Of Life
import numba
from numba import cuda, int32
import numpy as np

# setting up the values for the grid
ON = 1
OFF = 0
vals = [ON, OFF]

# hyperparameters
gridWidth = 8*3 # multiple of 8 for cuda
initWidth = 3
generationCount = 15
p_count = 16

xStart = (gridWidth-initWidth) // 2
yStart = (gridWidth-initWidth) // 2

def setGrid(newGrid):
    grid = np.zeros((gridWidth, gridWidth)).astype(int)
    grid[yStart:(yStart+newGrid.shape[0])%gridWidth, xStart:(xStart+newGrid.shape[0])%gridWidth] = newGrid.astype(int)

def randomGrid():
    grid = np.zeros((gridWidth, gridWidth)).astype(int)
    grid[yStart:(yStart+initWidth)%gridWidth, xStart:(xStart+initWidth)%gridWidth] = np.random.randint(2, size=(initWidth, initWidth))
    return grid

def newPop(size):
    pop = np.empty((size, gridWidth, gridWidth))
    pop[:] = [ randomGrid() for x in range(size) ]
    fitness = [0 for x in range(size)]
    return updatePop(pop, fitness)

def resetPop():
    pop = i_pop.copy()
    fitness = np.zeros(pop.shape[0]).astype(int)
    return pop, fitness

def updatePop(pop, fitness, count=2):
    for i in range(count):
        griddim = pop.shape[0]
        blockdim = gridWidth, gridWidth
        res = np.zeros(pop.shape)
        
        stream = cuda.stream()
        dpop = cuda.to_device(pop, stream=stream)
        dres = cuda.to_device(res, stream=stream)
        parallel_compute[griddim, blockdim](dpop, dres)
        dres.to_host(stream)
        stream.synchronize()
        pop = res.astype(int)
        tmpfitness = np.sum(pop, axis=(1,2))
        print(tmpfitness)
    return pop, tmpfitness

@cuda.jit(argtypes=(int32[:,:,:],int32[:,:,:]), target='gpu')
def parallel_compute(pop, res):
    blocknum = cuda.blockIdx.x
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y
    res[blocknum, i, j] = 1
    if i < gridWidth and j < gridWidth:
        res[0, i, j] = 1
        total = 0
        if(j>0):
            total += pop[blocknum, i, (j-1)%gridWidth]
            if(i>0):
                total += pop[blocknum, (i-1)%gridWidth, (j-1)%gridWidth] 
            if(i<gridWidth-1):
                total += pop[blocknum, (i+1)%gridWidth, (j-1)%gridWidth] 
        if(j<gridWidth-1):
            total += pop[blocknum, i, (j+1)%gridWidth] 
            if(i<gridWidth-1):
                total += pop[blocknum, (i+1)%gridWidth, (j+1)%gridWidth]
            if(i>0):
                total += pop[blocknum, (i-1)%gridWidth, (j+1)%gridWidth] 
        if(i<gridWidth-1):        
            total += pop[blocknum, (i+1)%gridWidth, j] 
        if(i>0):
            total += pop[blocknum, (i-1)%gridWidth, j]
        total = int(total/ON)
        cuda.syncthreads()
        # apply Conway's rules
        if pop[blocknum, i, j]  == ON:
            if (total < 2) or (total > 3):
                pop[blocknum, i, j] = OFF
        else:
            if total == 3:
                pop[blocknum, i, j] = ON
        
def breed(index1, index2):
    parent1 = pop[index1]
    parent2 = pop[index2]
    childGrid = np.zeros((gridWidth, gridWidth)).astype(int)
    #Crossover
    for i in range(initWidth):
        splitPoint = np.random.randint(initWidth)
        temp1 = parent1[i+xStart][yStart:splitPoint+yStart]
        temp2 = parent2[i+xStart][splitPoint+yStart:yStart+initWidth]
        childGrid[i+xStart,yStart:yStart+initWidth] = np.append(temp1, temp2)

    return childGrid

def evolve(pop, fitness, retain=0.2, random_select=0.05, mutate=0.01):
    pop, fitness = resetPop()
    graded = [ (fitness[x], pop[x]) for x in range(pop.shape[0])]
    tempFitness = [ x[0] for x in sorted(graded, key=lambda x: x[0], reverse=True)]
    graded = [ x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]
    retain_length = int(len(graded)*retain)
    parents = np.asarray(graded[:retain_length])
    fitness = tempFitness[:retain_length]

    # randomly add other individuals to promote genetic diversity
    for pos, ind in enumerate(graded[retain_length:]):
        if random_select > np.random.random():
            np.append(parents[:], ind)
            fitness.append(tempFitness[pos])

    #mutation
    for pos, ind in enumerate(parents):
        if mutate > np.random.random():
            x_modify = np.random.randint(0,initWidth-2)+xStart
            y_modify = np.random.randint(0,initWidth-2)+yStart
            ind[x_modify:x_modify+2, y_modify:y_modify+2] = np.random.randint(2, size=(2, 2)).astype(int)
            fitness[pos] = 0
            
    parents_length = len(parents)
    desired_length = pop.shape[0] - parents_length
    children = np.empty((desired_length, gridWidth, gridWidth))

    i=0
    while i < desired_length:
        male = np.random.randint(0, parents_length-1)
        female = np.random.randint(0, parents_length-1)
        if male != female:
            child = breed(male, female)
            children[i] = child
            fitness.append(0)
            i+=1
    
    children, tmpfitness = updatePop(children, fitness)
    parents = np.append(parents, children, axis=0)
    fitness.extend(tmpfitness)
    i_pop = parents.copy()
    
    return i_pop, fitness

def save(pop, fitness):
    f = open("data.txt", "w+")
    f.write("Population Size: %d\nInitWidth: %d\nGridWidth: %d\n" %(pop.shape[0], initWidth, gridWidth));
    for pos in range(pop.shape[0]):
        f.write("%s,%d\n" %(i_pop[pos], fitness[pos]))

# call main
if __name__ == '__main__':
    numba.cuda.select_device(0)
    pop, fitness = newPop(p_count)
    i_pop = pop.copy()
    fitness_history = []
    for generation in range(generationCount):
        pop, fitness = evolve(pop, fitness)
        save(pop, fitness)
        fitness_history.append(sum(fitness))

    for datum in fitness_history:
       print(datum)
