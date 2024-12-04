import numpy as np
import matplotlib.pyplot as plt
import numba
import random
from scipy.optimize import curve_fit
import sys
sys.setrecursionlimit(1000000)

@numba.jit()
def add_grain_of_sand(x, y, grid,l, size_of_avalanche, mass):
    #Adds grains of sand in x,y
    if x==0 or x==l-1 or y==0 or y==l-1:
        mass-=1
        return size_of_avalanche, mass
    if grid[x,y]>=1:
        size_of_avalanche+=1
        neighbours = [[x+1,y],[x-1,y],[x,y+1],[x,y-1]]
        grid[x,y] -=1
        pairs = [[0,0],[0,1],[0,2],[0,3],[1,1],[1,2],[1,3],[2,2],[2,3],[3,3]]
        pair = random.choice(pairs)
        neighbour1_x = neighbours[pair[0]][0]
        neighbour1_y = neighbours[pair[0]][1]
        neighbour2_x = neighbours[pair[1]][0]
        neighbour2_y = neighbours[pair[1]][1]
        
        size_of_avalanche, mass = add_grain_of_sand(neighbour1_x,neighbour1_y,grid,l,size_of_avalanche,mass)
        size_of_avalanche, mass = add_grain_of_sand(neighbour2_x,neighbour2_y,grid,l,size_of_avalanche, mass)
        
    else:
        grid[x,y]+=1
    return size_of_avalanche, mass

@numba.jit()
def add_grain_of_sand_3(x, y, x0,y0, grid, size_of_avalanche,up,down,right,left): 
    #this function also adds grainf of sand but keeps track of the largest linear dimention
    dx = x0-x
    dy = y0-y
    if dy<0:
        down = np.min([down,dy])
    else:
        up = np.max([up,dy])
    if dx>0:
        left = np.max([left,dx])
    else:
        right = np.min([right,dx])
    if x==0 or x==len(grid)-1 or y==0 or y==len(grid)-1:
        return size_of_avalanche, grid,up,down,right,left
    if grid[x,y]>=1:
        size_of_avalanche+=1
        
        grid[x,y] =0
        neighbours = [[x+1,y],[x-1,y],[x,y+1],[x,y-1]]
        
        grid[x,y] -=1
        pairs = [[0,0],[0,1],[0,2],[0,3],[1,1],[1,2],[1,3],[2,2],[2,3],[3,3]]
        pair = random.choice(pairs)
        neighbour1_x = neighbours[pair[0]][0]
        neighbour1_y = neighbours[pair[0]][1]
        neighbour2_x = neighbours[pair[1]][0]
        neighbour2_y = neighbours[pair[1]][1]

        size_of_avalanche,grid, up,down,right,left = add_grain_of_sand_3(neighbour1_x,neighbour1_y,x0,y0,grid,size_of_avalanche,up,down,right,left)
        size_of_avalanche,grid, up,down,right,left = add_grain_of_sand_3(neighbour2_x,neighbour2_y,x0,y0,grid,size_of_avalanche,up,down,right,left)
        
    else:
        grid[x,y]+=1
    return size_of_avalanche,grid,up,down,right,left

@numba.jit()
def power_law(x, a, b):
    return a*np.power(x, b)

@numba.jit()
def question_1(l, iterations):
    grid = np.zeros([l+2,l+2])#the grid has an extra row and collumn in each side which acts like the window in boureocracy model
    sizes_of_avalanches = np.zeros([iterations])
    masses = np.zeros([iterations])
    mass=0
    for i in range(iterations):
        mass+=1
        size_of_avalanche = 0
        x = random.randint(1,l)#x coordinate of the random placement
        y = random.randint(1,l)#y coordinate of the random placement
        sizes_of_avalanches[i], mass = add_grain_of_sand(x,y,grid,l+2,size_of_avalanche, mass)
        masses[i] = mass
    #plotting and saving the data
    fig, ax1 = plt.subplots(figsize=(10,7))
    print('avararage number of grains per site = ',np.round(np.mean(masses[1000:2000])/l**2,2))
    ax2 = ax1.twinx()
    ax1.plot(sizes_of_avalanches, color='navy')
    plt.style.use('seaborn-whitegrid')

    ax2.plot(masses/(l**2), color='firebrick')

    ax1.set_xlabel('Update step', size=15)
    ax1.set_ylabel('Avalanche size', size=15, color='navy')
    ax2.set_ylabel(r'$\langle$ mass $\rangle$', size=15, color='firebrick')
    ax2.set_ylim([0,1])
    plt.savefig('Avalanche_size(Time)_'+str(l))



@numba.jit()
def question_2():
    ls = [50,100,200]
    iterationss = [4000,15000,40000]
    sample = 5000000
    j=0
    for l in ls:
        grid = np.zeros([l+2,l+2])
        iterations = iterationss[j]
        sizes_of_avalanches = []
        for i in range(iterations):
        
            size_of_avalanche = 0
            x = random.randint(1,l)
            y = random.randint(1,l)
            add_grain_of_sand(x,y,grid,l+2,size_of_avalanche)

        for i in range(sample):
        
            size_of_avalanche = 0
            x = random.randint(1,l)
            y = random.randint(1,l)
            size_of_avalanche = add_grain_of_sand(x,y,grid,l+2,size_of_avalanche)
            sizes_of_avalanches.append(size_of_avalanche)

        count_avalanches_distribution = np.zeros([int(np.array(sizes_of_avalanches).max())])

        for i in range(len(count_avalanches_distribution)):
            count_avalanches_distribution[i]=sizes_of_avalanches.count(i)/sample
        x = range(max(sizes_of_avalanches))
        y= count_avalanches_distribution
        #saving the data
        np.savetxt('count_avalanches_distribution_BIG'+str(l)+'.txt',(x,y), newline='\r\n')
        print(max(sizes_of_avalanches))
        j+=1

def question_plot_2():
    #this function requires the txt file that question_2 creates
    ls = [200,100,50,25]
    plt.figure(6)

    parameters = []
    for l in ls:
        x,y = np.loadtxt('count_avalanches_distribution_'+str(l)+'.txt')
        plt.scatter(x, y, label = 'L ='+str(l))
        res = y - power_law(x, 0.22, -1.1)
        pars, cov = curve_fit(f=power_law, xdata=x[10:200], ydata=y[10:200], p0=[0.2, -1.2], bounds=(-2, 2))
        parameters.append(pars)
        print(pars)
        stdevs = np.sqrt(np.diag(cov))
        res = y - power_law(x, *pars)
        plt.yscale('log')
        plt.xscale('log')

    plt.xlabel("Size of avalanches")
    plt.ylabel("Possibility")
    plt.text(1000,0.001,"Slope ="+str(np.round(parameters[0][1],3)))
    plt.plot(x, power_law(x, parameters[0][0], parameters[0][1]),color='black')
    plt.savefig('Probability(Avalanche_Size)')

    return parameters

def question_plot_2_cut_off():
    #this function plots the cut-off graph but also requires the txt from question_2
    plt.figure(7)
    
    ls = [200,100,50,25]
    for l in ls:
        x,y = np.loadtxt('count_avalanches_distribution_'+str(l)+'.txt')
        plt.scatter(x/(l**2.4), y*(x**1.211), label = 'L ='+str(l))

        plt.yscale('log')
        plt.xscale('log')
    plt.ylabel("y(x^2.21)")
    plt.xlabel(f'$x/l^D)$')
    plt.savefig("Cut-off")

    

@numba.jit()
def question_3():
    l = 200
    iterations = 30000
    sample = 1000000
    grid = np.zeros([l+2,l+2])
    sizes_of_avalanches = []
    largest_linear_dimensions = []
    for i in range(iterations):
        
        up,down,right,left = 0,0,0,0
        size_of_avalanche = 0
        x = random.randint(1,l)
        y = random.randint(1,l)
        add_grain_of_sand_3(x,y,x,y,grid,size_of_avalanche, up,down,right,left)

    for i in range(sample):
        up,down,right,left = 0,0,0,0
        size_of_avalanche = 0
        x0 = random.randint(1,l)
        y0 = random.randint(1,l)
        size_of_avalanche, grid, up,down,right,left = add_grain_of_sand_3(x0,y0,x0,y0,grid,size_of_avalanche, up,down,right,left)
        largest_linear_dimension = np.max([(np.abs(up)+np.abs(down)),(np.abs(left)+np.abs(right))])
        sizes_of_avalanches.append(size_of_avalanche)
        largest_linear_dimensions.append(largest_linear_dimension)

    #saving the data in a txt file
    x= largest_linear_dimensions
    y= sizes_of_avalanches
    np.savetxt('size_of_avalanche(spread)_'+str(l)+'.txt',(x,y), newline='\r\n')



def question_plot_3():
    #this function requires the txt file that question_3 creates
    
    l = 100
    x,y = np.loadtxt('size_of_avalanche(spread)'+str(l)+'.txt')
    plt.figure(3)
    plt.scatter(x, y, label = 'L ='+str(l))
    plt.xlabel("Largest linear dimension")
    plt.ylabel("Size of avalanche")
    pars, cov = curve_fit(f=power_law, xdata=x[1:], ydata=y[1:], p0=[1, 2], bounds=(-1, 3))
    plt.text(20,8000,"D ="+str(np.round(pars[1],2)))
    plt.plot(np.linspace(0,100,1000),power_law(np.linspace(0,100,1000),*pars), color = 'red')
    plt.savefig('size_of_avalanche(spread)')

@numba.jit()
def question_4(l):
    iterations = 12000
    sample = 1000000
    grid = np.zeros([l+2,l+2])
    sizes_of_avalanches = []
    mass = 0
    for i in range(iterations):
        a = random.randint(1,l+1)
        key = random.randint(0,1)
        if key == 0:
            x=1
            y=a
        else:
            y =1
            x = a
        size_of_avalanche = 0
        add_grain_of_sand(x,y,grid,l+2,size_of_avalanche, mass)

    for i in range(sample):
        a = random.randint(1,l+1)
        key = random.randint(0,1)
        if key == 0:
            x=1
            y=a
        else:
            y =1
            x = a
        size_of_avalanche = 0
        size_of_avalanche,mass = add_grain_of_sand(1,1,grid,l+2,size_of_avalanche,mass)
        sizes_of_avalanches.append(size_of_avalanche)

    count_avalanches_distribution = np.zeros([int(np.array(sizes_of_avalanches).max())])

    for i in range(len(count_avalanches_distribution)):
        count_avalanches_distribution[i]=sizes_of_avalanches.count(i)/sample
    x = range(max(sizes_of_avalanches))
    y= count_avalanches_distribution
    np.savetxt('count_avalanches_distribution_edge'+str(l)+'.txt',(x,y), newline='\r\n')
    print(max(sizes_of_avalanches))
    
def question_plot_4(l):
    #this function requires the txt file that question_4 creates
    
    x,y = np.loadtxt('count_avalanches_distribution_edge'+str(l)+'.txt')
    plt.figure(4)
    plt.scatter(x, y, label = 'L ='+str(l))
    pars, cov = curve_fit(f=power_law, xdata=x[2:100], ydata=y[2:100], p0=[0.2, -1.2], bounds=(-5, 5))
    plt. plot(x[x<2*10**2], power_law(x[x<2*10**2], *pars),color='black')
    plt.text(105,0.01,"Slope ="+str(np.round(pars[1],3)))
    print(pars)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Size of avalanches")
    plt.ylabel("Probability")
    plt.savefig('AvalanchesSizeDIstributionEdge')

#Code Execution
    
question_1(25,2000)
question_2()
question_plot_2()  #needs the file that question_2() function creates
question_plot_2_cut_off()  #needs the file that question_2() function creates

question_3()
question_plot_3()  #needs the file that question_3() function creates

question_4(100)
question_plot_4(100)  #needs the file that question_4() function creates 
plt.legend(loc='best')
plt.show()