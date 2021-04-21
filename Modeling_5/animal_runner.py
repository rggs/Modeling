#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 21:26:55 2020

@author: ryanswope
"""
import numpy as np
import matplotlib.pyplot as plt
from animals import World, Animal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



#Grid Side Length
A=20
def count(grid, val):
    count=0
    for i in grid:
        for j in i:
            if j==val:
                count+=1
    return count

AnWorld=World(A)

#Number of initial animals
'''
F=10

guy_list=[]

#Fox Creation
for i in range(0,F):
    spot=AnWorld.findSpot(2)
    guy_list.append(Animal(AnWorld, 2, spot[0], spot[1] ))
    
    
#Wolves
W=0

for i in range(0,W):
    spot=AnWorld.findSpot(3)
    guy_list.append(Animal(AnWorld, 3, spot[0], spot[1] ))
    
#Rabbit Creation
R=20
for i in range(0,R):
    
    spot=AnWorld.findSpot(1)
    guy_list.append(Animal(AnWorld, 1, spot[0], spot[1] ))
'''

#Zombies
F=2
guy_list=[]
for i in range(0,F):
    spot=AnWorld.findSpot(5)
    guy_list.append(Animal(AnWorld, 5, spot[0], spot[1] ))

#Humans
R=398
for i in range(0,R):
    
    spot=AnWorld.findSpot(4)
    guy_list.append(Animal(AnWorld, 4, spot[0], spot[1] ))




#Simulate
steps=[0]
allDead=False
numFox=[F]
numRab=[R]
#numWolf=[W]

while not allDead:
    grid=AnWorld.grid
    # f1=plt.figure()
    # ax1=f1.add_subplot(111)
    # for guy in guy_list:
    #     if guy.state==1:
    #         ax1.scatter(guy.xpos,guy.ypos,color='blue', s=0.05)
    #     elif guy.state==2:
    #         ax1.scatter(guy.xpos,guy.ypos,color='red', s=0.1)
            
    # ax1.set_xlabel('X Position')
    # ax1.set_ylabel('Y Position')
    # ax1.set_title('Fox and Rabbit Locations')
    # ax1.set_xlim(-1,A)
    # ax1.set_ylim(-1,A)
    # filename='/home/users/rswope/Modeling/Stochastic/AnimGif/animal'+str(steps[-1]+1000)+'.png'
    # plt.savefig(filename,dpi=300)
    # plt.gca()
        

    
    for guy in guy_list:
        temp=guy.act()
        if isinstance(temp, Animal):
            guy_list.append(temp)
    for guy in guy_list:
        guy.move()
        if guy.state==0:
            guy_list.remove(guy)
        
    
    if 2 in AnWorld.grid or 1 in AnWorld.grid or 3 in AnWorld.grid or 4 in AnWorld.grid:
        allDead=False
    if (0 not in AnWorld.grid) and (2 not in AnWorld.grid) and (3 not in AnWorld.grid) and (4 not in AnWorld.grid):
        allDead=True
    elif 2 not in AnWorld.grid and 1 not in AnWorld.grid and 3 not in AnWorld.grid and 4 not in AnWorld.grid:
        allDead=True
        
    #Counter Values changed to zombie/human values
    fox_counter=count(AnWorld.grid,5)
    numFox.append(fox_counter)
    rab_counter=count(AnWorld.grid,4)
    numRab.append(rab_counter)
    #wolf_counter=count(AnWorld.grid,3)
    #numWolf.append(wolf_counter)

    steps.append(steps[-1]+1)
    print('Steps='+ str(steps[-1]))
    #print('Number of Foxes='+ str(numFox[-1]))
    #print('Number of Rabbits='+ str(numRab[-1]))
    #print('Number of Wolves='+ str(numWolf[-1]))
    print('Number of Zombies='+ str(numFox[-1]))
    print('Number of Humans='+ str(numRab[-1]))
''' 
f4=plt.figure()
ax4=f4.add_subplot(111)
ax4.plot(steps, numFox, label='Fox Population', color='orange')
ax4.plot(steps,numRab, label='Rabbit Population', color='brown')
ax4.plot(steps,numWolf, label='Wolf Population', color='black')

ax4.set_xlabel('Steps')
ax4.set_ylabel('Population')
ax4.set_title('Population Dynamics')
ax4.legend()

f5=plt.figure()
ax5=f5.add_subplot(111)
ax5.plot(numFox, numRab, label='Rabbits')
ax5.plot(numFox, numWolf, label='Wolves')
ax5.set_xlabel("Number of Foxes")
ax5.set_ylabel('Other Population')
ax5.set_title('Fox and Rabbit Discretetized Phase Space, R0='+str(R)+', F0='+str(F))
ax5.legend()

f6=plt.figure()
ax6=f6.add_subplot(111, projection='3d')
ax6.plot(numRab, numFox, zs=numWolf)
ax6.set_xlabel('Rabbit Population')
ax6.set_ylabel('Fox Population')
ax6.set_zlabel('Wolf Population')
ax6.set_title('Three Population Phase Space')

f6=plt.figure()
ax6=f6.add_subplot(111, projection='3d')
ax6.plot_trisurf(numRab, numFox, numWolf)
ax6.set_xlabel('Rabbit Population')
ax6.set_ylabel('Fox Population')
ax6.set_zlabel('Wolf Population')
ax6.set_title('Three Population Phase Space')
'''

f4=plt.figure()
ax4=f4.add_subplot(111)
ax4.plot(steps, numFox, label='Zombie Population', color='green')
ax4.plot(steps,numRab, label='Human Population', color='orange')
ax4.set_xlabel('Steps')
ax4.set_ylabel('Population')
ax4.set_title('Population Dynamics')
ax4.legend()

f5=plt.figure()
ax5=f5.add_subplot(111)
ax5.plot(numFox, numRab)
ax5.set_xlabel("Number of Zombies")
ax5.set_ylabel('Number of Humans')
ax5.set_title('Fox and Rabbit Discretetized Phase Space, H0='+str(R)+', Z0='+str(F))
ax5.legend()