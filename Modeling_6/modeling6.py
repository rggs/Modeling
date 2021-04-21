#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 21:17:40 2020

@author: ryanswope
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
from scipy import stats
from scipy.optimize import curve_fit
import time

k=1.38e-23
g=9.81
size=20
beads=np.zeros((20,20))
beads[0,:]=1

def anneal(energy, temp):
    P=np.exp(-energy/(temp))
    
    if P>np.random.rand():
        return True
    else:
        return False

# def cosh(x, *params):
#     a,b,c=params
#     return a*np.cosh((x+b)/a)+c

# def beadEnergy(beads,pos,K):
#     if pos==0:
#         y=-1*list(beads[:,pos]).index(1)
#         yright=y- -1*list(beads[:,pos+1]).index(1)
#         yleft=y
        
#     elif pos==19:
#         y=-1*list(beads[:,pos]).index(1)
#         yleft=y- -1*list(beads[:,pos-1]).index(1)
#         yright=y
        
#     else:
#         y=-1*list(beads[:,pos]).index(1)  
#         yright=y- -1*list(beads[:,pos+1]).index(1)
#         yleft=y- -1*list(beads[:,pos-1]).index(1)
#     E=.5*K*(yright)**2 + .5*K*(yleft)**2 + g*y
      
#     return E

# def moveBead(beads, pos, K, temp):
#     energy1=beadEnergy(beads,pos,K)
#     new_beads=np.copy(beads)
#     direct=np.random.rand()
#     verpos=list(beads[:,pos]).index(1)
#     if direct<.5:
#         if verpos==19:
#             pass
#         else:
#             new_beads[verpos,pos]=0
#             verpos+=1
#     elif direct>=.5:
#         if verpos==0:
#             pass
#         else:
#             new_beads[verpos,pos]=0
#             verpos-=1
#     new_beads[verpos,pos]=1
#     energy2=beadEnergy(new_beads,pos,K)
#     if energy2<energy1:
#         return new_beads
#     else:
#         if anneal((energy2-energy1),temp):
#             return new_beads
#         else:
#             return beads
    

# T=100
# N=1000*size
# temps=np.linspace(T,0,11)

# energy_arr=[]

# for temp in temps:
#     j=0
#     print(temp)
#     avg_energy=0
#     while j<N:       
#         beads=moveBead(beads,np.random.randint(0,20), 20, temp)
#         j+=1
#         energy=0
#         for i in range(size):
#             energy+=beadEnergy(beads,i,20)
#         avg_energy+=energy
#     energy_arr.append(avg_energy/(N+1))
# print(beads)

# beadx=[]
# beady=[]

# x=0
# for i in beads.T:
#     y=-list(i).index(1)+19
#     beadx.append(x)
#     beady.append(y)
#     x+=1


# popc, popv = curve_fit(cosh,beadx,beady,p0=[4,-10,0])
# cosh_fit=cosh(beadx,*popc)


# f1=plt.figure()
# ax1=f1.add_subplot(111)
# ax1.plot(beadx,beady, 'r.',label='Beads')
# ax1.plot(beadx,cosh_fit,label='Cosh Fit')
# ax1.set_title('Final Bead Locations')
# ax1.set_xlabel('X Position')
# ax1.set_ylabel('Y Position')
# ax1.legend()

# #Fit this with polynomial, cosh and compare?

# line_fit=stats.linregress(temps, energy_arr)
# line_x=np.linspace(temps[-1],temps[0],10)
# line_y=line_fit[0]*line_x +line_fit[1]

# f2=plt.figure()
# ax2=f2.add_subplot(111)
# ax2.scatter(temps,energy_arr, label='Data')
# ax2.set_xlabel('Temperature')
# ax2.set_ylabel('Average Energy')
# ax2.plot(line_x,line_y,color='r',label='Regresssion: a='+str(round(line_fit[0],5))+', b='+str(round(line_fit[1],5))+', R^2='+str(round(line_fit[2],4)))
# ax2.legend()
############################ Ising #########################

# def initialstate(N):   
#     ''' generates a random spin configuration for initial condition'''
#     state = 2*np.random.randint(2, size=(N,N))-1
#     return state


# def mcmove(config, beta):
#     '''Monte Carlo move using Metropolis algorithm '''
#     for i in range(N):
#         for j in range(N):
#                 a = np.random.randint(0, N)
#                 b = np.random.randint(0, N)
#                 s =  config[a, b]
#                 nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]
#                 cost = 2*s*nb
#                 if cost < 0:
#                     s *= -1
#                 elif rand() < np.exp(-cost*beta):
#                     s *= -1
#                 config[a, b] = s
#     return config


# def calcEnergy(config):
#     '''Energy of a given configuration'''
#     energy = 0
#     for i in range(len(config)):
#         for j in range(len(config)):
#             S = config[i,j]
#             nb = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
#             energy += -nb*S
#     return energy/4.


# def calcMag(config):
#     '''Magnetization of a given configuration'''
#     mag = np.sum(config)
#     return mag

# nt      = 22         #  number of temperature points
# N       = 32         #  size of the lattice, N x N
# eqSteps = 2048       #  number of MC sweeps for equilibration
# mcSteps = 2048       #  number of MC sweeps for calculation

# T       = np.linspace(1.53, 3.28, nt); 
# E,M,C,X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
# n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N)
# avg_probs=[]

# for tt in range(nt):
#     print(tt)
#     E1 = M1 = E2 = M2 = 0
#     config = initialstate(N)
#     iT=1.0/T[tt]; iT2=iT*iT;
    
#     for ii in range(eqSteps):         # equilibrate
#         mcmove(config, iT)           # Monte Carlo moves

#     for ij in range(mcSteps):
#         mcmove(config, iT)           
#         Ene = calcEnergy(config)     # calculate the energy
#         Mag = calcMag(config)        # calculate the magnetisation

#         E1 = E1 + Ene
#         M1 = M1 + Mag
#         M2 = M2 + Mag*Mag 
#         E2 = E2 + Ene*Ene

#     E[tt] = n1*E1
#     M[tt] = n1*M1
#     C[tt] = (n1*E2 - n2*E1*E1)*iT2
#     X[tt] = (n1*M2 - n2*M1*M1)*iT
    
#     probs=[]
#     for i in range(len(config)):
#         for j in range(len(config)):
#             if config[i,j]==1:
#                 one_count=0
#                 n_count=0
#                 if i!= 0:
#                     n_count+=1
#                     if config[i-1,j]==1:
#                         one_count+=1
#                 if i!= len(config)-1:
#                     n_count+=1
#                     if config[i+1,j]==1:
#                         one_count+=1
#                 if j!= 0: 
#                     n_count+=1
#                     if config[i,j-1]==1:
#                         one_count+=1
#                 if j!= len(config)-1: 
#                     n_count+=1
#                     if config[i,j+1]==1:
#                         one_count+=1
#                 probs.append(one_count/n_count)
                
#     avg_prob=sum(probs)/len(probs)
#     avg_probs.append(avg_prob)
    
    
    
    
# f = plt.figure(figsize=(18, 10)); # plot the calculated values    

# sp =  f.add_subplot(2, 2, 1 );
# plt.scatter(T, E, marker='o', color='IndianRed')
# plt.xlabel("Temperature (T)");
# plt.ylabel("Energy ");         plt.axis('tight');

# sp =  f.add_subplot(2, 2, 2 );
# plt.scatter(T, abs(M), marker='o', color='RoyalBlue')
# plt.xlabel("Temperature (T)"); 
# plt.ylabel("Magnetization ");   plt.axis('tight');

# sp =  f.add_subplot(2, 2, 3 );
# plt.scatter(M, C, marker='o', color='IndianRed')
# plt.xlabel("Magnetization");  
# plt.ylabel("Specific Heat ");   plt.axis('tight');   

# sp =  f.add_subplot(2, 2, 4 );
# plt.scatter(M, X, marker='o', color='RoyalBlue')
# plt.xlabel("Magnetization"); 
# plt.ylabel("Susceptibility");   plt.axis('tight');


# f1=plt.figure()
# ax1=f1.add_subplot(111)
# ax1.plot(T, avg_probs)
# ax1.set_xlabel('Temperature')
# ax1.set_ylabel('Neighbor Probabilities')
# ax1.set_title('Probability of +1 Spin Neighbor given +1 Spin')

# probs=[]
# for i in range(len(config)):
#     for j in range(len(config)):
#         if config[i,j]==1:
#             one_count=0
#             n_count=0
#             if i!= 0:
#                 n_count+=1
#                 if config[i-1,j]==1:
#                     one_count+=1
#             if i!= len(config)-1:
#                 n_count+=1
#                 if config[i+1,j]==1:
#                     one_count+=1
#             if j!= 0: 
#                 n_count+=1
#                 if config[i,j-1]==1:
#                     one_count+=1
#             if j!= len(config)-1: 
#                 n_count+=1
#                 if config[i,j+1]==1:
#                     one_count+=1
#             probs.append(one_count/n_count)
            
# avg_prob=sum(probs)/len(probs)



#################### Traveling Salesman #####################
# from itertools import permutations 

# def pathDistance(cities,toll=False):
#     distance=0
#     if toll:
#         x,y,tolls= zip(*cities)
#     else:
#         x,y=zip(*cities)
#     for i in range(len(x)-1):
#         d=np.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2)
#         #if toll:
#             #d+=100*((tolls[i+1]-tolls[i])**2)
#         distance+=d
        
#     return distance   



# cities1=[(0,0), (1,0),(2,3),(4,5),(0,5),(5,0),(8,2),(3,4),(9,9),(3,3),(6,7),(8,1)]
# #cities=[(0,0), (1,0),(2,3),(4,5),(0,5),(5,0),(2,1),(3,8)],(3,2)]
# '''

# odyssey=[(38.419589, 20.665490),(39.945599, 26.248911),(40.929242, 25.709940),(34.172745, 9.916010),(37.498106, 13.176425),(36.785461, 11.992596),(36.812331, 8.219195),(39.587721, 2.988063),(40.567337, 0.518741),(38.860765, 8.409876),(38.237527, 15.688259),(35.904657, 14.410413),(38.959907, 1.374839),(41.300790, 9.341731)]
# cities=list(np.copy(odyssey))

# loc_names=['Ithaca', 'Troy', 'Cicones', 'Lotus Eaters', 'Cyclopes', 'King Aiolos', 'Laestryonians', 'Circe', 'Underworld', 'Sirens', 'Scylla and Charybdis', 'Helios', 'Calypso', 'Phaeacians','Ithaca']

# cities.append(cities[0])

# '''
#cities1.append(cities1[0])


# f3=plt.figure()
# ax3=f3.add_subplot(111)
# ax3.plot(cityx,cityy)
# for i in range(len(cityx)-1):
#     ax3.scatter(cityx[i],cityy[i])#,label=loc_names[i])
# ax3.set_title("Brute Force Solution")


# def cityMove(cities,temp, names=None, toll=False):
#     init_distance=pathDistance(cities, toll)
#     start=np.random.randint(1,len(cities)-2)
#     end=np.random.randint(start+1,len(cities))
#     new_cities=np.copy(cities)
#     new_cities[start:end]=new_cities[start:end][::-1]
#     final_distance=pathDistance(new_cities, toll)
#     if final_distance<init_distance:
#         if names!=None:
#             names[start:end]=names[start:end][::-1]
#         return new_cities
        
#     else:
#         if anneal((final_distance-init_distance), temp):
#             if names!=None:
#                 names[start:end]=names[start:end][::-1]
#             return new_cities
            
#         else:
#             return cities

# brute_times=[]
# anneal_times=[]
# anneal_error=[]
# cities_list=[]
# for i in range(4,len(cities1)+1):
#     cities_list.append(i)

#     cities=cities1[0:i]
    
#     brute_start=time.time()
#     brute_poss=[]
#     comb=permutations(cities)
#     for i in comb:
#         if i[0]==(0,0):# and i[-1]==(0,0):
#             brute_poss.append(i)
    
#     min_distance=1e9
#     min_loc=0
#     for i in range(len(brute_poss)):
#         dist=pathDistance(brute_poss[i])
#         if dist<min_distance:
#             min_distance=dist
#             min_loc=i
    
#     print(brute_poss[min_loc], min_distance)
    
#     optimal_brute=brute_poss[min_loc]
    
#     cityx,cityy= zip(*optimal_brute)
#     brute_end=time.time()
    
#     anneal_start=time.time()
#     T=100
#     N=1000*len(cities)
#     temps=np.linspace(T,0,11)
    
    
#     for temp in temps:
#         j=0
#         print(temp)
#         while j<N:       
#             cities=cityMove(cities,temp)#,names=loc_names,toll=False)
#             j+=1
    
#     anneal_end=time.time()
#     brute_times.append(brute_end-brute_start)
#     anneal_times.append(anneal_end-anneal_start)
#     anneal_error.append((abs(pathDistance(optimal_brute)-pathDistance(cities))/pathDistance(optimal_brute)))
    

# f6=plt.figure()
# ax6=f6.add_subplot(111)
# ax6.plot(cities_list, brute_times, label='Brute Force')
# ax6.plot(cities_list, anneal_times, label='Annealing')
# ax6.set_xlabel('Number of Cities')
# ax6.set_ylabel('Time To Compute')
# ax6.set_title("Time Comparison")
# ax6.legend()

# f7=plt.figure()
# ax7=f7.add_subplot(111)
# ax7.plot(cities_list, anneal_error, label='Error')
# ax7.set_xlabel('Number of Cities')
# ax7.set_ylabel('Annealing Error')
# ax7.set_title('Percent Error of Annealing Method')



'''
#cityx,cityy= zip(*cities)
cityy,cityx=zip(*cities)
# origy,origx=zip(*odyssey)
# img=plt.imread('/Users/ryanswope/PythonDocs/Modeling6/mapim.jpg')


f4=plt.figure()
ax4=f4.add_subplot(111)
# ax4.imshow(img, extent=[-1.21493, 28.40522, 33.324, 44.836])
#ax4.plot(cityx,cityy,label='Original Odyssey')
for i in range(len(cityx)-1):
    ax4.scatter(cityx[i],cityy[i])#,label=loc_names[i])
ax4.plot(cityx,cityy,label='Optimized Path')
ax4.set_title("Annealed  Path")
#ax4.legend(loc=9,bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=7)


#'Geneva',
#,(46.200321, 6.150134,1)

hann_names=['Montpellier','Geneva', 'Zurich', 'Munich', 'Bologna', 'Milan', 'Genoa', 'Nice', 'Florence', 'Venice']
conquest=[(43.602703, 3.879129,1),(46.200321, 6.150134,1),(47.385162, 8.528355,1),(48.111937, 11.575159,1),(44.485873, 11.372535,-1),(45.441901, 9.190987,-1),(44.415372, 8.937535,-1),(43.706396, 7.259681,-1),(43.770618, 11.254277,-1),(45.433877, 12.321371,-1)]

T=100
N=1000*len(conquest)
temps=np.linspace(T,0,11)


for temp in temps:
    j=0
    print(temp)
    while j<N:       
        conquest=cityMove(conquest,temp,names=hann_names,toll=True)
        j+=1
        
conquesty,conquestx,tolls=zip(*conquest)
img2=plt.imread('/Users/ryanswope/PythonDocs/Modeling6/hannibal.jpg')
f5=plt.figure()
ax5=f5.add_subplot(111)
ax5.imshow(img2, extent=[1.9458, 16.50499, 43.236, 48.3846])
for i in range(len(conquestx)):
    ax5.scatter(conquestx[i],conquesty[i],s=100, label=hann_names[i])
ax5.plot(conquestx,conquesty,color='firebrick',label='Hannibal\'s Route')
ax5.set_title('Hannibal\'s Conquest In Northern Italy With Lots of Dead Elephants')
ax5.legend(loc=9,bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=6)
'''

######### Portfolio Management ###########
stocks=[['TSLA', .2823, 250.],['NFLX',.1268,250.],['AAPL',0.3034,250.],['AMZN',0.1081,250.]]

def cost(stocks, risk):
    names,risks,vals=zip(*stocks)
    vals=list(vals)
    for i in range(len(vals)):
        vals[i]=float(vals[i])
    expret=0
    exprisk=0
    for i in range(len(names)):
        expret+=float(risks[i])*float(vals[i])
    exprisk=expret/1000#sum(vals)
    
    if exprisk>risk:
        return expret-500
    else:
        return expret

def makeMove(stocks, risk,temp):
    current_val=cost(stocks, risk)
    loc1=np.random.randint(0,len(stocks))
    loc2=loc1
    while loc2==loc1:
        loc2=np.random.randint(0,len(stocks))
    stock1=list(np.copy(stocks[loc1]))
    stock2=list(np.copy(stocks[loc2]))
    amount=float(stock1[2])*(np.random.rand()/10)
    #print(stock1, stock2, amount)
    stock1[2]=float(stock1[2])-amount
    stock2[2]=float(stock2[2])+amount+.009992
    #print(stock1, stock2)
    new_stocks=np.copy(stocks)
    new_stocks[loc1]=stock1
    new_stocks[loc2]=stock2
    if cost(new_stocks,risk)>current_val:
        return new_stocks
    else:
        if anneal((current_val-cost(new_stocks,risk)),temp):
            return new_stocks
        else:
            return stocks
   
T=100
N=1000*len(stocks)
temps=np.linspace(T,0,11)


for temp in temps:
    j=0
    print(temp)
    while j<N:       
        stocks=makeMove(stocks,.2,temp)#,names=loc_names,toll=False)
        j+=1
        names,risks,vals=zip(*stocks)
        vals=list(vals)
        for i in range(len(vals)):
            vals[i]=float(vals[i])
        print(sum(vals))
        
print(stocks)

