#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 21:52:47 2020

@author: ryanswope
"""

import numpy as np
import scipy
from scipy.optimize import linprog

foodsieat=[1032, 20120, 43217, 43261, 1123,1077,1202,1236,4053,5112,5220,7972,9003,9037,9236,9252,9279,11090,
           11124,11457,12087,13000,15083,16098,19353,27027, 18064, 14278, 5306, 9040]

cols_touse=[0, 2, 7, 4, 5, 3, 21, 22, 18]

file = '/Users/ryanswope/PythonDocs/nutrition/usda_2016_food_facts.txt'
data = np.loadtxt(file, dtype=str, delimiter='\t', usecols=cols_touse, skiprows=1)
data=list(data)

file2='/Users/ryanswope/PythonDocs/nutrition/revised.txt'
data2 = np.loadtxt(file2, dtype=str, delimiter='\t', skiprows=1)
data2=list(data2)
prices=[]
for p in data2:
    prices.append(float(p[-1])/100)

def getNuts(cals, fats, carbs, prot, ca, fe, amount_arr):
    nut_arr=np.array([cals, fats, carbs, prot, ca, fe])
    res=np.dot(nut_arr, amount_arr)
    return {'Calories': res[0], 'Fats (g):': res[1], 'Carbs (g)':res[2], "Proteins (g)": res[3], "Calcium (g)": res[4], 'Iron (g)': res[5]}



headers=data.pop(0)

foods=[]

for i in data:
    if int(i[0]) in foodsieat:
        foods.append(i)
        
names=[]
cals=[]
fats=[]
carbs=[]
proteins=[]
ca=[]
fe=[]
weights=[]





for i in foods:
    #Divide by 100 to account for serving size in excel file of 100g
    names.append(i[1])
    cals.append(float(i[2])/100)
    fats.append(float(i[3])/100)
    carbs.append(float(i[4])/100)
    proteins.append(float(i[5])/100)
    ca.append(float(i[6])/100)
    fe.append(float(i[7])/100)
    weights.append(1)
    
ca=list(np.array(ca)/1000)
fe=list(np.array(fe)/1000)
#divide by 1000 to convert from mg to g



############# Part 1 ##################
#We are maximizing these values, so everything will be multiplied by -1
    
A_ub=np.array([fats, carbs, proteins, ca, fe, weights])
A_ub=-1*A_ub
A_ub[-1]=-1*A_ub[-1] #weights need to be positive
b_ub=np.array([-70, -310, -50, -1, -18/1000, 2000])
c=cals


res=linprog(c, A_ub, b_ub)

amounts=res.x
print("Number of calories: "+ str(res.fun))
print('Total Food Weight: '+str(sum(amounts)))
print("Amount of each food (g): ")
for a in range(len(amounts)):
    print(names[a] + ' ' + str(amounts[a]))

print("\nAmount of each food with more than 1g (g): ")
for a in range(len(amounts)):
    if amounts[a]>=1: print(names[a].strip('"') + '&' + str(round(amounts[a], 2))+'\\\\')
    
############ Part 2 ###########
    
# A_ub=np.array([cals, fats, carbs, proteins, ca, fe, weights])
# A_ub=-1*A_ub
# A_ub[-1]=-1*A_ub[-1] #weights need to be positive
# b_ub=np.array([-2000, -310, -50, -1, -18/1000, 2000])
# c=fats


# res=linprog(c, A_ub, b_ub)

# amounts=res.x
# print("Amount of Fat (g): "+ str(res.fun))
# print('Total Food Weight: '+str(sum(amounts)))
# print("Amount of each food (g): ")
# for a in range(len(amounts)):
#     print(names[a] + ' ' + str(amounts[a]))

# print("\nAmount of each food with more than 1g (g): ")
# for a in range(len(amounts)):
#     if amounts[a]>=1: print(names[a].strip('"') + '&' + str(round(amounts[a], 2))+'\\\\')
    
    
############# Part 3 ###############
    
# A_ub=np.array([cals, fats, carbs, proteins, ca, fe, weights])
# A_ub=-1*A_ub
# A_ub[-1]=-1*A_ub[-1] #weights need to be positive
# b_ub=np.array([-2500, -70, -310, -50, -1, -18/1000, 3000])
# c=cals
# bounds=(0, 600)


# res=linprog(c, A_ub, b_ub, bounds=bounds)

# amounts=res.x
# print("Number of calories: "+ str(res.fun))
# print('Total Food Weight: '+str(sum(amounts)))
# print("Amount of each food (g): ")
# for a in range(len(amounts)):
#     print(names[a] + ' ' + str(amounts[a]))

# print("\nAmount of each food with more than 1g (g): ")
# for a in range(len(amounts)):
#     if amounts[a]>=1: print(names[a].strip('"') + '&' + str(round(amounts[a], 2))+'\\\\')


    
    
############## Part 4 ##############
    
# A_ub=np.array([cals, fats, carbs, proteins, ca, fe, weights])
# A_ub=-1*A_ub
# A_ub[-1]=-1*A_ub[-1] #weights need to be positive
# b_ub=np.array([-2500, -70, -310, -50, -1, -18/1000, 3000])
# c=prices
# bounds=(0, 800)


# res=linprog(c, A_ub, b_ub, bounds=bounds)

# amounts=res.x
# print("Price for one day: "+ str(res.fun))
# print('Total Food Weight: '+str(sum(amounts)))
# print("Amount of each food (g): ")
# for a in range(len(amounts)):
#     print(names[a] + ' ' + str(amounts[a]))

# print("\nAmount of each food with more than 1g (g): ")
# for a in range(len(amounts)):
#     if amounts[a]>=1: print(names[a].strip('"') + '&' + str(round(amounts[a], 2))+'\\\\')


############## Part 5 #############
# file = '/Users/ryanswope/PythonDocs/nutrition/usda_2016_food_facts.txt'
# data = np.loadtxt(file, dtype=str, delimiter='\t', usecols=cols_touse, skiprows=1)
# data=list(data)

# headers=data.pop(0)

# foods=data
        
# names=[]
# cals=[]
# fats=[]
# carbs=[]
# proteins=[]
# ca=[]
# fe=[]
# weights=[]
# sugars=[]




# for i in foods:
#     #Divide by 100 to account for serving size in excel file of 100g
#     if i[6]=='NULL':
#         i[6]=0
#     if i[7]=='NULL':
#         i[7]=0
#     if i[8]=='NULL':
#         i[8]=0
    
#     names.append(i[1])
#     cals.append(float(i[2])/100)
#     fats.append(float(i[3])/100)
#     carbs.append(float(i[4])/100)
#     proteins.append(float(i[5])/100)
#     ca.append(float(i[6])/100)
#     fe.append(float(i[7])/100)
#     sugars.append(float(i[8])/100)
#     weights.append(1)
    
# ca=list(np.array(ca)/1000)
# fe=list(np.array(fe)/1000)


# A_ub=np.array([cals, fats, carbs, proteins, ca, fe, sugars,fats,  weights])
# A_ub=-1*A_ub
# A_ub[-2:]=-1*A_ub[-2:] #weights need to be positive
# #A_ub[-1]=-1*A_ub[-1] 
# b_ub=np.array([-2800, -70, -310, -200, -1, -18/1000, -37.5, 150, 3000])
# c=sugars
# bounds=(0, 600)


# res=linprog(c, A_ub, b_ub, bounds=bounds, method='revised simplex')

# amounts=res.x
# print("Carbs: "+ str(res.fun))
# print('Total Food Weight: '+str(sum(amounts)))
# print("Amount of each food (g): ")
# for a in range(len(amounts)):
#     print(names[a] + ' ' + str(amounts[a]))

# print("\nAmount of each food with more than 1g (g): ")
# for a in range(len(amounts)):
#     if amounts[a]>=1: print(names[a].strip('"') + '&' + str(round(amounts[a], 2))+'\\\\')
    

from math import *
from scipy.stats import norm
from wallstreet import Stock, Call, Put

# Input Parameters

def BSM_call(sigma, T, S, r, K):

    # Calculations for the solution to BSM equation
    dplus = (1/(sigma*sqrt(T)))*((log(S/K))+(r+(sigma**2)/2)*T)
    dminus = (1/(sigma*sqrt(T)))*((log(S/K))+(r-(sigma**2)/2)*T)
    
    # Calculating price of Call and Put
    Call = S*norm.cdf(dplus) - K*exp(-r*T)*norm.cdf(dminus)
    Put = K*exp(-r*T)*norm.cdf(-dminus)-S*norm.cdf(-dplus)
    
    # # Printing the values of call an put options
    # print("The Price of the Call option is %s" % round(Call, 2))
    # print("The Price of the Put option is %s" % round(Put,  2))
    return Call

def BSM_put(sigma, T, S, r, K):

    # Calculations for the solution to BSM equation
    dplus = (1/(sigma*sqrt(T)))*((log(S/K))+(r+(sigma**2)/2)*T)
    dminus = (1/(sigma*sqrt(T)))*((log(S/K))+(r-(sigma**2)/2)*T)
    
    # Calculating price of Call and Put
    Call = S*norm.cdf(dplus) - K*exp(-r*T)*norm.cdf(dminus)
    Put = K*exp(-r*T)*norm.cdf(-dminus)-S*norm.cdf(-dplus)
    
    # # Printing the values of call an put options
    # print("The Price of the Call option is %s" % round(Call, 2))
    # print("The Price of the Put option is %s" % round(Put,  2))
    return Put


import scipy.stats as si
import sympy as sy
from sympy.stats import Normal, cdf
from sympy import init_printing

tsla_c=Call('TSLA', d=27, m=3, y=2020)

strikes=tsla_c.strikes
obs=[]
for i in strikes:
    obs.append(Call('TSLA', d=27, m=3, y=2020, strike=i))
call_vals=[]
put_vals=[]
for tsla_c in obs:
    #tsla_c.set_strike(i)
    p=BSM_call(tsla_c.implied_volatility(), 8/365, tsla_c.underlying.price, .0118, i)
    call_vals.append([tsla_c.strike, tsla_c.price, p, tsla_c.delta(), tsla_c.gamma(), tsla_c.theta(), tsla_c.rho(), tsla_c.vega()])
    print('Success')
    
calls=np.array(call_vals)
pmark=calls[:,1]
pther=calls[:,2]
delt=calls[:,3]
gam=calls[:,4]
thet=calls[:,5]
rho=calls[:,6]
vega=calls[:,7]

buy=pther-pmark
sell=pmark=pther

bs=[]
ss=[]

for k in range(len(pmark)):
    bs.append(str(k))
    ss.append(str(k))
from pulp import *
b=(LpVariable(bs[i], lowBound=0, upBound=1) for i in range(len(bs)))
s=LpVariable.dicts('s',(str(i) for i in range(len(pmark))), lowBound=0, upBound=1, cat=LpContinuous)
prob=LpProblem('calls_prob',LpMaximize)

# for j in range(len(pmark)):
prob+= np.dot(buy,b) - np.dot(sell,s)
prob+=np.dot(b,delt)-np.dot(s,delt)+b+s==0
prob+=np.dot(b,gam)-np.dot(s,gam)==0
prob+=np.dot(b,thet)-np.dot(s,thet)==0
prob+=np.dot(b,rho)-np.dot(s,rho)==0
prob+=np.dot(b,vega)-np.dot(s,vega)==0
prob.solve()

# tsla_p=Put('TSLA', d=27, m=3, y=2020)
# strikes=tsla_p.strikes
# obs=[]
# for i in strikes:
#     obs.append(Put('TSLA', d=27, m=3, y=2020, strike=i))
# for tsla_p in obs:
#     #tsla_p.set_strike(i)
#     p=BSM_put(tsla_p.implied_volatility(), 8/365, tsla_p.underlying.price, .0118, i)
#     put_vals.append([tsla_p.strike, tsla_p.price, p, tsla_p.delta(), tsla_p.gamma(), tsla_p.theta(), tsla_p.rho(), tsla_p.vega()])


