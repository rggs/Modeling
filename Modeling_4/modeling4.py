#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:38:42 2020

@author: ryanswope
"""

import numpy as np
import scipy.optimize as optimize
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from scipy import integrate

################## Part 1 #######################

file = '/Users/ryanswope/PythonDocs/census.txt'
data = np.genfromtxt(file, delimiter='\t')

years=data[:,0]
pops=data[:,1]

norm_years=years-years[0]

def logistic_fit(t,*args):
    N,k,P0=args
    A=1-P0/N
    
    P=1/(1/A *np.e**(-k*t) -1/N)
    return P

#Exponential Fit
def exponential(t, k):
    return pops[0]*np.exp(t*k)

p1=[.1]
pop_exponential, pcov_exponential = curve_fit(exponential, norm_years, pops, p0=p1)
exponential_fit = exponential(norm_years, *pop_exponential)


p0=np.asarray([0.5,1.,pops[0]])
popfit, popcov=curve_fit(logistic_fit,norm_years,pops,p0=p0)



fit_curve=logistic_fit(norm_years, *popfit)


f1=plt.figure()
ax1=f1.add_subplot(111)
ax1.plot(norm_years,pops,label='Census Data')
ax1.plot(norm_years,fit_curve,label='Logistic Curve')
ax1.plot(norm_years,exponential_fit,label='Exponential Curve')
ax1.set_xlabel('Years Since '+str(int(years[0])))
ax1.set_ylabel('Population')
ax1.set_title('Population Curve for the U.S.')
ax1.legend()

print('N='+str(popfit[0]) + '\n' + 'k='+str(popfit[1]))

################# Part 2 ########################


# def lotka(t,y):
#     n=3/4
#     R,F=y
#     Rdot=R*(1-F)
#     Fdot=n*F*(R-1)
    
    
#     func=[Rdot, Fdot]
#     return func

# t=(0,50)
# y0=[10,1]

# LSODAsol=integrate.solve_ivp(fun=lotka,t_span=t,y0=y0, method='LSODA', max_step=.01, first_step=.01,rtol=1e-8, atol=1e-8)

# rs=LSODAsol.y[0]
# fs=LSODAsol.y[1]
# times=LSODAsol.t[:]

# f2=plt.figure()
# ax2=f2.add_subplot(111)
# ax2.plot(times,rs,label='Rabbit Population')
# ax2.plot(times,fs,label='Fox Population')
# ax2.set_ylabel('Normalized Population')
# ax2.set_xlabel('Normalized Time')
# ax2.legend()
# f2.tight_layout()

# f5=plt.figure()
# ax5=f5.add_subplot(111)
# ax5.set_ylabel('Fox Population')
# ax5.set_xlabel('Rabbit Population')
# ax5.set_title('Population Dynamics in Phase Space')

# #Iterate through possible populations
# total = np.linspace(1,10,10,endpoint=True)
# for j in total:
    
#     rabs=np.linspace(1,int(j),int(j),endpoint=True)
#     for i in range(len(rabs)):
#         y0=[rabs[i],rabs[-i]]
        
#         LSODAsol=integrate.solve_ivp(fun=lotka,t_span=t,y0=y0, method='LSODA', max_step=.01, first_step=.01,rtol=1e-8, atol=1e-8)
    
#         rs=LSODAsol.y[0]
#         fs=LSODAsol.y[1]
#         times=LSODAsol.t[:]
#         if j==1:
#             ax5.scatter(rs,fs)
#         else: ax5.plot(rs,fs)
#         f5.tight_layout()

# def lotka3(t,y):
#     a=.2
#     b=.04
#     c=.5
#     d=.045
#     e=.02
#     f=.25
#     g=.02
#     h=.04
#     i=.0185
    
#     R,F,D=y
#     RDot= a*R-b*R*F-h*D*R
#     FDot= -c*F+d*R*F-e*F*D
#     Ddot= -f*D+g*D*F+i*D*R#-d*R*F
    
#     return [RDot, FDot,Ddot]

# t=(0,200)
# y0=[50,10,5]

# three_LSODAsol=integrate.solve_ivp(fun=lotka3,t_span=t,y0=y0, method='LSODA', max_step=.01, first_step=.01,rtol=1e-8, atol=1e-8)

# three_rs=three_LSODAsol.y[0]
# three_fs=three_LSODAsol.y[1]
# three_ds=three_LSODAsol.y[2]
# three_times=three_LSODAsol.t[:]

# f3=plt.figure()
# ax3=f3.add_subplot(111)
# ax3.plot(three_times,three_rs,label='Rabbit Population')
# ax3.plot(three_times,three_fs,label='Fox Population')
# ax3.plot(three_times,three_ds,label='Dinosaur Population')
# ax3.set_ylabel('Population')
# ax3.set_xlabel('Time')
# ax3.legend()
# f3.tight_layout()


################# Part 3 #################

def disease(t,y):
    H,S,I=y
    a=.000001
    d=.0036
    
    Hdot=-a*H*S
    Sdot=a*H*S-d*S
    Idot=d*S
    
#3000 is immunity
#a=b and c=d to maintain constant population
    
    return [Hdot,Sdot,Idot]

t=(0,2000)
y0=[10000,5,0]

sick_LSODAsol=integrate.solve_ivp(fun=disease,t_span=t,y0=y0, method='LSODA', max_step=1, first_step=1,rtol=1e-6, atol=1e-6)
Hs=sick_LSODAsol.y[0]
Ss=sick_LSODAsol.y[1]
Is=sick_LSODAsol.y[2]
sick_times=sick_LSODAsol.t[:]

f4=plt.figure()
ax4=f4.add_subplot(111)
ax4.plot(sick_times,Hs,label='Healthy Population')
ax4.plot(sick_times,Ss,label='Sick Population')
ax4.plot(sick_times,Is,label='Immune (Dead) Population')
ax4.set_ylabel('Population')
ax4.set_xlabel('Days')
ax4.set_title('Epidemic Model')
ax4.legend()
f4.tight_layout()

########### Ebola ############

def ebola(t,y):
    
    SL,SH,E,I,H,R=y
    pi=1.7
    p=.2
    N=100000
    psiH=1.2
    nu=.8
    alpha=.1
    tau=.16
    mu=1/(63*365)
    thetaH=.2
    thetaI=.1
    deltaI=.1
    deltaH=.5
    lam=.344*(I+nu*H)#/N
    
    dSL=pi*(1-p)-lam*SL-mu*SL
    dSH=pi*p-psiH*lam*SH-mu*SH
    dE=lam*(SL+psiH*SH)-(alpha+mu)*E
    dI=alpha*E - (tau+thetaI+deltaI+mu)*I
    dH=tau*I-(thetaH+deltaH+mu)*H
    dR=thetaI*I+thetaH*H-mu*R
    
    return [dSL,dSH,dE,dI,dH,dR]

y0=[1000000,20000,15,10,0,0]
t=(0,60)

ebola_LSODAsol=integrate.solve_ivp(fun=ebola,t_span=t,y0=y0, method='LSODA', max_step=.01, first_step=.01,rtol=1e-6, atol=1e-6)

ebolaSL=ebola_LSODAsol.y[0]
ebolaSH=ebola_LSODAsol.y[1]
ebolaE=ebola_LSODAsol.y[2]
ebolaI=ebola_LSODAsol.y[3]
ebolaH=ebola_LSODAsol.y[4]
ebolaR=ebola_LSODAsol.y[5]

ebola_times=ebola_LSODAsol.t[:]

f7=plt.figure()
ax7=f7.add_subplot(111)
ax7.plot(ebola_times,ebolaE,label='Exposed')
ax7.plot(ebola_times,ebolaI,label='Infected')
ax7.plot(ebola_times,ebolaR,label='Recovered')
ax7.plot(ebola_times,ebolaH,label='Hospitalized')
ax7.set_xlabel('Weeks')
ax7.set_ylabel('Population')
ax7.set_title('Model of 2014 Ebola Outbreak in Sierra Leone')
ax7.legend()

# infected_tot=[20]
# for i in range(len(ebolaI)-1):
#     new_cases=ebolaI[i+1]-ebolaI[i]
#     new_cured=ebolaR[i+1]-ebolaR[i]
#     new=new_cases-new_cured
#     if new>0:
#         infected_tot.append(new)
#     else: infected_tot.append(0)
# infected_total=np.cumsum(infected_tot)
    
# f8=plt.figure()
# ax8=f8.add_subplot(111)
# ax8.plot(ebola_times,infected_total)
    
    



