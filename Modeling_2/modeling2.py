#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 15:37:23 2020

@author: ryanswope
"""

import numpy as np
import scipy as scipy
import scipy.optimize as optimize
import matplotlib.pyplot as plt

from scipy import integrate
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

P=1
w=2*np.pi*P
G=6.67e-11*((9.95839577e-14)**(-2))*2.98692e-34
Msun=1.989e30
Mjup=1.898e27
#q=Mjup/Msun
q=10
a=1.0
'''
def potential(x,y,z, positions):
    x1,y1,x2,y2= positions
    r_perp=(x**2 + y**2)/(x**2 + y**2 + z**2)
    s1=((x-x1)**2 + ((y-y1)**2) + z**2)**(1/2)
    s2=((x-x2)**2 + ((y-y2)**2) + z**2)**(1/2)
    
    pot= -(G*Msun)/s1 + -(G*Msun)/s2 + -(1/2)*(w**2)*(r_perp**2)
    
    return pot
'''
y0=[0,0,a,0]
cmx1=-(q*a)/(1+q)
cmx2=a+cmx1
y0_cm=[y0[0]+cmx1,0,cmx2,0]

def potential(x,y,z, positions):
    x1,y1,x2,y2= positions
    r=np.sqrt(x**2 + y**2 + z**2)
    r_perp=(x**2 + y**2)**(1/2)
    s1=((x-x1)**2 + ((y-y1)**2) + z**2)**(1/2)
    s2=((x-x2)**2 + ((y-y2)**2) + z**2)**(1/2)
    
    #pot= (1/r)+q*((1/np.sqrt(1-(2*x)+r**2))-x)+((q+1)/2)*r_perp
    #pot=(2/((1+q)*s1))+((2*q)/((1+q)*s2))+(x-(q/(q+1)))**2 + y**2
    pot=(2/((1+q)*s1))+((2*q)/((1+q)*s2))+r_perp**2
    
    #Find Del(Phi) by hand and then solve for 0 to find potential points
    #Redo dimensionless eq: https://courses.lumenlearning.com/boundless-physics/chapter/velocity-acceleration-and-force/
    #m*a*w^2=Force of gravity
    #Add center of mass so we end up with x=(x-xc), y =(y-yc)
    return -pot

def gravforce(x,y,z, positions):
    x1,y1,x2,y2= positions
    s1=((x-x1)**2 + ((y-y1)**2) + z**2)**(1/2)
    s2=((x-x2)**2 + ((y-y2)**2) + z**2)**(1/2)
    
    delx=(-2/(1+q))*((x-x1)/s1**3) + ((-2*q)/(1+q))*((x-x2)/s2**3) + 2*x    #(x-(q/(1+q)))
    dely=(-2/(1+q))*((y-y1)/s1**3) + ((-2*q)/(1+q))*((y-y2)/s2**3) + 2*y
    delz=(-2/(1+q))*((z)/s1**3) + ((-2*q)/(1+q))*((z)/s2**3)
    
    mag=np.sqrt(delx**2 + dely**2 +delz**2)
    return -mag

def gravforce1(x,y,z, positions):
    x1,y1,x2,y2= positions
    s1=((x-x1)**2 + ((y-y1)**2) + z**2)**(1/2)
    s2=((x-x2)**2 + ((y-y2)**2) + z**2)**(1/2)
    
    delx=(-2/(1+q))*((x-x1)/s1**3) + ((-2*q)/(1+q))*((x-x2)/s2**3)    #(x-(q/(1+q)))
    dely=(-2/(1+q))*((y-y1)/s1**3) + ((-2*q)/(1+q))*((y-y2)/s2**3)
    delz=(-2/(1+q))*((z)/s1**3) + ((-2*q)/(1+q))*((z)/s2**3)
    
    mag=np.sqrt(delx**2 + dely**2 +delz**2)
    return -mag
    
P1=[(-q-1),a*(2*q+3),-(a**2)*(q+3),q*(a**3),-2*(a**4)*q,(a**5)*q]
P2=[(q+1),a*(2*q+3),(a**2)*(q+3),-q*(a**3),-2*(a**4)*q,-(a**5)*q]
#P3=[(q+1),-8*a*(q+(7/8)),(a**2)*(25*q+19),(a**3)*(-37*q -24), (a**4)*(26*q + 12),-7*(a**5)*q]
P3=[(q+1),(3+2*q),3+q,1+1+q, 2*q,q]

n=1/((1/q)+1)
#L1=[a-a*((n/3)**(1/3))-cmx1,0]
L1=[cmx2-np.roots(P1)[4],0]
L2=[cmx2+np.roots(P2)[2],0]
L3=[cmx2+np.roots(P3)[0],0]
L4=[a*((1/2)*((1-q)/(1+q))), a*np.sqrt(3)/2]
L5=[a*((1/2)*((1-q)/(1+q))), -a*np.sqrt(3)/2]
LX=[L1[0],L2[0],L3[0],L4[0],L5[0]]
LY=[L1[1],L2[1],L3[1],L4[1],L5[1]]

    
y0=[0,0,1,0]
x=np.linspace(-8,8,1000)
y=x

phi_1d=potential(x,0, 0,y0_cm)

X,Y=np.meshgrid(x,y)
phi_2d=potential(X,Y, 0,y0_cm)
grav_2d=gravforce(X,Y,0,y0_cm)


#2-D Potential
f2=plt.figure()
ax2 = plt.axes(projection='3d')
ax2.plot_surface(X,Y,phi_2d)
ax2.set_title('Potential in the XY Plane')
ax2.set_xlabel('X Position')
ax2.set_ylabel('Y Position')
ax2.set_zlabel('Potential')

#2-D Force
f21=plt.figure()
ax21 = plt.axes(projection='3d')
ax21.plot_surface(X,Y,grav_2d)
ax21.set_title('Force in the XY Plane')
ax21.set_xlabel('X Position')
ax21.set_ylabel('Y Position')
ax21.set_zlabel('Potential')

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

levels=np.linspace(-5,0,25)
f1=plt.figure()
ax1=f1.add_subplot(111)
ax1.contour(X,Y, phi_2d,levels=levels,colors='black',linewidths=.5,vmin=-3)
ax1.scatter(LX, LY,c='red',marker='+')
ax1.set_title("Equipotential Lines showing Lagrange Points")
ax1.set_xlabel('X position')
ax1.set_ylabel('Y Position')
ax1.set_xlim(-3,3)
ax1.set_ylim(-3,3)

#1D Potential
f4=plt.figure()
ax4=f4.add_subplot(111)
ax4.plot(x,phi_1d)
ax4.set_title('1D Plot of Potential')
ax4.set_xlabel('X Position')
ax4.set_ylabel('Potential')

#1D Force
grav_1d=gravforce(x,0,0,y0_cm)
grav_1d2=gravforce1(x,0,0,y0_cm)
f6=plt.figure()
ax6=f6.add_subplot(111)
ax6.plot(x,grav_1d,label='Effective Force')
ax6.plot(x,grav_1d2, label='Gravitational Force')
ax6.set_title('1D Plot of Force')
ax6.set_xlabel('X Position')
ax6.set_ylabel('Force')
ax6.legend()



'''
z=np.linspace(-1,1,51)
j=10
for i in z:
    phi_3d=potential(X,Y, i,y0_cm)
    grav_3d=gravforce(X,Y,i,y0_cm)
    
    
    f5=plt.figure()
    ax5=f5.gca(projection='3d')
    ax5.plot_surface(X,Y,phi_3d)
    ax5.set_xlabel('X Position')
    ax5.set_ylabel('Y Position')
    ax5.set_zlabel('Gravitational Potential')
    ax5.set_title('3D Potential Representation at Z='+str(round(i,2)))
    #ax5.set_zlim(-35,-5)
    filename='/Users/ryanswope/PythonDocs/Potential_Gif/potstep'+str(j)+'.png'
    plt.savefig(filename,dpi=300)
    plt.gca()
    
    f7=plt.figure()
    ax7=f7.gca(projection='3d')
    ax7.plot_surface(X,Y,grav_3d)
    ax7.set_xlabel('X Position')
    ax7.set_ylabel('Y Position')
    ax7.set_zlabel('Gravitational Force')
    ax7.set_title('3D Potential Representation at Z='+str(round(i,2)))
    #ax7.set_zlim(-35,-5)
    filename='/Users/ryanswope/PythonDocs/Force_Gif/forcestep'+str(j)+'.png'
    plt.savefig(filename,dpi=300)
    plt.gca()
    j+=1


    
#Stability

z=np.linspace(-.8,.8,81)



j=10
for i in z:
    f11=plt.figure()
    ax11=f11.add_subplot(111)
    phi_2d=potential(X,Y,i,y0_cm)
    ax11.contour(X,Y, phi_2d,levels=[-4],colors='black',linewidths=.5,vmin=-3)
    ax11.set_title("Roche Lobes, z="+str(round(i,4)))
    ax11.set_xlabel('X position')
    ax11.set_ylabel('Y Position')
    ax11.set_xlim(-3,3)
    ax11.set_ylim(-3,3)
    filename='/Users/ryanswope/PythonDocs/Lobe_Gif/lobestep'+str(j)+'.png'
    plt.savefig(filename,dpi=300)
    plt.gca()
    j+=1


j=1
while j<40:
    q=q/j
    L4=[a*((1/2)*((1-q)/(1+q))), a*np.sqrt(3)/2]
    x=np.linspace(L4[0]-.3,L4[0]+.3,100)
    y=np.linspace(L4[1]-.3,L4[1]+.3,100)
    
    X, Y=np.meshgrid(x,y)
    grav_2d=gravforce(X,Y,0,y0_cm)
    phi_2d=potential(X,Y,0,y0_cm)
    f11=plt.figure()
    ax11=f11.add_subplot(111)
    ax11.contour(X,Y, phi_2d,color='cp')

    ax11.set_title("Equipotential Lines showing Lagrange Points")
    ax11.set_xlabel('X position')
    ax11.set_ylabel('Y Position')

    
    phi_3d=potential(X,Y,0,y0_cm)
    yline=potential(L4[0],y,0,y0_cm)
    xline=potential(x,L4[1],0,y0_cm)
    L4=[a*((1/2)*((1-q)/(1+q))), a*np.sqrt(3)/2]
    
    f5=plt.figure()
    ax5=f5.gca(projection='3d')
    ax5.plot_surface(X,Y,phi_3d)
    ax5.set_xlabel('X Position')
    ax5.set_ylabel('Y Position')
    ax5.set_zlabel('Gravitational Potential')
    
    f8=plt.figure()
    ax8=f8.add_subplot(211)
    ax9=f8.add_subplot(212)
    ax8.plot(y,yline, label='Potential Along Y Axis')
    ax9.plot(x,xline, label='Potential Along X Axis')
    ax8.axvline(L4[1])
    ax9.axvline(L4[0])
    ax8.legend()
    ax8.set_xlabel('Distance')
    ax9.set_ylabel('Distance')
    ax9.legend()
    filename='/Users/ryanswope/PythonDocs/Force_Gif/forcestep'+str(j)+'.png'
    
    j+=1
'''




