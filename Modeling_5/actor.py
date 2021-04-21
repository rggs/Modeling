#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:00:47 2020

@author: ryanswope
"""

import numpy as np
import matplotlib.pyplot as plt

class World:
    def __init__(self, size):
        self.size=size
        self.grid=np.zeros((self.size,self.size))
    
    def place(self, x,y,state):
        self.grid[x,y]=state
            
    def findSpot(self, state):
        placed=False
        if self.grid.all()!=0:
            print("Cannot place this character; grid spot full")
            pass
        
        else:
            while not placed:
                x = np.random.randint(0, self.size)
                y = np.random.randint(0,self.size)
                if self.grid[x,y]==0:
                    self.grid[x,y]=state
                    placed=True
            return x,y
            
            
            
    def canMove(self, x,y):
        can=True
        if x<0 or x>self.size-1 or y<0 or y>self.size-1 or self.grid[x,y]!=0:
            can=False
        
        return can
        
    def Neighbors(self, x, y):
        if x==0:
            xrange=[0,x+1]
        elif x==self.size-1:
            xrange=[x-1,self.size-1]
        else:
            xrange=[x-1,x,x+1]
        if y==0:
            yrange=[0,y+1]
        elif y==self.size-1:
            yrange=[y-1,self.size-1]
        else:
            yrange=[y-1,y,y+1]
            
        neighbor_locs=[]
        for i in xrange:
            for j in yrange:
                if i==x and j==y:
                    pass
                else:
                    if self.grid[i,j]!=0:
                        neighbor_locs.append([i,j])
        return neighbor_locs
    
    def hasNeighbors(self,x,y):
        if self.Neighbors(x,y)!=[]:
            return True
        else: return False
                
        
    #def iterate(self):
        

class Actor():
    def __init__(self, InWorld, state, xpos, ypos):
        self.state = state
        self.xpos=xpos
        self.ypos=ypos
        self.InWorld=InWorld
        self.InWorld.place(self.xpos,self.ypos,self.state)
        
    def getSick(self):
        if self.InWorld.hasNeighbors(self.xpos, self.ypos):
            for j in self.InWorld.Neighbors(self.xpos, self.ypos):
                if self.InWorld.grid[j[0],j[1]]==2:
                    val = np.random.randint(1,11)
                    if val >=6:
                        self.state = 2
                        self.InWorld.place(self.xpos,self.ypos,self.state)
        
    def move(self):
            
        if len(self.InWorld.Neighbors(self.xpos, self.ypos))==4:
            pass
        else:
 	
            val = np.random.randint(1, 5) 
 	
            if val == 1 and self.InWorld.canMove(self.xpos+1,self.ypos):
                self.InWorld.place(self.xpos,self.ypos,0)
                self.xpos+=1
                self.InWorld.place(self.xpos,self.ypos,self.state)
                      	
            elif val == 2 and self.InWorld.canMove(self.xpos-1,self.ypos): 
                self.InWorld.place(self.xpos,self.ypos,0)
                self.xpos=self.xpos-1
                self.InWorld.place(self.xpos,self.ypos,self.state)
                        
            elif val == 3 and self.InWorld.canMove(self.xpos,self.ypos+1): 
                self.InWorld.place(self.xpos,self.ypos,0)
                self.ypos+= 1
                self.InWorld.place(self.xpos,self.ypos,self.state)
                        
            elif val == 4 and self.InWorld.canMove(self.xpos,self.ypos-1):
                self.InWorld.place(self.xpos,self.ypos,0)
                self.ypos=self.ypos-1
                self.InWorld.place(self.xpos,self.ypos,self.state)
                
    
            else: pass
        
    def moveTo(self, x, y):
        if self.InWorld.canMove(x,y):
            self.InWorld.place(self.xpos,self.ypos,0)
            self.xpos=x
            self.ypos=y
            self.InWorld.place(x,y,self.state)
        