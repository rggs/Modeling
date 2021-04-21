#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 22:36:49 2020

@author: ryanswope
"""

import pygame

pygame.init()
# load and set the logo
#logo = pygame.image.load("logo32x32.png")
#pygame.display.set_icon(logo)
pygame.display.set_caption("Foxes and Rabbits")
fox = pygame.image.load("/Users/ryanswope/PythonDocs/FoxGame/fox.png")
rabbit = pygame.image.load("/Users/ryanswope/PythonDocs/FoxGame/rabbit.png")



# create a surface on screen that has the size of 240 x 180
screen = pygame.display.set_mode((720,720))
 
# define a variable to control the main loop
running = True


 
# main loop
while running:
    # event handling, gets all event from the event queue
    for event in pygame.event.get():
        # only do something if the event is of type QUIT
        if event.type == pygame.QUIT:
            # change the value to False, to exit the main loop
            running = False
            
    screen.fill((255,255,255))
    screen.blit(fox,(360,360))
    screen.blit(rabbit,(360, 540))
    pygame.display.flip()
        
            



pygame.display.quit()
pygame.quit()