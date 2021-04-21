#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:38:09 2019

@author: ryanswope
Name:
File Name:
Due Date:
Comments:
"""

import mnist_loader
import numpy as np
import matplotlib.pyplot as plt

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import Number_Network as network

hidden=50
eta=3.0
epochs=30
mini_batch=10

net=network.Network([784,hidden,10])

learning_rate=net.SGD(training_data, epochs, mini_batch, eta, test_data=test_data)


import PIL


def recognize(filename, known_value):
    print(str(known_value)+":")
    
    im=PIL.Image.open(filename)
    im=im.convert(mode="L")
    im=np.array(im)
    im_input=np.reshape(im,(784,1))
    
    im_input=1-(im_input/255.0)
    
    trial1=net.feedforward(im_input)
    
    for i in range (0,len(trial1[:,0])):
        print(str(i)+": "+str(np.sum(trial1[i,:])))
        
    print("I think this is a: "+str(np.where(trial1[:,0]==trial1[:,0].max())[0]))
    
    
f1=plt.figure()
ax1=f1.add_subplot(111)
ax1.plot(learning_rate, label='Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Percent Correctly Identified')
ax1.set_title('Learning Rate, H.L. Size: '+str(hidden)+ ' Epochs: '+str(epochs)+' eta: '+str(eta))
f1.legend()
    
    

