#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 01:51:39 2020

@author: ryanswope
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cpu_loss=pd.read_csv('/Users/ryanswope/PythonDocs/Modeling_Final/cpu_loss.csv')
gpu_myloss=pd.read_csv('/Users/ryanswope/PythonDocs/Modeling_Final/gpu_myloss.csv')
gpu_other=pd.read_csv('/Users/ryanswope/PythonDocs/Modeling_Final/gpu_other.csv')
loss_explosion=pd.read_csv('/Users/ryanswope/PythonDocs/Modeling_Final/loss_explosion.csv')

f1=plt.figure()
ax1=f1.add_subplot(111)
ax1.plot(loss_explosion['Step'], loss_explosion['Value'])
ax1.set_xlabel('Step')
ax1.set_ylabel('Value')
ax1.set_title('Loss Explosion Due to K')

f2=plt.figure()
ax2=f2.add_subplot(111)
ax2.plot(cpu_loss['Step'], cpu_loss['Value'], label='CPU on my set')
ax2.plot(gpu_myloss['Step'], gpu_myloss['Value'], label='GPU on my set')
ax2.plot(gpu_other['Step'], gpu_other['Value'], label='Model from other Set')
ax2.set_xlabel('Step')
ax2.set_ylabel('Value')
ax2.set_title('Model Architecture Comparison')
ax2.legend()

