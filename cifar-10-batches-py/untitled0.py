# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 09:36:52 2018

@author: hzhang
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,1,200)

y = np.zeros_like(x,dtype = np.int32)

x[0:100] = np.sin(4*np.pi*x)[0:100]

x[100:200] = np.cos(4*np.pi*x)[100:200]
y = 4*np.linspace(0,1,200)+0.3*np.random.randn(200)
label= np.ones_like(x)

label[0:100]=0

plt.scatter(x,y,c = label)