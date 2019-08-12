# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 10:08:33 2018

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

plt.scatter(x,y,c=label)
plt.plot(lx,ly)

x_bound = np.linspace(0,1,200)
y_bound = np.linspace(0,1,200)
    
cov1 = np.cov(x[0:100],y[0:100])
cov2 = np.cov(x[101:200],y[101:200])
covall = np.cov(x,y)
x1_mean = x[0:100].mean()
y1_mean = y[0:100].mean()
x2_mean = x[101:200].mean()
y2_mean =y[101:200].mean()
invcovall=np.linalg.inv(covall)
invcov1 = np.linalg.inv(cov1)
invcov2 = np.linalg.inv(cov2)
detcov1 = np.linalg.det(cov1)
detcov2 = np.linalg.det(cov2)
    

def da_solver(DA_type):
    import sympy
    #Currently LDA is definitely broken. 
    #A bug was introduced when I was refactoring
    if(DA_type == 'LDA'):
        lhs_inv_cov = invcovall
        lhs_det = sympy.Matrix([0])
        rhs_inv_cov = invcovall
        rhs_det = sympy.Matrix([0])
    if (DA_type == 'QDA'):
        lhs_inv_cov = invcov1
        lhs_det = sympy.Matrix([np.log(np.sqrt(1/detcov1))])
        rhs_inv_cov = invcov2
        rhs_det = sympy.Matrix([np.log(np.sqrt(1/detcov2))])
        
    sympy.var('y0')
    sympy.var('x0')
    cept_mat = sympy.Matrix([x0-x1_mean,y0-y1_mean])
    lhs_inv_cov = sympy.Matrix(lhs_inv_cov)
    LHS = ((cept_mat.T).multiply(lhs_inv_cov)).multiply(cept_mat)
    
    cept_mat = sympy.Matrix([x0-x2_mean,y0-y2_mean])
    rhs_inv_cov = sympy.Matrix(rhs_inv_cov)
    RHS = ((cept_mat.T).multiply(rhs_inv_cov)).multiply(cept_mat)
    
    equation = (LHS-RHS)
    if(DA_type == 'QDA'):
        s1p = sympy.plot_implicit((equation[0])>0)
    if(DA_type == 'LDA'):
        solution = sympy.solve(equation[0],y0)
        sympy.plot(solution[0][y0])
    return equation

da_colver('LDA')