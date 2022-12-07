#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import scipy.linalg as la
from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[14]:


def get_cases():
    with open("allocation.csv") as cases:
        case=[]
        reader = csv.reader(cases)
        for row in reader:
            if row[0] == '1697173':
                row.remove(row[0])
                return(row)


# In[15]:


def get_parameters(no):
    with open("parameters.csv") as parametersFile:
        reader = csv.reader(parametersFile)
        parameters=[]
        for row in reader:
            parameters.append(row)
        parameters.remove(parameters[0])
    b= parameters[no]
    b.remove(b[0])
    return(b)


# In[16]:


def build_b1(n,case):
    N=n**2
    boundaries = get_parameters(case)
    a = int(boundaries[0]) #top
    b = int(boundaries[1]) #left
    c = int(boundaries[2]) #right
    d = int(boundaries[3]) #bottom
    
    b1 = np.zeros(shape=(N,1))
     
    nodeNo =0    
    for i in range(N):
        nodeNo += 1
        #edge nodes
        if 1<nodeNo<n: #top edge
            b1[i,0]=a
        if (nodeNo-1)%n == 0: #left edge
            b1[i,0]=b
        if nodeNo%n == 0: #right edge
            b1[i,0]=c
        if (n*(n-1))<nodeNo<(n**2): #bottom edge
            b1[i,0]=d
            
    nodeNo = 0
    for i in range(N):
        nodeNo+=1
        #corner nodes
        if nodeNo == 1: #top left
            b1[i,0]=a+b
        if nodeNo == n: #top right
            b1[i,0]=a+c
        if nodeNo == (n*(n-1))+1: #bottom left
            b1[i,0]=b+d
        if nodeNo == n**2: #bottom right
            b1[i,0]=c+d
        
    return(b1)


# In[27]:


def build_b2(n,case):
    N=n**2
    boundaries = get_parameters(case)
    
    b2 = np.zeros(shape=(N,1))
    
    def b_temp(x,y):
        a = int(boundaries[0]) #top
        b = int(boundaries[1]) #left
        c = int(boundaries[2]) #right
        d = int(boundaries[3]) #bottom
        #temp = (a*math.sin(b*(math.radians(x))))+(c*math.cos(d*(math.radians(y))))
        temp = (a*math.sin(b*x))+(c*math.cos(d*y))
        return(temp)
    
    nodeNo=1
    i=0
    for y in np.arange(1,0,-1/n):
        for x in np.arange(0,1,1/n):
            b2[i,0]= b_temp(x,y)
            if (not 1<nodeNo<n) and ((nodeNo-1)%n!=0) and (nodeNo%n != 0) and(not (n*(n-1))<nodeNo<(n**2)): #if not on boundary
                b2[i,0]=0
            nodeNo+=1
            i+=1
    
    return(b2)


# In[18]:


def forward_sub(L,b):
    n = len(L)
    y = np.zeros(shape=(n,1))
    for i in range(len(b)):
        y[i] =(b[i])
        for j in range(i):
            y[i]=y[i]-(L[i, j]*y[j])
        y[i] = y[i]/L[i, i]
    return(y)

def back_sub(U,y):    
    x = np.zeros(shape=(len(y),1))
    for i in range(len(x),0,-1):
        x[i-1]=(y[i-1]-np.dot(U[i-1,i:],x[i:]))/U[i-1,i-1]
    return(x)


# In[19]:


def assemble_K(n):
    N = n**2
    K = np.zeros(shape=(N,N))
    for i in range(N): #create main diagonal
        K[i,i]=4 
    for i in range(N-1): #create first diagonal
        K[i+1,i]=-1
        K[i,i+1]=-1
    for i in range(N-n): #create mth diagonal
        K[i+n,i]=-1
        K[i,i+n]=-1
    k=1
    for i in range(N-1): #replace -1 with 0 at boundary
        if (k)%n==0:
            K[i+1,i]=0
            K[i,i+1]=0
        k+=1
        
    return(K)
    


# In[21]:


def form_Z_list(T): #Forms all of the t vectors into a graphable list
    z=[]
    for case in T:
        zi=[]
        for i in case:
            for j in i:
                zi.append(j)
        z.append(zi)

    Z_list = []
    for i in z:
        Z=np.empty((n,n))
        count=0
        for j in range(n):
                for k in range(n):
                    Z[k,j] = i[count]
                    count += 1
        Z_list.append(Z)
    return(Z_list)


# In[22]:


def cholesky(A):
    n = len(A)                                          
    L = np.zeros(shape=(n,n))                           
    for j in range(n):                                
        Lkk_sqrd = A[j,j]-np.dot(L[j,0:j],L[j,0:j])
        L[j,j] = np.sqrt(Lkk_sqrd)
        for i in range(j+1,n):
            L[i,j] = (A[i,j] - np.dot(L[i,0:j],L[j,0:j]))/L[j,j]
    LU = np.matmul(L,L.transpose())
    assert((K==LU).any()),"error in factorization" #check that the factorization is correct
    return(L)


# In[23]:


def jacobi(A,b,N=10, tol=1.0e-9):
    n = len(A)                                          
    x0 = np.zeros(shape=(n,1))                          
    x = np.zeros(shape=(n,1))                           

    condition_no = np.linalg.cond(A)

    for i in range(n):
        if np.abs(A[i,i]) < tol:                       
            row_ind = np.argmax(abs(A[:,i]))            
            A[i,:] = A[i,:]+A[row_ind,:]                
            b[i,0] = b[i,0]+b[row_ind,0]                

    d = 2.0*tol                                         
    k = 0                                               

    while k<N and d > tol:
        for i in range(n):
            x[i,0] = (b[i,0]- np.dot(A[i,:i],x0[:i,0]) - np.dot(A[i,(i+1):],x0[(i+1):,0])  )/A[i,i]

        if condition_no < 2:
            d = la.norm(x-x0)                              
        else:
            d = la.norm(np.matmul(A,x) - b)                 

        k = k + 1                                          
        x0 = x                                              
        
    return x


# In[24]:


# SET 1

n=4 #size of grid of nodes
K = assemble_K(n)
b = [] # list containing each boundary condition vectors
cases = get_cases()
for caseNo in cases:
    caseNo = int(caseNo)
    boundaries = build_b1(n,caseNo) 
    b.append(boundaries)
    
#solve for t    
L = cholesky(K)
U = L.transpose()
T=[]
for condition in b:
    y = forward_sub(L,condition)
    t = back_sub(U,y)
    t = (t.reshape((n,n)))
    t = np.flip(t,0)
    T.append(t)

#create data
X = np.arange(0,1,1/n)
Y = np.arange(0,1,1/n)
#Z = form_Z_list(T)
Z = T

#plot contour graphs
for i in range(len(Z)):
    h = plt.contourf(X,Y,Z[i])
    heading = "Case:"+ str(cases[i])
    plt.title(heading)
    plt.xlabel('x')
    plt.ylabel('y')
    ax = plt.subplot()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(h, cax=cax)
    cbar.set_label('Temperature', rotation=270)
    plt.show()


# In[26]:


#SET 2
n=50 #size of grid of nodes
K = assemble_K(n)
cases = get_cases()
#case = int(cases[0])


#create data
X = np.arange(0,1,1/n)
Y = np.arange(0,1,1/n)
for case in cases:
    case = int(case)
    b = build_b2(n,case)
    T = jacobi(K,b,N=50, tol=1.0e-9)
    #T = la.solve(K,b)
    T = T.reshape((n, n))
    T = np.flip(T, 0)

    h = plt.contourf(X,Y,T)
    heading = "Case:"+ str(case)
    plt.title(heading)
    plt.xlabel('x')
    plt.ylabel('y')
    ax = plt.subplot()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(h, cax=cax)
    cbar.set_label('Temperature', rotation=270)
    plt.show()


# In[ ]:





# In[ ]:




