import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('50_Startups.csv',header=None,names=['R&D Spend','Administration','Marketing Spend','Profit'])
data = data.drop(data.index[0])
print(data.head(20))
data = ((data- data.mean(numeric_only=True))/data.std(numeric_only=True))
data.insert(0,'ones',1)
cols = data.shape[1]
x = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
x = np.matrix(x.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0,0,0]))
def CostFunction(x,y,theta):
    inner = np.power(((x*theta.T)-y),2)
    return np.sum((inner)/2*len(x))

def GradientDescent(x,y,theta,alpha,itres):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(itres)
    
    for i in range (itres):
        error =(x*theta.T)-y
        
        
        for j in range (parameters):
            term =np.multiply(error,x[:,j])
            temp[0,j] = theta[0,j] - ((alpha/len(x))*np.sum(term))
        
        theta=temp
        cost[i]=CostFunction(x,y,theta)
    return theta,cost

alpha = 0.1
iters = 100
t,cost = GradientDescent(x,y,theta,alpha,iters)
print (t)
print (cost)

figure,a =plt.subplots(figsize=(5,3))
a.plot(np.arrange(iters),cost,'r')
a.set_xlabel("iteartions")
a.set_ylabel("cost")
