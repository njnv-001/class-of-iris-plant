import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import random
from sigmoid import sigmoid
from sklearn.linear_model import LogisticRegression;

data = pd.read_csv('iris.data')
data_frame = pd.DataFrame(data)
data_frame.head()
m = data_frame.shape[0]
n = 4
X = np.zeros((m, n))
iris_satosa = data_frame[data_frame['e'] == 'Iris-setosa']
iris_versicolour = data_frame[data_frame['e'] == 'Iris-versicolor']
iris_virginica = data_frame[data_frame['e'] == 'Iris-virginica']

# a = plt 

# a.scatter('a','b',data=iris_satosa.loc[:,['a','b']],c='red',label='Iris-setosa')
# a.scatter('a','b',data=iris_versicolour.loc[:,['a','b']],c='green',label='Iris-versicolor')
# a.scatter('a','b',data=iris_virginica.loc[:,['a','b']],c='pink',label='Iris-virginica')
# a.xlabel('Sepal length in cm')
# a.ylabel('Sepal width in cm')
# a.legend()
# a.show()

# b = plt 

# b.scatter('c','d',data=iris_satosa.loc[:,['c','d']],c='red',label='Iris-setosa')
# b.scatter('c','d',data=iris_versicolour.loc[:,['c','d']],c='green',label='Iris-versicolor')
# b.scatter('c','d',data=iris_virginica.loc[:,['c','d']],c='pink',label='Iris-virginica')
# b.xlabel('Petal length in cm')
# b.ylabel('Petal length in cm')
# b.legend()
# b.show()

# c = plt 
# c.hist('a',data=data_frame,color='green',edgecolor='black',label='Sepals length distributon')
# c.legend()
# c.show()

# d=plt
# d.hist('b',data=data_frame,color='blue',edgecolor='black',label='Sepals width distributon')
# d.legend()
# d.show()

# e=plt
# e.hist('c',data=data_frame,color='red',edgecolor='black',label='Petals length distributon')
# e.legend()
# e.show()

# f=plt
# f.hist('d',data=data_frame,color='yellow',edgecolor='black',label='Petals width distributon')
# f.legend()
# f.show()
input_layer = 4 
hidden_layer = 5 
labels = 3 

X[:, 0] = data_frame['a'].values
X[:, 1] = data_frame['b'].values
X[:, 2] = data_frame['c'].values
X[:, 3] = data_frame['d'].values

theta1 = np.random.rand(4,5)
theta2 = np.random.rand(3,5)

data_frame.loc[data_frame['e'] == 'Iris-setosa'] = 1
data_frame.loc[data_frame['e'] == 'Iris-versicolor'] = 2
data_frame.loc[data_frame['e'] == 'Iris-virginica'] = 3
y = np.zeros((m, 1))
y[:,0] = data_frame['e'].values
lamb = 1
y = y.ravel()
y_new = np.zeros((m,labels))
for i in range(0,m):
    y_new[i,int(y[i]-1)] = 1
    
new_theta1 = np.zeros((4,5))
new_theta2 = np.zeros((3,5))

# q = np.append(theta2.flatten(),theta1.flatten())
# e = 1/10000
# t_theta1 = theta1+e
# t_theta2 = theta2+e

# a1 = X
# a1 = np.append(np.ones((m,1)),X,1)
# a2 = sigmoid(t_theta1,a1)
# a2 = np.append(np.ones((m,1)),a2,1)
# a3 = sigmoid(t_theta2,a2)
# JT = (1/m)*(-y_new*np.log(a3)-(1-y_new)*np.log(1-a3)) +lamb*(np.sum(np.sum(np.multiply(t_theta1[:,2:],t_theta1[:,2:])))+np.sum(np.sum(np.multiply(t_theta2[:,2:],t_theta2[:,2:]))))/(2*m)
# u_theta1 = theta1-e
# u_theta2 = theta2-e

# a1 = X
# a1 = np.append(np.ones((m,1)),X,1)
# a2 = sigmoid(u_theta1,a1)
# a2 = np.append(np.ones((m,1)),a2,1)
# a3 = sigmoid(u_theta2,a2)
# JU = (1/m)*(-y_new*np.log(a3)-(1-y_new)*np.log(1-a3)) +lamb*(np.sum(np.sum(np.multiply(u_theta1[:,2:],u_theta1[:,2:])))+np.sum(np.sum(np.multiply(u_theta2[:,2:],u_theta2[:,2:]))))/(2*m)
# M = JT-JU/2*e
# print(np.shape(M))
# alpha = 0.001
# cost = []
# del_31 = a3 - y_new
# del_21 = np.multiply(np.dot(del_31,t_theta2),np.multiply(a2,(1-a2)))
# del_21 = del_21[:,1:]
# new_t_theta1 = (1/m)*np.dot(del_21.T,a1)
# new_t_theta2 = (1/m)*np.dot(del_31.T,a2)
# new_t_theta1[:,1:] = new_t_theta1[:,1:] + (lamb/m)*(t_theta1[:,1:])
# new_t_theta2[:,1:] = new_t_theta2[:,1:] + (lamb/m)*(t_theta2[:,1:]) 


a1 = X
a1 = np.append(np.ones((m,1)),X,1)
a2 = sigmoid(theta1,a1)
a2 = np.append(np.ones((m,1)),a2,1)
a3 = sigmoid(theta2,a2)
J = (1/m)*np.sum(-y_new*np.log(a3)-(1-y_new)*np.log(1-a3)) +lamb*(np.sum(np.sum(np.multiply(theta1[:,2:],theta1[:,2:])))+np.sum(np.sum(np.multiply(theta2[:,2:],theta2[:,2:]))))/(2*m)
alpha = 0.001
cost = []
del_31 = a3 - y_new
del_21 = np.multiply(np.dot(del_31,theta2),np.multiply(a2,(1-a2)))
del_21 = del_21[:,1:]
new_theta1 = (1/m)*np.dot(del_21.T,a1)
new_theta2 = (1/m)*np.dot(del_31.T,a2)
new_theta1[:,1:] = new_theta1[:,1:] + (lamb/m)*(theta1[:,1:])
new_theta2[:,1:] = new_theta2[:,1:] + (lamb/m)*(theta2[:,1:]) 

# u = np.append(new_theta2.flatten(),new_theta1.flatten())
# print(np.shape(u))
# for i in range(0,m):
#     a_1 = X[i,:]
#     a_1 = np.append([1],a_1)
#     a_2 = sigmoid(theta1,a_1)
#     a_2 = np.append([1],a_2)
#     a_3 = sigmoid(theta2,a_2)
#     del_3 = a_3 - y_new[i,:].T
#     del_2 = np.multiply(np.dot(theta2.T,del_3),np.multiply(a_2,(1-a_2)))
#     del_2 = del_2[1:]
#     new_theta1 = new_theta1 +del_2[:,np.newaxis]*a_1[:,np.newaxis].T
#     new_theta2 = new_theta2 +del_3[:,np.newaxis]*a_2[:,np.newaxis].T

# new_theta1  = new_theta1/m
# new_theta2 = new_theta2/m
# new_theta1[:,1:] = new_theta1[:,1:] + (lamb/m)*(theta1[:,1:])
# new_theta2[:,1:] = new_theta2[:,1:] + (lamb/m)*(theta2[:,1:]) 


for i in range(0,800):
    theta1 = theta1 - (alpha*new_theta1)
    theta2 = theta2 - (alpha*new_theta2)
    J1 = -(1/m)*np.sum(y_new*np.log(a3)+(1-y_new)*np.log(1-a3)) +(lamb/(2*m))*(np.sum(np.multiply(theta1[:,2:],theta1[:,2:]))+np.sum(np.multiply(theta2[:,2:],theta2[:,2:])))
    cost.append(J1)

plt.plot(cost)
plt.show()

T = np.array([[5.1,3.5,1.4,0.2]])
p1 = T
p1 = np.append(np.ones((1,1)),T,1)
p2 = sigmoid(theta1,p1)
p2 = np.append(np.ones((1,1)),p2,1)
p3 = sigmoid(theta2,p2)
# print(np.argmax(p3,axis=1)+1)