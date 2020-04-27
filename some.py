import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import random
from sigmoid import sigmoid
from sklearn.linear_model import LogisticRegression;
from sklearn.neural_network import MLPClassifier;

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

nn  = MLPClassifier(hidden_layer_sizes=(5),max_iter=2000,alpha=0.01)
nn.fit(X,y_new)
print(nn.predict([[5.1,3.5,1.4,0.2]]))





