import pandas as pd 
import matplotlib.pyplot as plt 
data = pd.read_csv('iris.data')
data_frame = pd.DataFrame(data)
iris_satosa = data_frame[data_frame['e'] == 'Iris-setosa']
iris_versicolour = data_frame[data_frame['e'] == 'Iris-versicolor']
iris_virginica = data_frame[data_frame['e'] == 'Iris-virginica']
a = plt 

a.scatter('a','b',data=iris_satosa.loc[:,['a','b']],c='red',label='Iris-setosa')
a.scatter('a','b',data=iris_versicolour.loc[:,['a','b']],c='green',label='Iris-versicolor')
a.scatter('a','b',data=iris_virginica.loc[:,['a','b']],c='pink',label='Iris-virginica')
a.legend()
a.show()

