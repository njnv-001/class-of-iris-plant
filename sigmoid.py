import numpy as np 

def sigmoid(theta,X):
    # print(np.shape(theta),np.shape(X))
    return 1/(1+np.exp(-(np.dot(X,np.transpose(theta)))))
