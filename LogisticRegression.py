import numpy as np

class LogisticRegression :
    
    def __init__(self , learning_rate = 0.001, epochs=60 , reg_term=0.0001 , bias = True , batch_size=64):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_term = reg_term
        self.bias = True
        self.batch_size = batch_size
        print('Initialized Logistic Regression object')
        
    def add_bias(self, X):
        bias = np.ones((X.shape[0],1))
        return np.concatenate((bias,X),axis=1)
        
    
    def sigmoid(self,X , weight):
        z = X.dot(weight.T)
        result = 1/(1.0+np.exp(-z))
        result[result == 1 ] = 0.9999999
        result[result == 0 ] = 0.0000001
        return result

    def log_likelihood(self , X,y,weight):
        term1 = y.T*np.log(self.sigmoid(X,weight))  
        #DEBUG print('term1 shape :',term1.shape)
        term2 = (1-y).T*np.log(1-self.sigmoid(X,weight))
        #DEBUG print('term2 shape :', term2.shape)
        term3 = (0.5*self.reg_term*(weight.dot(weight.T)))
        #DEBUG print('term3 shape :', term3.shape)
        result = np.sum(term1+term2) - term3 
        #DEBUG print('log likelihood shape :', result.shape)
        return result
   
    def gradient(self , X , y ,weight ):
        result = X.T.dot((y - self.sigmoid(X,weight))) - (self.reg_term*weight).T
        #DEBUG print('gradient shape: ', result.shape)
        return result.T

    def fit(self ,X ,y):
        if self.bias :
            X = self.add_bias(X)
        weight = np.random.rand(1,X.shape[1])
        np.random.seed(seed=1)
        indices = np.random.permutation(len(X))
        X = X[indices,:]
        y = y[indices]
        cost_history = np.zeros(self.epochs)
        for i in range(0,self.epochs):
            for j in range(0,X.shape[0],self.batch_size):
                X_batch = X[i:i+self.batch_size,:]
                y_batch = y[i:i+self.batch_size]
                weight = weight + self.gradient(X_batch,y_batch,weight)*self.learning_rate
            loss = self.log_likelihood(X_batch , y_batch , weight)
            cost_history[i] = loss
        return weight , cost_history

    
    def predict(self, X ,weight , bias=True) :
        if bias:
            X=self.add_bias(X)
        return ( np.round(self.sigmoid(X , weight)))
        