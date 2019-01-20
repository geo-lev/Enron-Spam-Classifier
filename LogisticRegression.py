import numpy as np

class LogisticRegression :
    
    def __init__(self , learning_rate = 0.001, iters=500 , reg=0.001 , bias = True , batch_size=64):
        self.learning_rate = learning_rate
        self.iters=iters
        self.reg = reg
        self.bias = True
        self.batch_size = batch_size
        print('Initialized Logistic Regression object')
        
    def add_bias(self, X):
        bias = np.ones((X.shape[0],1))
        return np.concatenate((bias,X),axis=1)
        
    
    def sigmoid(self,X , weight):
        z = X.dot(weight.T)
        return 1/(1+np.exp(-z))

    def log_likelihood(self , X,y,weight):
        term1 = y*(self.sigmoid(X,weight))
        term2 = np.log(1-self.sigmoid(X,weight))
        return np.sum(term1+(1-y)*term2) + 0.5*self.reg*weight.T.dot(weight)
   
    def gradient(self , X , y ,weight ):
        term1 = self.sigmoid(X,weight)
        term2 = y-term1
        return X.T.dot(term2) - self.reg*weight
    

    def fit(self ,X ,y, epochs=2 ):
        print('Starting fit function')
        if self.bias :
            X = self.add_bias(X)
            print('Added bias column to matrix')
        weight = np.random.rand(1,X.shape[1])
        cost_history = np.zeros(epochs)
        for i in range(0,epochs) :
            print('Epoch :', i+1 )
            for j in range(0,X.shape[0],self.batch_size):
                X_batch = X[i:i+self.batch_size,:]
                y_batch = y[i:i+self.batch_size]
                prediction = self.sigmoid(X_batch , weight)
                print('computed prediction :' ,j)
                weight = weight + self.gradient(X_batch,y_batch,weight)*self.learning_rate
                print('adjusted weights :', j)
                #loss = self.log_likelihood(X_batch , y_batch , weight)
                #print('computed log likelihood : ' , loss)
            #cost_history[i] = loss
        return weight , cost_history
    
    
    def predict(self, X ,weight , bias=True) :
        if bias:
            X=self.add_bias(X)
        return ( np.round(self.sigmoid(X , weight)))
        