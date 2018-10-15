import warnings
warnings.filterwarnings("ignore")

import math
from scipy.io import savemat, loadmat
import keras
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Dense, Activation
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras import metrics, losses
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import time



class DNN_Net2:
    
    def __init__(self, X_train, y_train, n_hidden, n_unit, n_epoch = 10, X_test = None,
                 y_test = None, Normalize = True, optimizer_meth = 'SGD', method = 'Ensemble', act_fcn = 'tanh', reg = 0.001, batch_s = 512, dropout = .9):
        """
        Constructor for a class implementing Densely connected neural network that provides 
        uncertainty quantification using ensemble approach, dropout method, or batch normalization.
        
        @ param X_train        A numpy array containing the training data with shape (n_train, n_feature)
        @ param y_train        A numpy array with the target value for the training data with shape (n_train, 1)
        @ param n_hidden       Number of hidden layers
        @ param n_unit         A vector with the number of hidden units for each hidden layer
        @ param n_epoch        Number of epochs for training the network

        @ param X_test         A numpy array containg the test data
        @ param y_test         A numpy array containg the values for the test data
        @ param Normalize      Whether to normalize the input features. It is recommended to normalize the input features
                               unless they have very similar distribution
        
        @ param optimizer      Determines the type of optimizers to find the wieghting matrices. The default optimizer is SGD
        @ param Method         Determines the method used for obtaining the uncertainty in the estimation
                               The current options are:
                               
                               - Ensemble: Train the network and then at the prediction step, it uses an ensemble realization
                               of the input to make the prediction, take the mean as the estimated value and variance of the 
                               ensembles as the estimation variance.
                               The method is based on the approach used in the following paper.
                               "Jeong, H., Sun, A. Y., Lee, J., & Min, B. (2018). A Learning-based Data-driven
                               Forecast Approach for Predicting Future Reservoir Performance. Advances in Water Resources.
                               
                               - Dropout: use a dropout layer in each layer of the network in the training time. In the test time
                               the method uses a Monte Carlo approach and makes T stochastic prediction and then computes
                               the mean and variance.
                               The method is based on the approach used in the following paper:
                               "Gal, Y., & Ghahramani, Z. (2015). Dropout as a Bayesian approximation.
                               arXiv preprint arXiv:1506.02157."
                                                    
        """
        
        if Normalize:
            self.std_X_train = np.std(X_train, 0)
            self.std_X_train[self.std_X_train==0] = 1
            self.mean_X_train = np.mean(X_train,0)
            self.std_y_train = np.std(y_train,0)
            self.mean_y_train = np.mean(y_train,0)
        else:
            self.std_X_train = np.ones(X_train.shape[1])
            self.mean_X_train = np.zeros(X_train.shape[1])
            self.std_y_train = 1
            self.mean_y_train = 0
        
        X_train = (X_train - self.mean_X_train)/self.std_X_train
        y_train = (y_train - self.mean_y_train)/self.std_y_train
        
        
        # constructing the network
        feature_dim = X_train.shape[1]
        N = X_train.shape[0]
        
        if len(n_unit) != n_hidden:
            raise ValueError, "The length of number of units is not equal to the number of hidden layer"

        dnn_model = Sequential()
        if method =='Ensemble':
            dnn_model.add(Dense(units=n_unit[0],input_dim=feature_dim, activation=act_fcn, kernel_regularizer = regularizers.l2(reg)))
            #dnn_model.add(BatchNormalization())
            for i in range(1, n_hidden-2):
                dnn_model.add(Dense(units=n_unit[i],input_dim=n_unit[i-1],activation=act_fcn, kernel_regularizer = regularizers.l2(reg)))
            dnn_model.add(Dense(units=n_unit[n_hidden-1],input_dim=n_unit[n_hidden-2],activation='linear',kernel_regularizer = regularizers.l2(reg)))
        
        elif method =='dropout':
            ## todo: Need to modify the dropout method. Currently the result does not look good
            dnn_model.add(Dropout(dropout, input_shape=(X_train.shape[1],)))
            dnn_model.add(Dense(units=n_unit[n_hidden-1],input_dim=n_unit[n_hidden-2], activation='linear',kernel_regularizer = regularizers.l2(reg)))
            for i in range(1, n_hidden-1):
                dnn_model.add(Dropout(dropout))
                dnn_model.add(Dense(units=n_unit[i],input_dim=n_unit[i-1],activation='relu',kernel_regularizer = regularizers.l2(0.01)))
            dnn_model.add(Dropout(dropout))
            dnn_model.add(Dense(units=1,input_dim=n_unit[n_hidden-1],activation='linear',kernel_regularizer = regularizers.l2(0.01)))
        elif method =='batchnorm':
            ## todo: Need to modify the dropout method. Currently the result does not look good
            dnn_model.add(Dense(units=n_unit[0],input_dim=feature_dim,kernel_regularizer = regularizers.l2(reg)))
            dnn_model.add(BatchNormalization())
            dnn_model.add(Activation(act_fcn))
            #dnn_model.add(BatchNormalization())
            for i in range(1, n_hidden-2):
                dnn_model.add(Dense(units=n_unit[i],input_dim=n_unit[i-1], kernel_regularizer = regularizers.l2(reg)))
                dnn_model.add(BatchNormalization())
                dnn_model.add(Activation(act_fcn))
            dnn_model.add(Dense(units=n_unit[n_hidden-1],input_dim=n_unit[n_hidden-2], activation='linear',kernel_regularizer = regularizers.l2(reg)))

            
        else:
            raise NotImplementedError
        
        dnn_model.compile(loss='mean_squared_error', optimizer=optimizer_meth, metrics=['mse'])  
        
        start_time = time.time()
        history = dnn_model.fit(X_train, y_train, batch_size=batch_s, epochs=n_epoch, verbose=1, validation_split=.2)
        
        self.history = history
        self.running_time = time.time()-start_time
        
        self.model = dnn_model
        
        
    def predict(self, X_test):
        
        """
        Function for making prediction with DNN and also for providing estimation uncertainty
        using Ensemble approach or dropout method
        
        Outputs:
        
        y_pred:        Predicted mean for the test target value
        MSE:           MSE for the prediction
        var_y          The prediction variance
        
        """
        
        X_test = (X_test - self.mean_X_train)/self.std_X_train
        
        model = self.model
        
        prediction = model.predict(X_test, batch_size = 128, verbose=0)
        prediction = prediction * self.std_y_train + self.mean_y_train
        
        return prediction
        
    def get_std_mean_ens(self, create_sub_samples_test, X, n_ens, rec_len, A, nx, sig_pri, prior_cov = 'Gaussian', save_data = False):
        
        """
        Function for making prediction with DNN and also for providing estimation uncertainty
        using Ensemble approach
        
        Outputs:
        
        y_mean:        Predicted mean for the test target value
        y_std:         std_var for the prediction using ensemble approach
        Post_Cov       The posterior covariance
        
        """

        #np.random.seed(100)
        #ref_bathy = X
        #ref_bathy[ref_bathy<0.01] = 0.01
        #ref_bathy = ref_bathy.reshape(-1,1)
        #obs_vel_org = forward_model(ref_bathy, parallel=False)
        obs_vel_org = X
        N = (nx[0]-rec_len/2-1)*(nx[1]-rec_len/2-1)
        Y = np.zeros((N, n_ens))
        for i in range(n_ens):
            if prior_cov == 'Identity':
                obs_vel = obs_vel_org +np.random.randn(110*83,1)*sig_pri
            elif prior_cov == 'Gaussian':
                obs_vel = obs_vel_org +np.dot(A,np.random.randn(110*83,1))*sig_pri 
            else:
                raise NotImplementedError
            X_test= create_sub_samples_test(obs_vel, rec_len, nx)
            y_pred = self.predict(X_test.T)
            Y[:,i] = y_pred.squeeze()

        y_mean = np.mean(Y, axis = 1)
        Y_diff = Y - y_mean.reshape(-1,1)
        Post_Cov = np.dot(Y_diff, Y_diff.T)/(n_ens-1)
        y_std = np.sqrt(np.diag(Post_Cov))
    
        return y_mean, y_std, Post_Cov
        
    def get_std_mean_MC_dropout(self, forward_model, create_sub_samples, X, rec_len, nx, save_data = False):
        
        np.random.seed(100)
        ref_bathy = X
        ref_bathy[ref_bathy<0.01] = 0.01
        ref_bathy = ref_bathy.reshape(-1,1)
        obs_vel_org = forward_model(ref_bathy, parallel=False)
        
        X_tes,_ = create_sub_samples(obs_vel_org,ref_bathy, rec_len, nx)
        X_test = X_tes.T
        T = 10000
        predict_stochastic = K.function([self.model.layers[0].input, K.learning_phase()], [self.model.layers[-1].output])
        Yt_hat = np.array([predict_stochastic([X_test, 1]) for _ in range(T)])
        Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train
        
        return Yt_hat
        
        
        
