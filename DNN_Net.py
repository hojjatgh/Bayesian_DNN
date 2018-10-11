import warnings
# import keras
# import stwave as st
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dropout, Dense  # , Activation
# from keras.optimizers import SGD, Adam
from keras import regularizers
# from keras import metrics, losses
from keras import backend as K
import time
warnings.filterwarnings("ignore")


class DNN_Net:

    def __init__(self, params):

        """
        Constructor for a class implementing Densely connected neural network that provides
        uncertainty quantification using ensemble approach, dropout method, or batch normalization.

        @ param X_train        numpy array containing the training data with shape (n_train, n_feature)
        @ param y_train        numpy array with the target value for the training data with shape (n_train, 1)
        @ param n_hidden       Number of hidden layers
        @ param n_unit         A vector with the number of hidden units for each hidden layer
        @ param n_epoch        Number of epochs for training the network

        @ param X_test         A numpy array containg the test data
        @ param y_test         A numpy array containg the values for the test data
        @ param Normalize      Whether to normalize the input features. It is recommended to normalize
                               the input features unless they have very similar distribution

        @ param optimizer      Type of optimizers to find the wieghting matrices. Default optimizer is SGD
        @ param Method         Determines the method used for obtaining the uncertainty in the estimation
                               The current options are:

                               - Ensemble: Trains the network and then at the prediction step, it uses an ensemble
                               realization of the input to make the prediction, take the mean as the estimated
                               value and variance of the  ensembles as the estimation variance.
                               The method is based on the approach used in the following paper.
                               "Jeong, H., Sun, A. Y., Lee, J., & Min, B. (2018). A Learning-based Data-driven
                               Forecast Approach for Predicting Future Reservoir Performance. AWR.

                               - Dropout: use a dropout layer in each layer of the network in the training time.
                               In the test time the method uses a Monte Carlo approach and makes T stochastic
                               prediction and then computes the mean and variance. The method is based on the
                               approach used in the following paper: "Gal, Y., & Ghahramani, Z. (2015). Dropout
                               as a Bayesian approximation. arXiv preprint arXiv:1506.02157."

        """
        # X_train, y_train, n_hidden, n_unit, n_epoch=10, X_test=None,
    #             y_test=None, Normalize=True, optimizer_meth='SGD', method='Ensemble', dropout=.9):

        if params is not None:
            if 'X_train' in params:
                self.X_train = params['X_train']
            else:
                raise ValueError(' X_train is not defined')
            if 'Y_train' in params:
                self.Y_train = params['Y_train']
            else:
                raise ValueError(' Y_train is not defined')
            if 'n_hidden' in params:
                self.n_hidden = params['n_hidden']
            else:
                raise ValueError(' n_hidden is not defined')
            if 'n_unit' in params:
                self.n_unit = params['n_unit']
            else:
                raise ValueError(' n_unit is not defined')
            if 'n_epoch' in params:
                self.n_epoch = params['n_epoch']
            else:
                raise ValueError(' n_epoch is not defined')
            if 'method' in params:
                self.method = params['method']
            else:
                raise ValueError(' method is not defined')
            if 'Normalize' in params:
                self.Normalize = params['Normalize']
            else:
                raise ValueError(' Normalize is not defined')
            if 'optimizer_meth' in params:
                self.optimizer_meth = params['optimizer_meth']
            else:
                raise ValueError(' optimizer_meth is not defined')
            if 'X_test' in params:
                self.X_test = params['X_test']
            if 'y_test' in params:
                self.y_test = params['y_test']
            if 'dropout' in params:
                self.dropout = params['dropout']
            if 'verbose' in params:
                self.verbose = params['verbose']
            else:
                self.verbose = False

        if self.Normalize:
            self.std_X_train = np.std(self.X_train, 0)
            self.std_X_train[self.std_X_train == 0] = 1
            self.mean_X_train = np.mean(self.X_train, 0)
            self.std_Y_train = np.std(self.Y_train)
            self.mean_Y_train = np.mean(self.Y_train)
        else:
            self.std_X_train = np.ones(self.X_train.shape[1])
            self.mean_X_train = np.zeros(self.X_train.shape[1])
            self.std_Y_train = 1
            self.mean_Y_train = 0

        self.X_train = (self.X_train - self.mean_X_train)/self.std_X_train
        self.Y_train = (self.Y_train - self.mean_Y_train)/self.std_Y_train
        # constructing the network
        feature_dim = self.X_train.shape[1]
        # N = X_train.shape[0]

        if len(self.n_unit) != self.n_hidden:
            raise ValueError("The length of number of units is not equal to the number of hidden layer")
        dnn_model = Sequential()
        if self.method == 'Ensemble':
            dnn_model.add(Dense(units=self.n_unit[0], input_dim=feature_dim, activation='relu',
                                kernel_regularizer=regularizers.l2(0.01)))
            for i in range(1, self.n_hidden-1):
                dnn_model.add(Dense(units=self.n_unit[i], input_dim=self.n_unit[i-1], activation='relu',
                                    kernel_regularizer=regularizers.l2(0.01)))
            dnn_model.add(Dense(units=1, input_dim=self.n_unit[self.n_hidden-1], activation='linear',
                                kernel_regularizer=regularizers.l2(0.01)))

        elif self.method == 'dropout':
            # todo: Need to modify the dropout method. Currently the result does not look good
            dnn_model.add(Dropout(self.dropout, input_shape=(self.X_train.shape[1],)))
            dnn_model.add(Dense(units=self.n_unit[0], input_dim=feature_dim, activation='relu',
                                kernel_regularizer=regularizers.l2(0.01)))
            for i in range(1, self.n_hidden-1):
                dnn_model.add(Dropout(self.dropout))
                dnn_model.add(Dense(units=self.n_unit[i], input_dim=self.n_unit[i-1], activation='relu',
                                    kernel_regularizer=regularizers.l2(0.01)))
            dnn_model.add(Dropout(self.dropout))
            dnn_model.add(Dense(units=1, input_dim=self.n_unit[self.n_hidden-1], activation='linear',
                                kernel_regularizer=regularizers.l2(0.01)))

        else:
            raise NotImplementedError

        dnn_model.compile(loss='mean_squared_error', optimizer=self.optimizer_meth, metrics=['mse'])

        start_time = time.time()
        history = dnn_model.fit(self.X_train, self.Y_train, batch_size=128, epochs=self.n_epoch,
                                verbose= self.verbose, validation_split=.2)

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

        prediction = model.predict(X_test, batch_size=128, verbose=self.verbose)
        prediction = prediction * self.std_Y_train + self.mean_Y_train

        return prediction

    def get_std_mean_ens(self, create_sub_samples_test, X, n_ens, rec_len, A, nx,
                         sig_pri, prior_cov='Gaussian', save_data=False):

        """
        Function for making prediction with DNN and also for providing estimation uncertainty
        using Ensemble approach

        Outputs:

        y_mean:        Predicted mean for the test target value
        y_std:         std_var for the prediction using ensemble approach
        Post_Cov       The posterior covariance

        """

        # np.random.seed(100)
        # ref_bathy = X
        # ref_bathy[ref_bathy<0.01] = 0.01
        # ref_bathy = ref_bathy.reshape(-1,1)
        # obs_vel_org = forward_model(ref_bathy, parallel=False)
        obs_vel_org = X
        N = (nx[0]-rec_len/2-1)*(nx[1]-rec_len/2-1)
        Y = np.zeros((N, n_ens))
        for i in range(n_ens):
            if prior_cov == 'Identity':
                obs_vel = obs_vel_org + np.random.randn(nx[1]*nx[0], 1)*sig_pri
            elif prior_cov == 'Gaussian':
                obs_vel = obs_vel_org + np.dot(A, np.random.randn(nx[1]*nx[0], 1))*sig_pri
            else:
                raise NotImplementedError
            X_test = create_sub_samples_test(obs_vel, rec_len, nx)
            y_pred = self.predict(X_test.T)
            Y[:, i] = y_pred.squeeze()

        y_mean = np.mean(Y, axis=1)
        Y_diff = Y - y_mean.reshape(-1, 1)
        Post_Cov = np.dot(Y_diff, Y_diff.T)/(n_ens-1)
        y_std = np.sqrt(np.diag(Post_Cov))

        return y_mean, y_std, Post_Cov

    def get_std_mean_MC_dropout(self, forward_model, create_sub_samples, X, rec_len, nx, save_data=False):

        np.random.seed(100)
        ref_bathy = X
        ref_bathy[ref_bathy < 0.01] = 0.01
        ref_bathy = ref_bathy.reshape(-1, 1)
        obs_vel_org = forward_model(ref_bathy, parallel=False)

        X_tes, _ = create_sub_samples(obs_vel_org, ref_bathy, rec_len, nx)
        X_test = X_tes.T
        T = 10000
        predict_stochastic = K.function([self.model.layers[0].input, K.learning_phase()],
                                        [self.model.layers[-1].output])
        Yt_hat = np.array([predict_stochastic([X_test, 1]) for _ in range(T)])
        Yt_hat = Yt_hat * self.std_Y_train + self.mean_Y_train

        return Yt_hat
