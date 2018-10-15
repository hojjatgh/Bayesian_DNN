import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from scipy.io import savemat, loadmat
import math
from keras import optimizers
from DNN_Net2 import DNN_Net2 as DNN_Net
from sklearn.model_selection import train_test_split
import os
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from data_prep import river_data_prep as rdp

data = loadmat('data.mat')
prof_vel = data['vel']
prof_dep = data['depth']

params_rdp = {'nx': [501, 41], 'prof_vel': prof_vel, 'prof_dep': prof_dep,
             'n_edge': 11, 'shuffle': True,'seed_num': 101, 'divide_domain': False, 'mirror': True,
             'pca': True, 'n_pc':41, 'len_scale': [40, 50], 'kernel': 'Gaussian',
             'xmin': [0, 0], 'dx': [5, 5], 'size_dom':[11, 41]}
model_data = rdp(params_rdp)

X, Y = model_data.Gen_train_data()

import timeit
param = {}
n_layer = 8
#n_hidden = [400, 400, 300,200, 100, 41]
#n_hidden = [400, 400, 300,200, 100,5, 41]
#n_hidden = [550, 550, 350,200, 100, 41]
#n_hidden = [300, 300, 200,150, 100, 41]
n_hidden = [800, 800, 400,400, 100,100,70, 41]
#param['n_hidden'] = [30, 10,1]
np.random.seed(101)
n_epoch_ = 50
sgd = optimizers.SGD(lr=0.0006, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=2e-6, amsgrad=False)

optimizer = adam
start = timeit.default_timer()
network = DNN_Net(X, Y, n_layer, n_hidden, n_epoch = n_epoch_, Normalize = True, optimizer_meth = optimizer,
                  method = 'batchnorm', act_fcn='tanh', reg = 0.001, batch_s=512)
run_time = timeit.default_timer() -start

network.model.save('model_50.h5')


data_red_river = loadmat('red_river.mat')
vel_red = data_red_river['vel_ref']
prof_red = data_red_river['depth_ref']

params_rdp = {'nx': [501, 41], 'prof_vel': vel_red.reshape(-1,1), 'prof_dep': prof_red.reshape(-1,1),
             'n_edge': 11, 'shuffle': False, 'divide_domain': False, 'mirror': False,
             'pca': True, 'n_pc':41, 'len_scale': [40, 50], 'kernel': 'Gaussian',
             'xmin': [0, 0], 'dx': [5, 5], 'size_dom':[11, 41]}
red_river = rdp(params_rdp)

X, Y = red_river.Gen_train_data()

bathy_dnn = network.predict(X)
pred_river = red_river.post_process(bathy_dnn)

red_river.plt_im_tri(pred_river.ravel(), 'res2.jpg', False)