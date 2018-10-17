import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from keras import optimizers
from DNN_Net2 import DNN_Net2 as DNN_Net
from data_prep import river_data_prep as rdp
import timeit

data = loadmat('data.mat')
prof_vel = data['vel']
prof_dep = data['depth']

n_edge = 11
nx = [501, 41]
params_rdp = {'nx': nx, 'prof_vel': prof_vel, 'prof_dep': prof_dep,
              'n_edge': n_edge, 'shuffle': True, 'seed_num': 101, 'divide_domain': False, 'mirror': True,
              'pca': True, 'n_pc': 41, 'len_scale': [40, 50], 'kernel': 'Gaussian',
              'xmin': [0, 0], 'dx': [5, 5], 'size_dom': [n_edge, nx[1]]}
model_data = rdp(params_rdp)

X_train, Y_train = model_data.Gen_train_data()


n_layer = 6
n_hidden = [400, 400, 300, 200, 100, 41]
# n_hidden = [400, 400, 300,200, 100,5, 41]
# n_hidden = [550, 550, 350,200, 100, 41]
# n_hidden = [300, 300, 200,150, 100, 41]
# n_hidden = [900, 800, 800, 500, 400, 200, 100, 70, 41]
# param['n_hidden'] = [30, 10,1]
np.random.seed(101)
n_epoch_ = 1
sgd = optimizers.SGD(lr=0.0006, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=2e-6, amsgrad=False)

optimizer = adam
start = timeit.default_timer()
network = DNN_Net(X_train, Y_train, n_layer, n_hidden, n_epoch=n_epoch_, Normalize=True, optimizer_meth=optimizer,
                  method='batchnorm', act_fcn='tanh', reg=0.001, batch_s=512)
run_time = timeit.default_timer() - start

network.model.save('model_50.h5')


data_red_river = loadmat('red_river.mat')
vel_red = data_red_river['vel_ref']
prof_red = data_red_river['depth_ref']

X, Y = model_data.gen_test_data(vel=vel_red.reshape(-1, 1), dep=prof_red.reshape(-1, 1))

bathy_dnn = network.predict(X)
pred_river = model_data.post_process(bathy_dnn)


data = loadmat('A_Lx30_Ly10.mat')
A = data['A']
n_ens = 50
std = .1
realization = model_data.generate_realization(A, n_ens, std)
plt.figure(1)
model_data.plt_im_tri(realization[:, 0].ravel(), 'realization0.jpg', False, -.25, .25)
results = np.zeros((len(pred_river.ravel()), n_ens))
for i in range(n_ens):
    X_r, Y_r = model_data.gen_test_data(vel=vel_red, dep=prof_red, noise=realization[:, i])
    bathy_ens = network.predict(X_r)
    results[:, i] = model_data.post_process(bathy_ens).ravel()

y_mean = np.mean(results, axis=1)
Y_diff = results - y_mean.reshape(-1, 1)
Post_Cov = np.dot(Y_diff, Y_diff.T)/(n_ens-1)
y_std = np.sqrt(np.diag(Post_Cov))
savemat('prediction_ens.mat', {'pred_river': pred_river, 'y_std': y_std, 'y_mean': y_mean})
savemat('prediction.mat', {'pred_river': pred_river})
model_data.plt_im_tri(y_mean.ravel(), 'ens_res1.jpg', False)
model_data.plt_im_tri(y_std.ravel(), 'ens_std1.jpg', False, 0, 1)
model_data.plt_im_tri(np.abs(y_mean.ravel()-prof_red.ravel()), 'ens_err1.jpg', False, 0, 1.5)
model_data.plt_im_tri(pred_river.ravel(), 'res1.jpg', False)
model_data.plt_im_tri(np.abs(pred_river.ravel()-prof_red.ravel()), 'res1.jpg', False, 0, 1.5)
model_data.plot_1d(prof_red, pred_river, est_std=y_std, x=20, y=None, file_name='ens_1d1.jpg', plt_show=False)
