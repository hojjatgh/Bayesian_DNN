import numpy as np
import matplotlib
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from scipy.linalg import svd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import copy

class river_data_prep:

    def __init__(self, params=None):

        if params is None:
            raise ValueError('Params is not given.')

        if 'prof_vel' in params:
            self.prof_vel = params['prof_vel']
            if self.prof_vel.ndim != 2:
                raise ValueError('velocity array should be 2D')
        else:
            raise ValueError('prof_vel should be given for training purpose.')
        if 'prof_dep' in params:
            self.prof_dep = params['prof_dep']
            if self.prof_dep.ndim != 2:
                raise ValueError('river profile array should be 2D')
        else:
            raise ValueError('')
        if 'prof_vel_test' in params:
            self.prof_vel_test = params['prof_vel_test']
            if self.prof_vel.ndim != 2:
                raise ValueError('velocity array should be 2D')
        else:
            self.prof_vel_test = None
        if 'prof_dep_test' in params:
            self.prof_dep_test = params['prof_dep_test']
            if self.prof_dep_test.ndim != 2:
                raise ValueError('river profile array should be 2D')
        else:
            self.prof_dep_test = None
        if 'n_edge' in params:
            self.n_edge = params['n_edge']
        else:
            raise ValueError('.')
        if 'shuffle' in params:
            self.shuffle = params['shuffle']
        else:
            self.shuffle = False
        if 'seed_num' in params:
            self.seed_num = params['seed_num']
        else:
            self.seed_num = 101
        if 'nx' in params:
            self.nx = params['nx']
        else:
            raise ValueError(' nx should be given')
        if 'divide_domain' in params:
            self.divide_domain = params['divide_domain']
        else:
            self.divide_domain = False
        if self.divide_domain:
            if 'sep_point' in params:
                self.sep_point = params['sep_point']
            else:
                raise ValueError(' sep_point should be given.')
        if 'n_edge_out' in params:
            self.n_edge_out = params['n_edge_out']
        else:
            self.n_edge_out = 1
        if 'mirror' in params:
            self.mirror = params['mirror']
        else:
            self.mirror = False
        if 'pca' in params:
            self.pca = params['pca']
        else:
            self.pca = False
        if self.pca:
            if 'n_pc' in params:
                self.n_pc = params['n_pc']
            else:
                raise ValueError('n_pc should be given')
            if 'len_scale' in params:
                self.len_scale = params['len_scale']
            else:
                raise ValueError('len_scale should be given.')
            if 'kernel' in params:
                self.kernel = params['kernel']
            else:
                self.kernel = 'Gaussian'
            if 'xmin' in params:
                self.xmin = params['xmin']
            else:
                raise ValueError('xmin should be given')
            if 'dx' in params:
                self.dx = params['dx']
            else:
                raise ValueError('x should be given')
            if 'size_dom' in params:
                self.size_dom = params['size_dom']
            else:
                raise ValueError('size_dom')
            Q = self.generate_Q(self.xmin, self.dx, self.size_dom, self.len_scale, self.kernel)
            self.u = self.generate_basis(Q, self.n_pc)

    def create_sub_samples(self, prof_vel, prof_dep, rec_len, nx):

        """
        This function will get the wave celerity and bathymetry values for the whole domain and generate
        a two dimentional numpy arrays X and a 1D numpy array y. The i'th column of X contains the
        wave celerity values in a square with length 'rec_len' around the
        i'th grid point. The y(i) contains the bathymetry value corresponding to i'th grid point.
        inputs:

        @ prof_vel:         a 2D numpy array containg the wave celerity for the domain
        @ prof_dep:         a 2D numpy array containg the depth for the domain
        @ rec_len:          The length of the square around each grid point for measuring wave celerity.
                            rec_len should be an odd integer
        @ nx                a list containing the length of domain in the x and y directions.

        outputs:
        X:                  2D numpy array of size (rec_len**2, N) containing the wave celerity (inputs of DNN)
        y:                  1D numpy array of size (N, ) containing the bathymetry values (outputs of DNN)
        """

        n_prof = prof_vel.shape[1]
        n_sample_per_prof = (nx[1]-rec_len+1)*(nx[0]-rec_len+1)
        n_sample = n_sample_per_prof*n_prof
        X = np.zeros((rec_len**2, n_sample))
        # print(X.shape)
        y = np.zeros((n_sample, 1))
        kk = 0
        for i in range(n_prof):
            dom_vel = np.reshape(prof_vel[:, i], (nx[1], nx[0]))
            dom = np.reshape(prof_dep[:, i], (nx[1], nx[0]))
            for j in range((rec_len-1)//2, nx[0]-((rec_len-1)//2)):
                for k in range((rec_len-1)//2, nx[1]-((rec_len-1)//2)):
                    X[:, kk] = dom_vel[k-(rec_len-1)//2:k+(rec_len-1)//2+1, j-(rec_len-1)//2:j+(rec_len-1)//2+1].ravel()
                    y[kk] = dom[k, j]
                    kk += 1

        return X, y

    def create_sub_samples_CNN(self, prof_vel, prof_dep, rec_len, nx):

        """
        This function will get the wave celerity and bathymetry values for the whole domain and
        generate a two dimentional numpy arrays X and a 1D numpy array y. The i'th column of X
        contains the wave celerity values in a square with length 'rec_len' around the i'th grid
        point. The y(i) contains the bathymetry value corresponding to i'th grid point.

        inputs:

        @ prof_vel:         a 2D numpy array containg the wave celerity for the domain
        @ prof_dep:         a 2D numpy array containg the depth for the domain
        @ rec_len:          The length of the square around each grid point for measuring wave celerity.
                            rec_len should be an odd integer
        @ nx                a list containing the length of domain in the x and y directions.

        outputs:
        X:                  2D numpy array of size (rec_len**2, N) containing the wave celerity (inputs of DNN)
        y:                  1D numpy array of size (N, ) containing the bathymetry values (outputs of DNN)
        """

        n_prof = prof_vel.shape[1]
        n_sample_per_prof = (nx[0]-rec_len+1)
        n_sample = n_sample_per_prof*n_prof
        X = np.zeros((rec_len*nx[1], n_sample))
        # print(X.shape)
        y = np.zeros((nx[1], n_sample))
        kk = 0
        for i in range(n_prof):
            dom_vel = np.reshape(prof_vel[:, i], (nx[1], nx[0]))
            dom = np.reshape(prof_dep[:, i], (nx[1], nx[0]))
            for k in range((rec_len-1)//2, nx[0]-((rec_len-1)//2)):
                X[:, kk] = dom_vel[:, k-(rec_len-1)//2:k+(rec_len-1)//2+1].ravel()
                y[:, kk] = dom[:, k]
                kk += 1

        return X, y

    def create_sub_samples_CNN_Tr_Test(self, prof_vel, prof_dep, rec_len, nx, sep_point=250):

        """
        This function will get the wave celerity and bathymetry values for the whole domain and generates
        a two dimentional numpy arrays X and a 1D numpy array y. The i'th column of X contains
        the wave celerity values in a square with length 'rec_len' around the i'th grid point.
        The y(i) contains the bathymetry value corresponding to i'th grid point.

        inputs:

        @ prof_vel:         a 2D numpy array containg the wave celerity for the domain
        @ prof_dep:         a 2D numpy array containg the depth for the domain
        @ rec_len:          The length of the square around each grid point for measuring wave celerity.
                            rec_len should be an odd integer
        @ nx                a list containing the length of domain in the x and y directions.

        outputs:
        X:                  2D numpy array of size (rec_len**2, N) containing the wave celerity (inputs of DNN)
        y:                  1D numpy array of size (N, ) containing the bathymetry values (outputs of DNN)
        """
        n_prof = prof_vel.shape[1]
        n_sample_per_prof_tr = (sep_point-rec_len+1)
        n_sample_per_prof_test = (nx[0]-sep_point-rec_len+1)
        n_sample_tr = n_prof*n_sample_per_prof_tr
        n_sample_test = n_prof*n_sample_per_prof_test
        X_tr = np.zeros((rec_len*nx[1], n_sample_tr))
        X_test = np.zeros((rec_len*nx[1], n_sample_test))
        # print(X.shape)
        y_tr = np.zeros((nx[1], n_sample_tr))
        y_test = np.zeros((nx[1], n_sample_test))
        kk = 0
        jj = 0
        for i in range(n_prof):
            dom_vel = np.reshape(prof_vel[:, i], (nx[1], nx[0]))
            dom = np.reshape(prof_dep[:, i], (nx[1], nx[0]))
            for k in range((rec_len-1)//2, sep_point-((rec_len-1)//2)):
                X_tr[:, kk] = dom_vel[:, k-(rec_len-1)//2:k+(rec_len-1)//2+1].ravel()
                y_tr[:, kk] = dom[:, k]
                kk += 1
            for k in range(sep_point+(rec_len-1)//2, nx[0]-((rec_len-1)//2)):
                X_test[:, jj] = dom_vel[:, k-(rec_len-1)//2:k+(rec_len-1)//2+1].ravel()
                y_test[:, jj] = dom[:, k]
                jj += 1

        return X_tr, y_tr, X_test, y_test

    def create_sub_samples_CNN2(self, prof_vel, prof_dep, rec_len, nx, rec_len_out):

        """
        This function will get the wave celerity and bathymetry values for the whole domain and generate
        a two dimentional numpy arrays X and a 1D numpy array y. The i'th column of X contains the wave
        celerity values in a square with length 'rec_len' around the i'th grid point. The y(i) contains
        the bathymetry value corresponding to i'th grid point.

        inputs:

        @ prof_vel:         a 2D numpy array containg the wave celerity for the domain
        @ prof_dep:         a 2D numpy array containg the depth for the domain
        @ rec_len:          The length of the square around each grid point for measuring wave celerity.
                            rec_len should be an odd integer
        @ nx                a list containing the length of domain in the x and y directions.

        outputs:
        X:                  2D numpy array of size (rec_len**2, N) containing the wave celerity (inputs of DNN)
        y:                  1D numpy array of size (N, ) containing the bathymetry values (outputs of DNN)
        """

        n_prof = prof_vel.shape[1]
        n_sample_per_prof = (nx[0]-rec_len+1)
        n_sample = n_sample_per_prof*n_prof
        X = np.zeros((rec_len*nx[1], n_sample))
        # print(X.shape)
        y = np.zeros((rec_len_out*nx[1], n_sample))
        kk = 0
        for i in range(n_prof):
            dom_vel = np.reshape(prof_vel[:, i], (nx[1], nx[0]))
            # dom = np.reshape(prof_dep[:, i], (nx[1], nx[0]))
            for k in range((rec_len-1)//2, nx[0]-((rec_len-1)//2)):
                X[:, kk] = dom_vel[:, k-(rec_len-1)//2:k+(rec_len-1)//2+1].ravel()
                y[:, kk] = dom_vel[:, k-(rec_len_out-1)//2:k+(rec_len_out-1)//2+1].ravel()
                kk += 1

        return X, y

    def xy_vel_sep(self, prof_vel):

        N, n_prof = prof_vel.shape
        x_vel = np.zeros((N//2, n_prof))
        y_vel = np.zeros((N//2, n_prof))
        mag_vel = np.zeros((N//2, n_prof))
        for i in range(n_prof):
            x_vel[:, i] = prof_vel[0::2, i]
            y_vel[:, i] = prof_vel[1::2, i]
            mag_vel[:, i] = np.sqrt(x_vel[:, i]**2+y_vel[:, i]**2)

        return x_vel, y_vel, mag_vel

    def DNN_train_predict(self, DNN_Net, vel_prof_tr, prof_tr, vel_prof_te, param, n_edge):

        n_layer = param['n_layer']
        n_hidden = param['n_hidden']
        n_epoch_ = param['n_epoch']
        optimizer = param['optimizer']
    #    network = DNN_Net.DNN_Net(vel_prof_tr, prof_tr, n_layer, n_hidden, n_epoch = n_epoch_,
    #                              Normalize = True, optimizer_meth = optimizer, method = 'Ensemble')
        network = DNN_Net.DNN_Net(vel_prof_tr, prof_tr, n_layer, n_hidden, n_epoch=n_epoch_,
                                  Normalize=True, optimizer_meth=optimizer, method='batchnorm')
        bathy_dnn = network.predict(vel_prof_te)
        plt.plot(network.history.history['val_loss'])
        plt.figure()
        N = n_edge-1
        plt.imshow(bathy_dnn.reshape(501-N, 41-N).T)
        plt.colorbar()

        return network

    def DNN_train_predict_B(self, DNN_Net, vel_prof_tr, prof_tr, vel_prof_te, param, n_edge):

        n_layer = param['n_layer']
        n_hidden = param['n_hidden']
        n_epoch_ = param['n_epoch']
        optimizer = param['optimizer']
    #    network = DNN_Net.DNN_Net(vel_prof_tr, prof_tr, n_layer, n_hidden, n_epoch = n_epoch_,
    #                              Normalize = True, optimizer_meth = optimizer, method = 'Ensemble')
        network = DNN_Net(vel_prof_tr, prof_tr, n_layer, n_hidden,
                          n_epoch=n_epoch_, Normalize=True, optimizer_meth=optimizer, method='batchnorm')
        bathy_dnn = network.predict(vel_prof_te)
        plt.plot(network.history.history['val_loss'])
        plt.figure()
        N = n_edge-1
        plt.imshow(bathy_dnn.reshape(501-N, 41).T)
        plt.colorbar()

        return network

    def DNN_predict(self, model, vel_prof_te, n_edge, clim):
        bathy_dnn = model.predict(vel_prof_te)
        N = n_edge-1
        plt.imshow(bathy_dnn.reshape(501-N, 41).T)
        plt.colorbar()
        plt.clim(clim)
        return bathy_dnn

    def generate_X_Y(self, mag_vel, riv_prof, mag_vel_red, prof_red_riv, n_edge, nx, test_size):
        X_tr_mag, Y_tr = self.create_sub_samples(mag_vel, riv_prof, n_edge, nx)
        X_train, _, y_train, _ = train_test_split(np.sqrt(X_tr_mag.T), Y_tr, test_size=0.1, random_state=101)
        np.random.seed(100)
        ord_shuffle = np.arange(X_train.shape[0])
        np.random.shuffle(ord_shuffle)
        X = X_train[ord_shuffle, :]
        Y = y_train[ord_shuffle]
        X_red_riv, Y_red_riv = self.create_sub_samples(mag_vel_red.reshape(-1, 1),
                                                       prof_red_riv.reshape(-1, 1), n_edge, nx)

        return X, Y, X_red_riv, Y_red_riv

    def replicate_pr(self, vel, depth):
        riv_prof_reverse = np.zeros(depth.shape)
        vel_reverse = np.zeros(vel.shape)
        for i in range(depth.shape[1]):
            a = vel[:, i].reshape(41, 501)
            aa = depth[:, i].reshape(41, 501)
            b = np.zeros(a.shape)
            bb = np.zeros(aa.shape)
            for j in range(a.shape[0]):
                b[j, :] = a[40-j, :]
                bb[j, :] = aa[40-j, :]
            vel_reverse[:, i] = b.reshape(-1,)
            riv_prof_reverse[:, i] = bb.reshape(-1,)
        v = np.concatenate((vel, vel_reverse), axis=1)
        riv_prof = np.concatenate((depth, riv_prof_reverse), axis=1)
        return v, riv_prof

    def plt_im_tri(self, depth, fig_name, show_file=True, vmin_=21.0, vmax_=29.0):
        mesh = loadmat("mesh.mat")
        triangles = mesh['triangles']
        meshnode = mesh['meshnode']
        # velocity_obs_loc = np.loadtxt("observation_loc_drogue12345_50ft.dat")
        # bathy = loadmat('true.mat')
        # z_in = np.squeeze(bathy['s_true'])
        # true in meter
        matplotlib.rcParams.update({'font.size': 16})

        offsetx = 220793.
        offsety = 364110.
        fig_index = 1
        plt.figure(fig_index, figsize=(10., 10.), dpi=100)
        fig_index += 1
        ax = plt.gca()
        im = plt.tripcolor(meshnode[:, 0]*0.3010-offsetx, meshnode[:, 1]*0.3010-offsety,
                           triangles, depth*0.301, cmap=plt.get_cmap('jet'), vmin=vmin_, vmax=vmax_, label='_nolegend_')
        ax.set_xlabel("Easting [m]")
        ax.set_ylabel("Northing [m]")
        plt.gca().set_aspect('equal', adjustable='box')

        plt.axis([0., 1000., 0., 530.])
        plt.xticks(np.arange(0., 1000.+10., 200.0))
        plt.yticks(np.arange(0., 530.+10., 200.0))

        cbar = plt.colorbar(im, fraction=0.025, pad=0.05)
        # cbar = plt.colorbar(im, pad=0.05 )
        cbar.set_label('Elevation [m]')
        plt.rcParams['axes.axisbelow'] = True
        plt.rc('axes', axisbelow=True)
        plt.grid()
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(fig_name)
        if show_file:
            plt.show()

    #    plt.close('all')

    def concatenate_data(self, file_names, v_name, z_name):
        if len(file_names) == 0:
            print('There is no file name in the array')
            return
        data = loadmat(file_names[0])
        velocity = data[v_name[0]]
        depth = data[z_name[0]]

        for i in range(1, len(file_names)):
            data = loadmat(file_names[i])
            velocity = np.concatenate((velocity, data[v_name[i]]), axis=1)
            depth = np.concatenate((depth, data[z_name[i]]), axis=1)
        count = 0
        for i in range(1, velocity.shape[1]):
            if np.linalg.norm(velocity[:, i]) < 1:
                velocity[:, i] = velocity[:, i-1]
                depth[:, i] = depth[:, i-1]
                count += 1
        print(count)
        return velocity, depth

    def plot_long_net_result(self, prof_red_riv, DNN_result, nx, n_edge, file_name):
        a = prof_red_riv.reshape(nx[1], nx[0])
        b = DNN_result.reshape((nx[0]-n_edge+1, nx[1])).T
        aa = np.zeros(a.shape)
        aa[:, 0:(n_edge//2)] = b[:, 0:(n_edge//2)]
        aa[:, nx[0]-(n_edge//2):] = b[:, nx[0]-n_edge+1-(n_edge//2):]
        aa[:, (n_edge//2):nx[0]-(n_edge//2)] = b
        depth = aa.reshape(-1,)
        self.plt_im_tri(depth.squeeze(), file_name)
        return depth

    def generate_Q(self, xmin, dx, nx, L, kernel='Gaussian'):
        """
        Generate covariance matrix for the given 2D grid
        inputs:
        @ xmin:   a list containing minimum values for x and y coordinates, [x_1, x_2].
        @ dx      a list containing the increment in the x and y directions.
        @ dx:     a list containing the length of domain in the x and y directions.
        @ nx:     a list containing the number of cells in the x and y directions.
        @ L:      a list containing the length scale in the x and y directions.
        Kernel:   The type of kernel for the covariance matrix
        Output:
        Q:        Covariance matrix
        """
        length_x = [nx[0]*dx[0], nx[1]*dx[1]]
        xr = np.linspace(xmin[0]+.5*dx[0],  xmin[0]+length_x[0]-.5*dx[0], nx[0])
        yr = np.linspace(xmin[1]+.5*dx[1],  xmin[1]+length_x[1]-.5*dx[1], nx[1])
        x, y = np.meshgrid(xr, yr)
        x1, x2 = np.meshgrid(x, x)
        y1, y2 = np.meshgrid(y, y)
        distance_x = (x1 - x2)**2
        distance_y = (y1 - y2)**2
        distance = distance_x/(L[0]**2) + distance_y/(L[1]**2)
        if kernel == 'Gaussian':
            Q = np.exp(-distance)
        elif kernel == 'Exponential':
            Q = np.exp(-np.sqrt(distance))
        else:
            raise NotImplementedError

        return Q

    def generate_basis(self, Q, n_pc):

        u, _, _ = svd(Q)
        self.u = u[:, :n_pc]
        return self.u

    def project_to_basis(self, vec, u):

        if len(vec) != u.shape[0]:
            raise ValueError('size of the vector does not match the size of basis')

        return np.dot(u.T, vec)

    def create_sub_samples_pc_Tr_Test(self, prof_vel, prof_dep, rec_len_dep, rec_len_vel, nx, u):

        """
        This function will get the wave celerity and bathymetry values for the whole domain and generates
        a two dimentional numpy arrays X and a 1D numpy array y. The i'th column of X contains
        the wave celerity values in a square with length 'rec_len' around the i'th grid point.
        The y(i) contains the bathymetry value corresponding to i'th grid point.

        inputs:

        @ prof_vel:         a 2D numpy array containg the wave celerity for the domain
        @ prof_dep:         a 2D numpy array containg the depth for the domain
        @ rec_len:          The length of the square around each grid point for measuring wave celerity.
                            rec_len should be an odd integer
        @ nx                a list containing the length of domain in the x and y directions.

        outputs:
        X:                  2D numpy array of size (rec_len**2, N) containing the wave celerity (inputs of DNN)
        y:                  1D numpy array of size (N, ) containing the bathymetry values (outputs of DNN)
        """
        n_prof = prof_vel.shape[1]
        n_sample_per_prof_tr = (self.sep_point-rec_len_vel+1)
        n_sample_per_prof_test = (nx[0]-self.sep_point-rec_len_vel+1)
        n_sample_tr = n_prof*n_sample_per_prof_tr
        n_sample_test = n_prof*n_sample_per_prof_test
        X_tr = np.zeros((rec_len_vel*nx[1], n_sample_tr))
        X_test = np.zeros((rec_len_vel*nx[1], n_sample_test))
        # print(X.shape)
        y_tr = np.zeros((u.shape[1], n_sample_tr))
        y_test = np.zeros((u.shape[1], n_sample_test))
        kk = 0
        jj = 0
        for i in range(n_prof):
            dom_vel = np.reshape(prof_vel[:, i], (nx[1], nx[0]))
            dom = np.reshape(prof_dep[:, i], (nx[1], nx[0]))
            for k in range((rec_len_vel-1)//2, self.sep_point-((rec_len_vel-1)//2)):
                X_tr[:, kk] = dom_vel[:, k-(rec_len_vel-1)//2:k+(rec_len_vel-1)//2+1].ravel()
                vel = dom[:, k-(rec_len_dep-1)//2:k+(rec_len_dep-1)//2+1].ravel()
                y_tr[:, kk] = self.project_to_basis(vel, u)
                kk += 1
            for k in range(self.sep_point+(rec_len_vel-1)//2, nx[0]-((rec_len_vel-1)//2)):
                X_test[:, jj] = dom_vel[:, k-(rec_len_vel-1)//2:k+(rec_len_vel-1)//2+1].ravel()
                vel = dom[:, k-(rec_len_dep-1)//2:k+(rec_len_dep-1)//2+1].ravel()
                y_test[:, jj] = self.project_to_basis(vel, u)
                jj += 1

        return X_tr, y_tr, X_test, y_test

    def create_sub_samples_pc(self, prof_vel, prof_dep, rec_len_dep, rec_len_vel, nx, u):

        """
        This function will get the wave celerity and bathymetry values for the whole domain and
        generate a two dimentional numpy arrays X and a 1D numpy array y. The i'th column of X
        contains the wave celerity values in a square with length 'rec_len' around the i'th grid
        point. The y(i) contains the bathymetry value corresponding to i'th grid point.

        inputs:

        @ prof_vel:         a 2D numpy array containg the wave celerity for the domain
        @ prof_dep:         a 2D numpy array containg the depth for the domain
        @ rec_len:          The length of the square around each grid point for measuring wave celerity.
                            rec_len should be an odd integer
        @ nx                a list containing the length of domain in the x and y directions.

        outputs:
        X:                  2D numpy array of size (rec_len**2, N) containing the wave celerity (inputs of DNN)
        y:                  1D numpy array of size (N, ) containing the bathymetry values (outputs of DNN)
        """

        n_prof = prof_vel.shape[1]
        n_sample_per_prof = (nx[0]-rec_len_vel+1)
        n_sample = n_sample_per_prof*n_prof
        X = np.zeros((rec_len_vel*nx[1], n_sample))
        # print(X.shape)
        y = np.zeros((u.shape[1], n_sample))
        kk = 0
        for i in range(n_prof):
            dom_vel = np.reshape(prof_vel[:, i], (nx[1], nx[0]))
            dom = np.reshape(prof_dep[:, i], (nx[1], nx[0]))
            for k in range((rec_len_vel-1)//2, nx[0]-((rec_len_vel-1)//2)):
                X[:, kk] = dom_vel[:, k-(rec_len_vel-1)//2:k+(rec_len_vel-1)//2+1].ravel()
                vel = dom[:, k-(rec_len_dep-1)//2:k+(rec_len_dep-1)//2+1].ravel()
                y[:, kk] = self.project_to_basis(vel, u)
                kk += 1

        return X, y

    def Gen_train_data(self):

        X, Y = self.Gen_data(self.prof_vel, self.prof_dep, mirror=self.mirror,
                             shuffle=self.shuffle, divide_domain=self.divide_domain)
        return X, Y

    def gen_test_data(self, vel=None, dep=None, gen_riv_prof=True, noise=None):

        if vel is None:
            raise ValueError('velocity profile is not given.')
        if dep is None:
            raise ValueError('river profile is not given.')
        vel = vel.reshape(-1, 1)
        dep = dep.reshape(-1, 1)
        X, Y = self.Gen_data(vel, dep, noise)

        return X, Y

    def Gen_data(self, prof_vel_in, prof_dep_in, noise=None, mirror=False, shuffle=False, divide_domain=False):

        x_vel, y_vel, vel_mag = self.xy_vel_sep(prof_vel_in)
        if mirror:
            prof_vel, prof_dep = self.replicate_pr(vel_mag, prof_dep_in)
        else:
            prof_vel = vel_mag
            prof_dep = prof_dep_in
        if noise is not None:
            prof_vel = prof_vel + noise.reshape(prof_vel.shape)
        tmp = copy.deepcopy(prof_dep)
        prof_dep = copy.deepcopy(prof_vel)
        prof_vel = copy.deepcopy(tmp)
        if self.pca:
            prof_dep = copy.deepcopy(x_vel)
            if divide_domain:
                X_tr, Y_trx, _, _ = self.create_sub_samples_pc_Tr_Test(prof_vel, prof_dep, self.n_edge,
                                                                      self.n_edge, self.nx, self.u)
            else:
                X_tr, Y_trx = self.create_sub_samples_pc(prof_vel, prof_dep, self.n_edge, self.n_edge, self.nx, self.u)
            prof_dep = copy.deepcopy(y_vel)
            if divide_domain:
                _, Y_try, _, _ = self.create_sub_samples_pc_Tr_Test(prof_vel, prof_dep, self.n_edge,
                                                                      self.n_edge, self.nx, self.u)
            else:
                _, Y_try = self.create_sub_samples_pc(prof_vel, prof_dep, self.n_edge, self.n_edge, self.nx, self.u)
            Y_tr= np.concatenate((Y_trx, Y_try), axis=0)
            
        elif self.n_edge_out > 1:
            X_tr, Y_tr = self.create_sub_samples_CNN2(prof_vel, prof_dep,
                                                      self.n_edge, self.nx, self.n_edge_out)
        elif divide_domain:
            X_tr, Y_tr, _, _ = self.create_sub_samples_CNN_Tr_Test(prof_vel,
                                                                   prof_dep, self.n_edge, self.nx, self.sep_point)
        else:
            X_tr, Y_tr = self.create_sub_samples(prof_vel, prof_dep, self.n_edge, self.nx)

        if shuffle:
            np.random.seed(self.seed_num)
            ord_shuffle = np.arange(X_tr.shape[1])
            np.random.shuffle(ord_shuffle)
            X_tr = X_tr[:, ord_shuffle]
            Y_tr = Y_tr[:, ord_shuffle]
        return X_tr.T, Y_tr.T

    def post_process(self, data):
        a = np.zeros((self.nx[1], self.nx[0]))
        for i in range(data.shape[0]):
            a[:, i:i+self.n_edge] += np.dot(self.u, data[i, :]).reshape(self.nx[1], -1)
        for i in range(self.n_edge):
            a[:, i] = a[:, i]/(i+1)
            a[:, self.nx[0]-i-1] = a[:, self.nx[0]-i-1]/(i+1)

        a[:, self.n_edge: self.nx[0]-self.n_edge] = a[:, self.n_edge: self.nx[0]-self.n_edge]/self.n_edge
        return a

    def generate_realization(self, A, n_ens, std):
        noise = np.dot(A.T, np.random.randn(A.shape[1], n_ens))*std
        return noise

    def plot_1d(self, ref_prof, pred_prof, est_std=None, x=None, y=None, file_name=None, plt_show=False):

        a = .3*ref_prof.reshape(self.nx[1], self.nx[0])
        b = .3*pred_prof.reshape(self.nx[1], self.nx[0])
        if est_std is not None:
            c = .3*est_std.reshape(self.nx[1], self.nx[0])
        if x is not None:
            plt.plot(a[:, x])
            plt.hold(True)
            plt.plot(b[:, x])
            if est_std is not None:
                plt.plot(b[:, x]+2*c[:, x], 'k-.')
                plt.plot(b[:, x]-2*c[:, x], 'k-.')
        elif y is not None:
            plt.plot(a[y, :])
            plt.hold(True)
            plt.plot(b[y, :])
            if est_std is not None:
                plt.plot(b[y, :]+2*c[y, :], 'k-.')
                plt.plot(b[y, :]-2*c[y, :], 'k-.')
        else:
            raise ValueError('please give x or y coordinates')
        if file_name:
            plt.savefig(file_name)
        if plt_show:
            plt.show()
