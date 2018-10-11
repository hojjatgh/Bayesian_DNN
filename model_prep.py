
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import stwave as st
from sklearn.model_selection import train_test_split


class Model_Prep:
    """
    This class needs to be modified soon. Currently, I only put the functions here.
    """
    
    def __init__(self, params=None):

        if params is not None:
            if 'n_dom' in params:
                self.n_dom = params['n_dom']

    def generate_Q(self, xmin, dx, length_x, nx, L, kernel='Gaussian'):
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

    def initializer_dist(self, bc_l, bc_r, nx, distribution='linear'):
        """
        Generates an initial bathymetry distribution for the domain

        inputs:
        @ bc_l               Bathymetry value at left (near the shore)
        @ nx                 a list containing the length of domain in the x and y directions.
        @ distribution:      Type of distribution for the initial condition
        """

        if distribution == 'linear':
            linear_variation = np.linspace(bc_l, bc_r, nx[0]).reshape(1, nx[0])
            Initial_dist = np.ones((nx[1], nx[0]))*linear_variation
            Initial_dist = Initial_dist.reshape(nx[0]*nx[1], 1)
        else:
            raise NotImplementedError

        return Initial_dist

    def forward_model(self, s, parallel, ncores=None):
        model = st.Model()
        if parallel:
            simul_obs = model.run(s, parallel, ncores)
        else:
            simul_obs = model.run(s, parallel)
        return simul_obs

    def observation_model(self, x, H, lin):
        if lin == 'linear':
            return np.dot(H, x)
        if lin == 'non_linear':
            return np.dot(H, np.sqrt(x))

    def create_sub_samples(self, prof_vel, prof_dep, rec_len, nx):

        """
        This function will get the wave celerity and bathymetry values for the
        whole domain and generate a two dimentional numpy arrays X and a 1D numpy
        array y. The i'th column of X contains the wave celerity values in a square
        with length 'rec_len' around the i'th grid point. The y(i) contains the
        bathymetry value corresponding to i'th grid point.

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

    def create_sub_samples_test(self, prof_vel, rec_len, nx):

        """
        This function will get the wave celerity and bathymetry values for the whole
        domain and generate a two dimentional numpy arrays X and a 1D numpy array y.
        The i'th column of X contains the wave celerity values in a square with length
        'rec_len' around the i'th grid point. The y(i) contains the bathymetry value
        corresponding to i'th grid point.

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
        kk = 0
        for i in range(n_prof):
            dom_vel = np.reshape(prof_vel[:, i], (nx[1], nx[0]))
            for j in range((rec_len-1)//2, nx[0]-((rec_len-1)//2)):
                for k in range((rec_len-1)//2, nx[1]-((rec_len-1)//2)):
                    X[:, kk] = dom_vel[k-(rec_len-1)//2:k+(rec_len-1)//2+1, j-(rec_len-1)//2:j+(rec_len-1)//2+1].ravel()
                    kk += 1

        return X

    def generate_training_data(self, n_dom, min_bc_l, max_bc_l, nx, rec_len, A, sig_obs, file_name='training_data.mat', save_data=True):

        """
        Generates synhetic training data by creating n_dom bathymetry profiles and then creating X and y
        arrays by calling create_sub_samples function for each domain.
        """

        bc_r = 0
        bc_l = np.linspace(min_bc_l, max_bc_l, n_dom)
        distribution = "linear"
        initial_dist = self.initializer_dist(bc_l[0], bc_r, nx, distribution)
        obs_vel = self.forward_model(initial_dist, parallel=False)
        bathy = np.array(initial_dist).reshape(1, -1)
        obs = np.array(obs_vel).reshape(1, -1)
        obs_vel = obs_vel + np.random.randn(110*83, 1)*sig_obs
        X_tr, y_tr = self.create_sub_samples(obs_vel, initial_dist, rec_len, nx)
        X_tr = X_tr.T
        for i in range(n_dom):
            initial_dist = self.initializer_dist(bc_l[i], bc_r, nx, distribution)
            b = np.dot(A.T, np.random.randn(110*83, 1))*.05
            input_bathy = initial_dist+b
            obs_vel = self.forward_model(input_bathy, parallel=False)
            obs_true = obs_vel.reshape(1, -1)
            obs_vel = obs_vel + np.random.randn(110*83, 1)*sig_obs
            X_obs, y_obs = self.create_sub_samples(obs_vel, input_bathy, rec_len, nx)
            X_tr = np.concatenate((X_tr, X_obs.T), axis=0)
            y_tr = np.concatenate((y_tr, y_obs), axis=0)
            bathy = np.concatenate((bathy, input_bathy.reshape(1, -1)), axis=0)
            obs = np.concatenate((obs, obs_true), axis=0)
        if save_data:
            savemat(file_name, {'X_tr': X_tr, 'y_tr': y_tr})
            #savemat('training_data_whole_domain.mat', {'bathy': bathy, 'obs': obs})

        return X_tr, y_tr

    def generate_test_data(self, rec_len, A, nx, sig_obs, save_data=True):

        data = loadmat('true_depth.mat', squeeze_me=True)
        ref_bathy = data['true']
        ref_bathy[ref_bathy < 0.01] = 0.01
        ref_bathy = ref_bathy.reshape(-1, 1)
        obs_vel = self.forward_model(ref_bathy, parallel=False)
        obs_vel = obs_vel + np.random.randn(110*83, 1)*.1
        X_test, y_test = self.create_sub_samples(obs_vel, ref_bathy, rec_len, nx)
        if save_data:
            savemat('test_data.mat', {'X_test': X_test, 'y_test': y_test})

        return X_test, y_test

    def generate_cov(self, L_div, kernel='Gaussian', file_name='Cov.mat', save_data=True):
        xmin = np.array([0, 0])
        nx = np.array([110, 83])
        dx = np.array([5., 5.])
        length_x = nx*dx
        Q = self.generate_Q(xmin, dx, length_x, nx, length_x/L_div, kernel)  # generating Q matrix
        A = np.linalg.cholesky(Q+.0001*np.eye(110*83))
        if save_data:
            savemat(file_name, {'A': A})
        return A

    def plot_im(self, data, nx, title, x_label, y_label, file_name=None, save_file=False):
        plt.figure()
        plt.imshow(data.reshape(nx[1], nx[0]))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.colorbar()
        plt.show()
        if save_file:
            plt.savefig(file_name)

    def get_std_mean_ens(self, model, X, n_ens, rec_len, A, nx,
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
            X_test = self.create_sub_samples_test(obs_vel, rec_len, nx)
            y_pred = model.predict(X_test.T)
            Y[:, i] = y_pred.squeeze()

        y_mean = np.mean(Y, axis=1)
        Y_diff = Y - y_mean.reshape(-1, 1)
        Post_Cov = np.dot(Y_diff, Y_diff.T)/(n_ens-1)
        y_std = np.sqrt(np.diag(Post_Cov))

        return y_mean, y_std, Post_Cov

