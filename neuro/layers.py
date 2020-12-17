import numpy as np
from neuro import activations


# Predict layer class
class Predict:
    def __init__(self,
                 n_units: int,
                 n_input: int,
                 l2: float,
                 activation_type: str,
                 trainable: bool = True,
                 bypass: bool = False) -> None:

        # Member properties
        self.n_units = n_units
        self.n_units_prev = n_input
        self.l2 = l2
        self._bypass = None
        self._trainable = None
        self.activate = None
        self.d_activate = None
        self.optimizer = None
        self.w = [np.random.randn(n_units, n_input) * np.sqrt(2 / n_input)]
        self.b = [np.zeros((n_units, 1))]
        self.dw, self.db = [None], [None]
        self.z, self.y = None, None
        self.X, self.dX = None, None
        self.set_activation(activation_type)
        self.bypass = bypass
        self.trainable = trainable
        self.opt = dict()

    # bypass property
    @property
    def bypass(self):
        return self._bypass

    # bypass setter
    @bypass.setter
    def bypass(self, bypass_):
        self._bypass = bypass_

    # trainable property
    @property
    def trainable(self):
        return self._trainable

    # trainable setter
    @trainable.setter
    def trainable(self, trainable_):
        self._trainable = trainable_

    # Set activation function
    def set_activation(self, activation_type):
        if activation_type == 'softmax':
            self.activate = activations.softmax
        elif activation_type == 'sigmoid':
            self.activate = activations.sigmoid
        elif activation_type == 'linear':
            self.activate = activations.linear
        else:
            raise Exception('PredictLayer.set_activation(): Undefined activation type')

    # Forward pass
    def forward(self, x):
        self.X = x
        self.z = np.matmul(self.X, self.w[0].T) + self.b[0].T
        self.y = self.activate(self.z)
        return self.y

    # Backward pass
    def backward(self, z_):
        self.dw[0] = np.matmul(z_.T, self.X) + self.l2 * self.w[0] / z_.shape[0]
        self.db[0] = np.sum(z_.T, axis=1, keepdims=True)
        self.dX = np.matmul(z_, self.w[0])
        return self.dX

    # Return zipped gradients
    def gradients(self):
        return zip(self.dw, self.db)

    # Contribution to regularization error
    def reg(self):
        return 0.5 * self.l2 * np.sum((np.square(self.w[0])))

    def update_params(self, opt_param, optimizer):
        if self.trainable:
            optimizer(self, opt_param)

    def build_opt(self, optimizer):
        if optimizer == 'adam':
            self.opt['mw'] = 0
            self.opt['mb'] = 0
            self.opt['vw'] = 0
            self.opt['vb'] = 0


# Dense layer class
class Dense:
    def __init__(self,
                 n_units: int,
                 n_input: int,
                 l2: float,
                 activation_type: str,
                 trainable: bool = True,
                 bypass: bool = False) -> None:

        # Member properties
        self.n_units = n_units
        self.n_units_prev = n_input
        self.l2 = l2
        self._bypass = None
        self._trainable = None
        self.activate = None
        self.d_activate = None
        self.w = [np.random.randn(n_units, n_input) * np.sqrt(2 / n_input)]
        self.b = [np.zeros((n_units, 1))]
        self.dw, self.db = [None], [None]
        self.g, self.h = [None], [None]
        self.X, self.dX = None, None
        self.set_activation(activation_type)
        self.bypass = bypass
        self.trainable = trainable
        self.opt = dict()

    # bypass property
    @property
    def bypass(self):
        return self._bypass

    # bypass setter
    @bypass.setter
    def bypass(self, bypass_):
        self._bypass = bypass_

    # trainable property
    @property
    def trainable(self):
        return self._trainable

    # trainable setter
    @trainable.setter
    def trainable(self, trainable_):
        self._trainable = trainable_

    # Set activation function
    def set_activation(self, activation_type):
        if activation_type == 'relu':
            self.activate = activations.relu
            self.d_activate = activations.d_relu
        elif activation_type == 'linear':
            self.activate = activations.linear
            self.d_activate = activations.d_linear
        elif activation_type == 'l_relu':
            self.activate = activations.l_relu
            self.d_activate = activations.d_l_relu
        elif activation_type == 'tanh':
            self.activate = activations.tanh
            self.d_activate = activations.d_tanh
        else:
            raise Exception('LinearLayer.set_activation(): Undefined activation type')

    # Forward pass
    def forward(self, x):
        self.X = x
        self.g[0] = np.matmul(self.X, self.w[0].T) + self.b[0].T
        self.h[0] = self.activate(self.g[0])
        return self.h[0]

    # Backward pass
    def backward(self, h_):
        g_ = h_ * self.d_activate(self.g[0])
        self.dw[0] = np.matmul(g_.T, self.X) + self.l2 * self.w[0] / h_.shape[0]
        self.db[0] = np.sum(g_.T, axis=1, keepdims=True)
        self.dX = np.matmul(g_, self.w[0])
        return self.dX

    # Return gradients
    def gradients(self):
        return zip(self.dw, self.db)

    # Contribution to regularization error
    def reg(self):
        return 0.5 * self.l2 * np.sum((np.square(self.w[0])))

    def update_params(self, opt_param, optimizer):
        if self.trainable:
            optimizer(self, opt_param)

    def build_opt(self, optimizer):
        if optimizer == 'adam':
            self.opt['mw'] = 0
            self.opt['mb'] = 0
            self.opt['vw'] = 0
            self.opt['vb'] = 0


# Residual block class
class ResBlock:
    # Constructor
    def __init__(self, n, n_prev, l2, trainable=True):
        # Save inputs
        self.n_units = n
        self.u_units_prev = n_prev
        self.l2 = l2
        self.trainable = trainable
        self.opt = dict()

        # Member properties
        self.w = [np.random.randn(n, n_prev) * np.sqrt(2 / n_prev),
                  np.random.randn(n, n) * np.sqrt(2 / n)]
        self.b = [np.zeros((n, 1)), np.zeros((n, 1))]
        self.dw, self.db = [None, None], [None, None]
        self.g, self.h = [None, None], [None, None]
        self.X, self.dX = None, None

    # Set layer trainability
    def trainability(self, trainability):
        self.trainable = trainability

    # ReLU
    def relu(self, x):
        return x * (x > 0)

    # ReLU Derivative
    def d_relu(self, x):
        return 1 * (x > 0)

    # Linear Combiner
    def alc(self, X, w, b):
        return np.matmul(X, w.T) + b.T

    # Forward propagation
    def forward(self, X):
        self.X = X
        self.g[0] = self.alc(self.X, self.w[0], self.b[0])
        self.h[0] = self.relu(self.g[0])
        self.g[1] = self.alc(self.h[0], self.w[1], self.b[1]) + self.X
        self.h[1] = self.relu(self.g[1])
        return self.h[1]

    # Back propagation
    def backward(self, x_):
        h2_ = x_
        g2_ = h2_ * self.d_relu(self.g[1])
        self.dw[1] = np.matmul(g2_.T, self.h[1]) + self.l2 * self.w[1] / h2_.shape[0]
        self.db[1] = np.sum(g2_.T, axis=1, keepdims=True)

        h1_ = np.matmul(g2_, self.w[1])
        g1_ = h1_ * self.d_relu(self.g[0])
        self.dw[0] = np.matmul(g1_.T, self.X) + self.l2 * self.w[0] / h1_.shape[0]
        self.db[0] = np.sum(g1_.T, axis=1, keepdims=True)

        self.dX = np.matmul(g1_, self.w[0]) + g2_
        return self.dX

    # Return gradients
    def gradients(self):
        return self.dw, self.db

    # Contribution to regularization error
    def reg(self):
        return 0.5 * self.l2 * (np.sum(np.square(self.w[0])) + np.sum(np.square(self.w[1])))
