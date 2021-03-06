import numpy as np
from typing import Union
from neuro import losses
from neuro import layers
from neuro import optimizers
from neuro.misc import defaults
from neuro.misc import utils


class Sequential:
    """
    Class: Sequential ----------------------------------------------------------

    This class allows one to build a machine learning model comprised of layer
    sequences. Currently, the available layer types are:
        Sequential: A sequantial models
             Dense: General dense (fully connected) layer
           Predict: Dense (fully connected) layer used to produce predictions
          Residual: Residual block (in Residual Neural Nets).
    This model can be explicitly trained on its own, or can be part of another
    sequential model and implicitly trained.

    ATTRIBUTES -----------------------------------------------------------------

    template:   *********************
                * name: (type)      *
                *     description   *
                *********************

    N: (int)
        Number of training examples
    data: (np.ndarray)
        Training data
    labels: (np.ndarray)
        Training labels
    labels_original: (np.ndarray)
        Original labels (stored)
    loss: (function)
        loss function
    d_loss: (function)
        loss function derivatives
    loss_type: (str)
        loss function type
    optimizer: (function)
        optimizer function
    optimizer_type: (str)
        optimizer type
    update_opt_param: (function)
        func for updating optimization params
    _bypass: (bool)
        flag for bypassing layer in forward(), backward()
    _trainable: (bool)
        flag for updating parameters in update_params()
    layers: (list)
        container for all layers in model
    training_loss: (list)
        list for storing per-iteration training loss
    accuracy: (list)
        list for storing per-iteration training accuracy
    global_iter: (list)
        global iteration counter

    METHODS --------------------------------------------------------------------

    template:   *************************************
                * name(param:type) -> return type   *
                * ** params description             *
                *                                   *
                * method description                *
                *************************************

    bypass(bypass_: bool) -> None
        ** bypass_: flag for layer bypass (true or false)

        sets its own _bypass attribute to bypass_, and also sets the _bypass
        attribute or every layer in self.layers to _bypass

    trainable(trainable_: bool) -> None
        ** trainable_: flag for layer trainability (true or fase)

        sets its own _trainable attribute to trainable_, and also sets the
        _trainable attribute or every layer in self.layers to _trainable


    """

    def __init__(self,
                 *layers_: Union['Sequential',
                                 layers.Dense,
                                 layers.Predict,
                                 layers.Residual]) -> None:

        # Initialize member properties
        self.N = None
        self.data = None
        self.labels = None
        self.labels_original = None
        self.loss = None
        self.d_loss = None
        self.loss_type = None
        self.optimizer = None
        self.optimizer_type = None
        self.update_opt_param = None
        self._bypass = False
        self._trainable = True
        self.layers = [layer for layer in layers_]
        self.training_loss = []
        self.accuracy = []
        self.global_iter = 0

    # bypass property
    @property
    def bypass(self) -> bool:
        return self._bypass

    # bypass setter
    @bypass.setter
    def bypass(self, bypass_: bool) -> None:
        self._bypass = bypass_
        for layer in self.layers:
            layer.bypass = bypass_

    # trainable property
    @property
    def trainable(self) -> bool:
        return self._trainable

    # trainable setter
    @trainable.setter
    def trainable(self, trainable_: bool) -> None:
        self._trainable = trainable_
        for layer in self.layers:
            layer.trainable = trainable_

    # Set labels
    def set_labels(self, labels: np.ndarray) -> None:
        if self.loss_type == 'softmax_cross_entropy':
            self.labels_original = labels.reshape((self.N,))  # save labels
            self.labels = utils.encode_onehot(labels)  # create onehot
        elif self.loss_type == 'binary_cross_entropy':
            self.labels_original = labels.reshape((self.N,))  # save labels
            self.labels = labels

    # Get regularization error
    def reg(self) -> np.double:
        return sum(layer.reg() for layer in self.layers)

    # Aggregate error
    def error(self, y: np.ndarray, labels: np.ndarray) -> np.double:
        return (self.reg() + self.loss(y, labels)) / labels.shape[0]

    # d(z)/d(error)
    def d_error(self, y: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return self.d_loss(y, labels) / labels.shape[0]

    # Forward propagation
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            if not layer.bypass:
                inputs = layer.forward(inputs)
        return inputs

    # Forward prop through a specified sequence of layers
    def partial_forward(self, inputs: np.ndarray, indices: tuple) -> np.ndarray:
        for index in indices:
            inputs = self.layers[index].forward(inputs)
        return inputs

    # Back propagation
    def backward(self, inputs: np.ndarray) -> None:
        for layer in reversed(self.layers):
            if not layer.bypass:
                inputs = layer.backward(inputs)
        return inputs

    # Propagate through network, calculate derivatives, loss, and predictions
    def propagate(self, data: np.ndarray, labels: np.ndarray) -> (np.double, np.ndarray):
        y = self.forward(data)
        self.backward(self.d_error(y, labels))
        return self.error(y, labels), utils.make_pred_class(y)

    @staticmethod
    def get_opt_param(optimizer: str, opt_param: dict) -> dict:
        if optimizer == 'adam':
            return {key: (opt_param[key] if key in opt_param.keys() else value)
                    for (key, value) in defaults.adam.items()}
        elif optimizer == 'grdec':
            return {key: (opt_param[key] if key in opt_param.keys() else value)
                    for (key, value) in defaults.grdec.items()}

    def build_opt(self, optimizer):

        self.optimizer_type = optimizer

        if optimizer == 'adam':
            # Set optimizer to ADAM
            self.optimizer = optimizers.adam
            self.update_opt_param = optimizers.update_opt_param_adam

        elif optimizer == 'grdec':
            # Set optimizer to GRADIENT DESCENT
            self.optimizer = optimizers.grdec
            self.update_opt_param = optimizers.update_opt_param_grdec

        for layer in self.layers:
            layer.build_opt(optimizer)

    # Build model
    def build(self, loss_type: str, optimizer: str = 'adam'):

        self.loss_type = loss_type
        self.build_opt(optimizer)

        # Set loss function
        if self.loss_type == 'softmax_cross_entropy':
            self.loss = losses.softmax_cross_entropy
            self.d_loss = losses.softmax_cross_entropy_derivative
        elif self.loss_type == 'binary_cross_entropy':
            self.loss = losses.binary_cross_entropy
            self.d_loss = losses.binary_cross_entropy_derivative

    # Fit model, calls desired optimizer
    def fit(self,
            data: np.ndarray,
            labels: np.ndarray,
            epochs: int = 1,
            optimizer: str = 'adam',
            batch_size: int = None,
            iter_epoch: int = 1000,
            **opt_param) -> None:

        # General variables
        self.N = data.shape[0]
        self.data = data
        self.set_labels(labels)

        self.optimize(epochs,
                      iter_epoch,
                      self.N if batch_size is None else batch_size,
                      self.get_opt_param(optimizer, opt_param))

    # Optimize parameters
    def optimize(self,
                 epochs: int,  # Number of training epochs
                 iter_epoch: int,  # Iterations per epoch (default 1000)
                 batch_size: int,  # batch size
                 opt_param: dict) -> None:  # learning rate decay

        # Run optimization epochs
        for epoch in range(epochs):
            # Perform optimization iterations
            for step in range(iter_epoch):
                # Increment global iteration counter
                self.global_iter += 1
                # Update optimization parameters
                self.update_opt_param(opt_param, self.global_iter)
                # Create batch indices
                idx = np.random.choice(self.N, batch_size, replace=False)
                # Propagate through network
                loss, y = self.propagate(self.data[idx, :], self.labels[idx, :])
                # Optimizer parameters in each layer
                self.update_params(opt_param, self.optimizer)
                # Store accuracy and training loss
                self.accuracy.append(np.mean(y == self.labels_original[idx]))
                self.training_loss.append(loss)

    # Update parameters
    def update_params(self, opt_param, optimizer):
        for layer in self.layers:
            if layer.trainable:
                layer.update_params(opt_param, optimizer)

    # Predict class given inputs
    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.forward(data)

    # Evaluate Model
    def evaluate(self, data: np.ndarray, labels: np.ndarray) -> None:
        y = self.predict(data)

        loss = self.error(y, utils.encode_onehot(labels))
        prediction = utils.make_pred_class((labels.size, 1))
        accuracy = np.mean(prediction == labels)

        print('Loss: ' + str(loss) + ', Accuracy: ' + str(accuracy))
