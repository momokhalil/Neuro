import numpy as np
from neuro.misc.defaults import EPSILON


def adam(layer, opt_param):

    for i, (dw, db) in enumerate(layer.gradients()):

        # Weights gradient moment estimates
        mw = opt_param['b1'] * layer.opt['mw'] + (1 - opt_param['b1']) * dw         # first moment
        vw = opt_param['b2'] * layer.opt['vw'] + (1 - opt_param['b2']) * dw ** 2    # second moment
        mw_ = mw / (opt_param['b1_'] + EPSILON)                                     # unbiased mw
        vw_ = vw / (opt_param['b2_'] + EPSILON)                                     # unbiased vw

        # Biases gradient moment estimates
        mb = opt_param['b1'] * layer.opt['mb'] + (1 - opt_param['b1']) * db         # first moment
        vb = opt_param['b2'] * layer.opt['vb'] + (1 - opt_param['b2']) * db ** 2    # second moment
        mb_ = mb / (opt_param['b1_'] + EPSILON)                                     # unbiased mb
        vb_ = vb / (opt_param['b2_'] + EPSILON)                                     # unbiased vb

        # Update weights and biases
        layer.w[i] -= opt_param['learning_rate'] * mw_ / (np.sqrt(vw_) + EPSILON)
        layer.b[i] -= opt_param['learning_rate'] * mb_ / (np.sqrt(vb_) + EPSILON)

        # Store moments
        layer.opt['mw'], layer.opt['vw'] = mw, vw
        layer.opt['mb'], layer.opt['vb'] = mb, vb


def grdec(layer, opt_param):

    for i, (dw, db) in enumerate(layer.gradients()):
        # Update weights and biases
        layer.w[i] -= opt_param['learning_rate'] * dw
        layer.b[i] -= opt_param['learning_rate'] * db


def update_opt_param_adam(opt_param, iteration):
    opt_param['b1_'] = 1 - opt_param['b1'] ** iteration
    opt_param['b2_'] = 1 - opt_param['b2'] ** iteration
    opt_param['learning_rate'] = opt_param['alpha'] * opt_param['decay'] ** iteration
    return opt_param


def update_opt_param_grdec(opt_param, iteration):
    opt_param['learning_rate'] = opt_param['alpha'] * opt_param['decay'] ** iteration
    return opt_param
