import numpy as np
import theano
import theano.tensor as T
from sklearn.base import BaseEstimator
import logging
import time
import os
import datetime
import cPickle as pickle

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')


# weight initializer, normal by default
def norm_weight(nin, nout=None, scale=0.01, ortho=False):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype('float32')
def uniform_weight(size,scale=0.1):
    return np.random.uniform(size=size,low=-scale, high=scale).astype(theano.config.floatX)


def glorot_uniform(size):
    fan_in, fan_out = size
    s = np.sqrt(6. / (fan_in + fan_out))
    print s
    return np.random.uniform(size=size,low=-s, high=s).astype(theano.config.floatX)


class BiGRU(object):
    def __init__(self, n_in, n_hidden, n_out, activation=T.tanh,inner_activation=T.nnet.sigmoid,
                 output_type='real',batch_size=200):

        self.gru_1 = GRU(n_in,n_hidden,n_out)
        self.gru_2 = GRU(n_in,n_hidden,n_out)

        self.params = self.gru_1.params
        self.params += self.gru_2.params

    def __call__(self, input, input_lm=None, return_list = False):
        reverse_input = input[:,::-1,:]
        reverse_mask = input_lm[:,::-1]

        res1 = self.gru_1(input,input_lm,return_list)
        if return_list == True:
            res2 = self.gru_2(reverse_input,reverse_mask,return_list)[:,::-1,:]
            return T.concatenate([res1,res2],2)
        else:
            res2 = self.gru_2(reverse_input,reverse_mask,return_list)
        return T.concatenate([res1,res2],1)


class GRU_Cell(object):
    def __init__(self, n_in, n_hidden, n_out, activation=T.tanh,inner_activation=T.nnet.sigmoid,
                 output_type='real',batch_size=200):

        self.activation = activation
        self.inner_activation = inner_activation
        self.output_type = output_type

        self.batch_size = batch_size
        self.n_hidden = n_hidden

        # recurrent weights as a shared variable
        self.U_z = theano.shared(ortho_weight(n_hidden),borrow=True)
        self.W_z = theano.shared(glorot_uniform((n_in,n_hidden)),borrow=True)
        self.b_z = theano.shared(value=np.zeros((n_hidden,),dtype=theano.config.floatX),borrow=True)

        self.params = [self.W_z,self.U_z,self.b_z]

    def __call__(self, x_t,h_tm1):
        x_z = T.dot(x_t, self.W_z) + self.b_z
        z = self.inner_activation(x_z + T.dot(h_tm1, self.U_z))

        h = z * h_tm1 + (1 - z) * x_t
        return h


class GRU(object):
    def __init__(self, n_in, n_hidden, n_out, activation=T.tanh,inner_activation=T.nnet.sigmoid,
                 output_type='real',batch_size=200):

        self.activation = activation
        self.inner_activation = inner_activation
        self.output_type = output_type

        self.batch_size = batch_size
        self.n_hidden = n_hidden

        # recurrent weights as a shared variable
        self.U_z = theano.shared(ortho_weight(n_hidden),borrow=True)
        self.W_z = theano.shared(glorot_uniform((n_in,n_hidden)),borrow=True)
        self.b_z = theano.shared(value=np.zeros((n_hidden,),dtype=theano.config.floatX),borrow=True)

        self.U_r = theano.shared(ortho_weight(n_hidden),borrow=True)
        self.W_r = theano.shared(glorot_uniform((n_in,n_hidden)),borrow=True)
        self.b_r = theano.shared(value=np.zeros((n_hidden,),dtype=theano.config.floatX),borrow=True)

        self.U_h = theano.shared(ortho_weight(n_hidden),borrow=True)
        self.W_h = theano.shared(glorot_uniform((n_in,n_hidden)),borrow=True)
        self.b_h = theano.shared(value=np.zeros((n_hidden,),dtype=theano.config.floatX),borrow=True)


        self.params = [self.W_z,self.W_h,self.W_r,
                       self.U_h,self.U_r,self.U_z,
                       self.b_h,self.b_r,self.b_z]

    def __call__(self, input,input_lm=None, return_list = False):
         # activation function
        if input_lm == None:
            self.h_l, _ = theano.scan(self.step2,
                        sequences=input.dimshuffle(1,0,2),
                        outputs_info=theano.shared(value=np.zeros((self.batch_size,self.n_hidden),
                                                                  dtype=theano.config.floatX),borrow=True))
        else:
            self.h_l, _ = theano.scan(self.step,
                        sequences=[input.dimshuffle(1,0,2),T.addbroadcast(input_lm.dimshuffle(1,0,'x'), -1)],
                        outputs_info=theano.shared(value=np.zeros((self.batch_size,self.n_hidden),
                                                                  dtype=theano.config.floatX),borrow=True))
        self.h_l = self.h_l.dimshuffle(1,0,2)
        if return_list == True:
            return self.h_l
        return self.h_l[:,-1,:]

    def step2(self,x_t, h_tm1):
        x_z = T.dot(x_t, self.W_z) + self.b_z
        x_r = T.dot(x_t, self.W_r) + self.b_r
        x_h = T.dot(x_t, self.W_h) + self.b_h
        z = self.inner_activation(x_z + T.dot(h_tm1, self.U_z))
        r = self.inner_activation(x_r + T.dot(h_tm1, self.U_r))

        hh = self.activation(x_h + T.dot(r * h_tm1, self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        return h

    def step(self,x_t,mask, h_tm1):
        #h_tm1 =  mask * h_tm1
        x_z = T.dot(x_t, self.W_z) + self.b_z
        x_r = T.dot(x_t, self.W_r) + self.b_r
        x_h = T.dot(x_t, self.W_h) + self.b_h
        z = self.inner_activation(x_z + T.dot(h_tm1, self.U_z))
        r = self.inner_activation(x_r + T.dot(h_tm1, self.U_r))

        hh = self.activation(x_h + T.dot(r * h_tm1, self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        h = mask * h 

        return h

class RNN(object):
    def __init__(self, input_l, input_r, n_in, n_hidden, n_out, activation=T.tanh,
                 output_type='real',batch_size=200,input_lm=None,input_rm=None):
        if input_lm == None:
            input_lm = theano.shared(value=np.ones((batch_size,20), dtype=theano.config.floatX),borrow=True)
        if input_rm == None:
            input_rm = theano.shared(value=np.ones((batch_size,20), dtype=theano.config.floatX),borrow=True)
        self.activation = activation
        self.output_type = output_type
        # Parameters are reshaped views of theta
        param_idx = 0  # pointer to somewhere along parameter vector

        # recurrent weights as a shared variable
        self.W = theano.shared(ortho_weight(n_hidden),borrow=True,name='W')
        # input to hidden layer weights
        self.W_in = theano.shared(glorot_uniform((n_in,n_hidden)),borrow=True,name='W_in')

        self.h0 = theano.shared(value=np.zeros((batch_size,n_hidden), dtype=theano.config.floatX),borrow=True,name='h0')
        self.bh = theano.shared(value=np.zeros((batch_size,n_hidden), dtype=theano.config.floatX),borrow=True,name='bh')
        #self.by = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX),borrow=True,name='by')
        # for convenience
        self.params = [self.W, self.W_in, self.bh]

        # activation function
        def step(x_t, mask, h_tm1):
            h_tm1 =  mask * h_tm1
            #h_t = h_tm1 + self.bh
            h_t = T.tanh(T.dot(x_t, self.W_in) + \
                                  T.dot(h_tm1, self.W) + self.bh)
            #y_t = T.dot(h_t, self.W_out) + self.by
            return h_t
        #a = T.addbroadcast(input_lm.dimshuffle(1,0), -1)
        self.h_l, _ = theano.scan(step,
                    sequences=[input_l.dimshuffle(1,0,2),T.addbroadcast(input_lm.dimshuffle(1,0,'x'), -1)],
                    outputs_info=theano.shared(value=np.zeros((batch_size,n_hidden), dtype=theano.config.floatX),borrow=True))
        self.h_r, _ = theano.scan(step,
                    sequences=[input_r.dimshuffle(1,0,2),T.addbroadcast(input_rm.dimshuffle(1,0,'x'), -1)],
                    outputs_info=theano.shared(value=np.zeros((batch_size,n_hidden), dtype=theano.config.floatX),borrow=True))
        self.h_l = self.h_l.dimshuffle(1,0,2)
        self.h_r = self.h_r.dimshuffle(1,0,2)


if __name__=="__main__":
    input = T.tensor3()
    input2 = T.tensor3()
    rnn = GRU(input,input2,100,50,2,batch_size=47)
    output = theano.function([input,input2],[rnn.h_l,rnn.h_r])
    print output(np.random.rand(47,20,100).astype('float32'),
                 np.random.rand(47,20,100).astype('float32'))[0].shape
