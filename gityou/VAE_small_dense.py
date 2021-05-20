import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Input, BatchNormalization, Conv3D, Dense, Flatten, Lambda, Reshape, Concatenate, Conv3DTranspose, MaxPooling3D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

input_shape = (3, 40, 40, 40)
z_dim = 50

def sampling(args):
    mu, sigma = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    epsilon = K.random_normal(shape = (batch, dim))
    return mu + K.exp(0.5 * sigma) * epsilon

def mish(x):
    return x*K.tanh(K.softplus(x))

def get_model():
    enc_in = Input(shape = input_shape)
    
    enc_conv0 = BatchNormalization()(
        Conv3D(
            filters = 6,
            kernel_size = (2, 2, 2),
            strides = (2, 2, 2),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = mish,
            data_format = 'channels_first')(enc_in))
    enc_conv0 = BatchNormalization()(
        Conv3D(
            filters = 12,
            kernel_size = (2, 2, 2),
            strides = (2, 2, 2),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = mish,
            data_format = 'channels_first')(enc_conv0))
   
    
    enc_fc0 = BatchNormalization()(
        Dense(
            units = 1000,
            kernel_initializer = 'glorot_normal',
            activation = mish)(Flatten()(enc_conv0)))
    enc_fc1 = BatchNormalization()(
        Dense(
            units = 1000,
            kernel_initializer = 'glorot_normal',
            activation = mish)((enc_fc0)))
    enc_fc2 = BatchNormalization()(
        Dense(
            units = 1000,
            kernel_initializer = 'glorot_normal',
            activation = mish)((enc_fc1)))
    mu = BatchNormalization()(
        Dense(
            units = z_dim,
            kernel_initializer = 'glorot_normal',
            activation = None)(enc_fc2))
    sigma = BatchNormalization()(
        Dense(
            units = z_dim,
            kernel_initializer = 'glorot_normal',
            activation = None)(enc_fc2))
    z = Lambda(
        sampling,
        output_shape = (z_dim, ))([mu, sigma])

    encoder = Model(enc_in, [mu, sigma, z])

    dec_in = Input(shape = (z_dim, ))

    dec_fc1 = BatchNormalization()(
        Dense(
            units = 1000,
            kernel_initializer = 'glorot_normal',
            activation = mish)(dec_in))
    dec_fc2 = BatchNormalization()(
        Dense(
            units = 1000,
            kernel_initializer = 'glorot_normal',
            activation = mish)(dec_fc1))
    dec_fc3 = BatchNormalization()(
        Dense(
            units = 3000,
            kernel_initializer = 'glorot_normal',
            activation = mish)(dec_fc2))
    dec_unflatten = Reshape(
        target_shape = (3,10,10,10))(dec_fc3)

    
    dec_conv3 = BatchNormalization()(
        Conv3DTranspose(
            filters = 32,
            kernel_size = (2, 2, 2),
            strides = (2, 2, 2),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = mish,
            data_format = 'channels_first')(dec_unflatten))
    dec_conv4 = BatchNormalization()(
        Conv3DTranspose(
            filters = 32,
            kernel_size = (2, 2, 2),
            strides = (2, 2, 2),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = mish,
            data_format = 'channels_first')(dec_conv3))
    dec_conv5 = BatchNormalization(
        beta_regularizer = l2(0.001),
        gamma_regularizer = l2(0.001))(
        Conv3DTranspose(
            filters = 3,
            kernel_size = (1, 1, 1),
            strides = (1, 1, 1),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            data_format = 'channels_first')(dec_conv4))

    decoder = Model(dec_in, dec_conv5)

    dec_conv5 = decoder(encoder(enc_in)[2])

    vae = Model(enc_in, dec_conv5)

    return {'inputs': enc_in, 
            'outputs': dec_conv5,
            'mu': mu,
            'sigma': sigma,
            'z': z,
            'encoder': encoder,
            'decoder': decoder,
            'vae': vae}
