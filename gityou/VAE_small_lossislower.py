import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Input, BatchNormalization, Conv3D, Dense, Flatten, Lambda, Reshape, Concatenate, Conv3DTranspose, MaxPooling3D, Dropout
from tensorflow.keras.regularizers import l1,l2
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
            filters = 32,
            kernel_size = (2, 2, 2),
            strides = (1, 1, 1),
            padding = 'same',
            kernel_initializer = 'random_uniform',
#            activity_regularizer=l1(0.01),
            activation = mish,
            data_format = 'channels_first')(enc_in))
    enc_conv1 = BatchNormalization()(
        Conv3D(
            filters = 128,
            kernel_size = (2, 2, 2),
            strides = (1, 1, 1),
            padding = 'same',
            kernel_initializer = 'random_uniform',
    #        activity_regularizer=l1(0.01),
            activation = mish,
            data_format = 'channels_first')(enc_conv0))
    enc_conv2 = BatchNormalization()(
        MaxPooling3D(
            pool_size=(2, 2, 2),
            padding="valid",
            data_format = 'channels_first')(enc_conv1))
    enc_conv3 = BatchNormalization()(
        Conv3D(
            filters = 512,
            kernel_size = (3, 3, 3),
            strides = (1, 1, 1),
            padding = 'same',
            kernel_initializer = 'random_uniform',
  #          activity_regularizer=l1(0.01),
            activation = mish,
            data_format = 'channels_first')(enc_conv2))
    enc_conv4 = BatchNormalization()(
        MaxPooling3D(
            pool_size=(2, 2, 2),
            padding="valid",
            data_format = 'channels_first')(enc_conv3))
    enc_conv5 = BatchNormalization()(
        Conv3D(
            filters = 1024,
            kernel_size = (2, 2, 2),
            strides = (1, 1, 1),
            padding = 'same',
            kernel_initializer = 'random_uniform',
            activation = mish,
            data_format = 'channels_first')(enc_conv4))
    enc_conv6 = BatchNormalization()(
        MaxPooling3D(
            pool_size=(2, 2, 2),
            padding="valid",
            data_format = 'channels_first')(enc_conv5))
    enc_conv7 = BatchNormalization()(
        Conv3D(
            filters = 2048,
            kernel_size = (2, 2, 2),
            strides = (1, 1, 1),
            padding = 'same',
            kernel_initializer = 'random_uniform',
            activation = mish,
            data_format = 'channels_first')(enc_conv6))
    
    enc_fc0 = BatchNormalization()(
        Dense(
            units = 1000,
            kernel_initializer = 'random_uniform',
            activation = mish)(Flatten()(enc_conv7)))
    enc_fc1 = BatchNormalization()(
        Dense(
            units = 1000,
            kernel_initializer = 'random_uniform',
            activation = mish)((enc_fc0)))
    mu = BatchNormalization()(
        Dense(
            units = z_dim,
            kernel_initializer = 'random_uniform',
            activation = None)(enc_fc0))
    sigma = BatchNormalization()(
        Dense(
            units = z_dim,
            kernel_initializer = 'random_uniform',
            activation = None)(enc_fc0))
    z = Lambda(
        sampling,
        output_shape = (z_dim, ))([mu, sigma])

    encoder = Model(enc_in, [mu, sigma, z])

    dec_in = Input(shape = (z_dim, ))

    dec_fc1 = BatchNormalization()(
        Dense(
            units = 375,
            kernel_initializer = 'random_uniform',
            activation = mish)(dec_in))
    dec_unflatten = Reshape(
        target_shape = (3,5,5,5))(dec_fc1)

    
    dec_conv1 = BatchNormalization()(
        Conv3DTranspose(
            filters = 2048,
            kernel_size = (2, 2, 2),
            strides = (1, 1, 1),
            padding = 'same',
            kernel_initializer = 'random_uniform',
            activation = mish,
            data_format = 'channels_first')(dec_unflatten))
    dec_conv1_1 = BatchNormalization()(
        Conv3DTranspose(
            filters = 1024,
            kernel_size = (2, 2, 2),
            strides = (1, 1, 1),
            padding = 'same',
            kernel_initializer = 'random_uniform',
            activation = mish,
            data_format = 'channels_first')(dec_conv1))
    dec_conv1_2 = Dropout(0.2)(
        BatchNormalization()(
        Conv3DTranspose(
            filters = 512,
            kernel_size = (2, 2, 2),
            strides = (1, 1, 1),
            padding = 'same',
            kernel_initializer = 'random_uniform',
            activation = mish,
            data_format = 'channels_first')(dec_conv1_1)))
    dec_conv2 = BatchNormalization()(
        Conv3DTranspose(
            filters = 128,
            kernel_size = (2, 2, 2),
            strides = (2, 2, 2),
            padding = 'same',
            kernel_initializer = 'random_uniform',
            activity_regularizer=l1(0.001),
            activation = mish,
            data_format = 'channels_first')(dec_conv1_2))
    dec_conv3 = Dropout(0.2)(
        BatchNormalization()(
        Conv3DTranspose(
            filters = 32,
            kernel_size = (2, 2, 2),
            strides = (2, 2, 2),
            padding = 'same',
            kernel_initializer = 'random_uniform',
            activation = mish,
            data_format = 'channels_first')(dec_conv2)))
    dec_conv4 = Dropout(0.2)(
        BatchNormalization()(
        Conv3DTranspose(
            filters = 32,
            kernel_size = (2, 2, 2),
            strides = (2, 2, 2),
            padding = 'same',
            kernel_initializer = 'random_uniform',
  #          activity_regularizer=l1(0.001),
            activation = mish,
            data_format = 'channels_first')(dec_conv3)))
    dec_conv5 = BatchNormalization(
        beta_regularizer = l2(0.001),
        gamma_regularizer = l2(0.001))(
        Conv3D(
            filters = 16,
            kernel_size = (3, 3, 3),
            strides = (1, 1, 1),
            padding = 'same',
            kernel_initializer = 'random_uniform',
            data_format = 'channels_first')(dec_conv4))
    dec_conv6 = BatchNormalization(
        beta_regularizer = l2(0.001),
        gamma_regularizer = l2(0.001))(
        Conv3D(
            filters = 3,
            kernel_size = (2, 2, 2),
            strides = (1, 1, 1),
            padding = 'same',
            kernel_initializer = 'random_uniform',
            data_format = 'channels_first')(dec_conv5))

    decoder = Model(dec_in, dec_conv6)

    dec_conv6 = decoder(encoder(enc_in)[2])

    vae = Model(enc_in, dec_conv6)

    return {'inputs': enc_in, 
            'outputs': dec_conv6,
            'mu': mu,
            'sigma': sigma,
            'z': z,
            'encoder': encoder,
            'decoder': decoder,
            'vae': vae}
