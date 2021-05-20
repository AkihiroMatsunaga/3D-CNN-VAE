import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Input, BatchNormalization, Conv3D, Dense, Flatten, Lambda, Reshape, Conv3DTranspose
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

input_shape = (4, 80, 80, 80)
z_dim = 128

def sampling(args):
    mu, sigma = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    epsilon = K.random_normal(shape = (batch, dim))

    return mu + K.exp(0.5 * sigma) * epsilon

def get_model():
    enc_in = Input(shape = input_shape)

    enc_conv1 = BatchNormalization()(
        Conv3D(
            filters = 8,
            kernel_size = (3, 3, 3),
            strides = (1, 1, 1),
            padding = 'valid',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            data_format = 'channels_first')(enc_in))
    enc_conv2 = BatchNormalization()(
        Conv3D(
            filters = 16,
            kernel_size = (3, 3, 3),
            strides = (2, 2, 2),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            data_format = 'channels_first')(enc_conv1))
    enc_conv3 = BatchNormalization()(
        Conv3D(
            filters = 32,
            kernel_size = (3, 3, 3),
            strides = (1, 1, 1),
            padding = 'valid',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            data_format = 'channels_first')(enc_conv2))
    enc_conv4 = BatchNormalization()(
        Conv3D(
            filters = 64,
            kernel_size = (3, 3, 3),
            strides = (2, 2, 2),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            data_format = 'channels_first')(enc_conv3))
    enc_conv5 = BatchNormalization()(
        Conv3D(
            filters = 128,
            kernel_size = (3, 3, 3),
            strides = (1, 1, 1),
            padding = 'valid',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            data_format = 'channels_first')(enc_conv4))
    enc_conv6 = BatchNormalization()(
        Conv3D(
            filters = 256,
            kernel_size = (3, 3, 3),
            strides = (2, 2, 2),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            data_format = 'channels_first')(enc_conv5))

    enc_fc1 = BatchNormalization()(
        Dense(
            units = 729,
            kernel_initializer = 'glorot_normal',
            activation = 'elu')(Flatten()(enc_conv6)))
    mu = BatchNormalization()(
        Dense(
            units = z_dim,
            kernel_initializer = 'glorot_normal',
            activation = None)(enc_fc1))
    sigma = BatchNormalization()(
        Dense(
            units = z_dim,
            kernel_initializer = 'glorot_normal',
            activation = None)(enc_fc1))
    z = Lambda(
        sampling,
        output_shape = (z_dim, ))([mu, sigma])

    encoder = Model(enc_in, [mu, sigma, z])

    dec_in = Input(shape = (z_dim, ))

    dec_fc1 = BatchNormalization()(
        Dense(
            units = 729,
            kernel_initializer = 'glorot_normal',
            activation = 'elu')(dec_in))
    dec_unflatten = Reshape(
        target_shape = (1,9,9,9))(dec_fc1)

    dec_conv0 = BatchNormalization()(
        Conv3DTranspose(
            filters = 256,
            kernel_size = (3, 3, 3),
            strides = (1, 1, 1),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            data_format = 'channels_first')(dec_unflatten))
    dec_conv01 = BatchNormalization()(
        Conv3DTranspose(
            filters = 128,
            kernel_size = (3, 3, 3),
            strides = (2, 2, 2),
            padding = 'valid',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            data_format = 'channels_first')(dec_conv0))
    dec_conv1 = BatchNormalization()(
        Conv3DTranspose(
            filters = 64,
            kernel_size = (3, 3, 3),
            strides = (1, 1, 1),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            data_format = 'channels_first')(dec_conv01))
    dec_conv2 = BatchNormalization()(
        Conv3DTranspose(
            filters = 32,
            kernel_size = (3, 3, 3),
            strides = (2, 2, 2),
            padding = 'valid',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            data_format = 'channels_first')(dec_conv1))
    dec_conv3 = BatchNormalization()(
        Conv3DTranspose(
            filters = 16,
            kernel_size = (3, 3, 3),
            strides = (1, 1, 1),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            data_format = 'channels_first')(dec_conv2))
    dec_conv4 = BatchNormalization()(
        Conv3DTranspose(
            filters = 8,
            kernel_size = (4, 4, 4),
            strides = (2, 2, 2),
            padding = 'valid',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            data_format = 'channels_first')(dec_conv3))
    dec_conv5 = BatchNormalization(
        beta_regularizer = l2(0.001),
        gamma_regularizer = l2(0.001))(
        Conv3DTranspose(
            filters = 1,
            kernel_size = (3, 3, 3),
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