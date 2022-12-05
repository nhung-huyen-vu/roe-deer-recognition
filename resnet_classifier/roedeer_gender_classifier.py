import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import tensorflow.keras
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, MaxPool2D, Concatenate, RandomRotation, RandomFlip, RandomContrast, RandomBrightness, Lambda, RandomTranslation, CategoryEncoding
from tensorflow.keras.preprocessing import image
from tensorflow.keras.initializers import glorot_uniform

# Single-hidden layer network for processing month information for each sample
def MonthBaseNetwork(input_shape=(1)):
    X_input = Input(input_shape)
    X = Lambda(lambda x: x - 1)(X_input)
    X = CategoryEncoding(num_tokens=12, output_mode = "one_hot")(X)
    X = Dense(1, activation='relu', name='month_out')(X)

    model = Model(inputs=X_input, outputs=X, name='MonthBase')

    return model

# Image augmentation layers
def AugmentationLayers(X_input):
    X = RandomFlip(mode="horizontal")(X_input)
    X = RandomRotation(factor=(-0.03, 0.03))(X)
    # X = RandomContrast(0.2)(X)
    # X = RandomBrightness((-0.2, 0.2))(X)
    X = RandomTranslation((-0.1, 0.1), (-0.1, 0.1))(X)
    return X

# Fully connected head concating month input and ResNet50 input
def HeadModel(resnet_base_output, month_base_output):
    headModel = Flatten()(resnet_base_output)
    headModel = Concatenate()([headModel, month_base_output])
    # headModel = Dense(256, activation='relu', name='fc1',kernel_initializer=glorot_uniform(seed=0))(headModel)
    headModel = Dense(128, activation='relu', name='fc2',kernel_initializer=glorot_uniform(seed=0))(headModel)
    headModel = Dense(4, activation='softmax', name='softmax_out')(headModel)
    return headModel

def RoeDeerGenderClassificationModel():
    X_input = Input((224,224,3))
    X = AugmentationLayers(X_input)
    base_resnet = tf.keras.applications.ResNet50V2(include_top=False, weights="imagenet", pooling="avg")
    X = base_resnet(X, training=False)

    base_month = MonthBaseNetwork()

    head = HeadModel(X, base_month.output)

    model = Model(inputs=[X_input, base_month.input], outputs=head)

    return model, base_resnet, base_month, head 

