import os
import json
import sys
import random
from scipy.ndimage import rotate
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import *
tf.config.set_visible_devices([], 'GPU')
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class FEMNIST:
    def __init__(self, person, angle, classes, path):
        self.person = person
        self.angle = angle
        self.classes = classes
        self.path = os.path.normpath(path)
    
    """
    ignore_classes --> the classes to ignore for the user selected;
    all_dataset --> return the entire dataset or only the data associated to an user;
    test_size --> the split of the test set (from 0 to 1);
    shots --> how many samples take from each class for the training set (>= 2)
    """
    def get_dataset(self,  ignore_classes = [], all_dataset = False, test_size = None, data_aug=False, shots = None):
        try:
            # load the entire dataset
            dataset = None
            with open(self.path) as json_file:
                dataset = json.load(json_file)

            if all_dataset: return dataset # return the entire dataset

            # return the data for a certain user
            images = []
            labels = []
            for i in range(len(dataset['user_data'][self.person]['y'])):
                label = dataset['user_data'][self.person]['y'][i]
                if label > self.classes-1: continue # ignore the letter dataset
                if ignore_classes and label in ignore_classes: continue
                image = np.array(dataset['user_data'][self.person]['x'][i]).reshape((28, 28, 1))
                images.append(image)
                labels.append(label)
                
            del dataset
            images = np.array(images)
            labels = np.array(labels)

            # rotate the images
            images = np.array([rotate(image.squeeze(), int(self.angle), reshape=False, cval=1.) for image in images])
            images[images > 1] = 1.0
            images[images < 0] = 0.0
            images = images.reshape((len(images), 28, 28, 1))

            # if shots is passed, return a training set with n-shots for each class and
            # a test set with all the other images
            if shots and shots >= 2:
                count_dict = {}
                for i, num in enumerate(labels):
                    if num in count_dict:
                        count_dict[num].append(i)
                    else:
                        count_dict[num] = [i]
                indices_for_classes = {key: value[:shots] for key, value in count_dict.items()}
                indices_selected = [item for sublist in indices_for_classes.values() for item in sublist]
                random.shuffle(indices_selected)

                # x_train, y_train, x_test, y_test
                return (np.array(images)[indices_selected],
                        convert_to_categorical(np.array(labels)[indices_selected], self.classes),
                        np.delete(images, indices_selected, 0).reshape((len(images)-len(indices_selected),28,28,1)),
                        convert_to_categorical(np.delete(labels, indices_selected), self.classes))

            # split the dataset into train and test


            x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, 
                                                                stratify=labels, shuffle=True, 
                                                                random_state=seed)
            return x_train, convert_to_categorical(y_train, self.classes), x_test, convert_to_categorical(y_test, self.classes)
        
        except Exception as ex: 
            print(ex)
            return [], [], [], []


    """
    k_size --> number of kernels in the first conv layer;
    latent_dim --> latent dimension of the VAE;
    dropout --> dropout used in the VAE model only;
    regularizer --> regularizer used in the VAE model only;
    classification_regularizer --> regularizer used in the Classification Head model only.
    """
    def create_model(self, k_size=16, latent_dim=3, regularizer=0.1, classification_regularizer=0.1, regularizer_decoder=0.2):
        # ====================== ENCODER ======================
        encoder_inputs = layers.Input(shape=(28, 28, 1), name="input")
        x = layers.Conv2D(k_size, 5, strides=1, padding="same", name="conv1", 
                          kernel_regularizer=regularizers.L1(regularizer))(encoder_inputs)
        #x = layers.BatchNormalization(name="batch1")(x)
        x = layers.ReLU(name="relu1")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
        x = layers.Conv2D(k_size*2, 3, strides=1, padding="same", name="conv2", 
                          kernel_regularizer=regularizers.L1(regularizer))(x)
        #x = layers.BatchNormalization(name="batch2")(x)
        x = layers.ReLU(name="relu2")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(32, activation="relu", name="dense1", kernel_regularizer=regularizers.L1(regularizer))(x)
        z_mean = layers.Dense(latent_dim, name="z_mean", kernel_regularizer=regularizers.L1(regularizer))(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var", kernel_regularizer=regularizers.L1(regularizer))(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        # ====================== DECODER ======================
        latent_inputs = layers.Input(shape=(latent_dim,))
        x = layers.Dense(100, activation="relu", kernel_regularizer=regularizers.L1(regularizer_decoder))(latent_inputs)
        x = layers.Dense(7 * 7 * k_size*2, activation="relu", kernel_regularizer=regularizers.L1(regularizer_decoder))(x)
        x = layers.Reshape((7, 7, k_size*2))(x)
        x = layers.Conv2DTranspose(k_size*2, 3, strides=2, padding="same", kernel_regularizer=regularizers.L1(regularizer_decoder))(x)
        #x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2DTranspose(k_size, 3, strides=2, padding="same", kernel_regularizer=regularizers.L1(regularizer_decoder))(x)
        #x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2DTranspose(1, 1, activation="sigmoid", padding="same",
                                   kernel_regularizer=regularizers.L1(regularizer_decoder))(x)
        decoder = keras.Model(latent_inputs, x, name="decoder")

        # ====================== CLASSIFICATION HEAD ======================
        classification_inputs = layers.Input(shape=(latent_dim,))
        x = layers.Dense(self.classes, activation="softmax", kernel_regularizer=regularizers.L1(classification_regularizer))(classification_inputs)
        class_model = keras.Model(classification_inputs, x, name="class_model")

        # ====================== CI-VAE ======================
        enc_inputs = layers.Input(shape=(28, 28, 1), name="input")
        _, _, z = encoder(enc_inputs)
        x = decoder(z)
        class_out = class_model(z)
        CI_VAE = keras.Model(enc_inputs, [x, class_out], name='CI_VAE')
        return CI_VAE
        
    def create_CH_model(self, k_size=16):
        input_layer = keras.Input(shape=(28,28,1), name="input")
        conv1 = layers.Conv2D(k_size, kernel_size=5, activation="relu", name="conv1")(input_layer)
        maxpool1 = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(conv1)
        conv2 = layers.Conv2D(k_size*2, kernel_size=3, activation="relu", name="conv2")(maxpool1)
        maxpool2 = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool2")(conv2)
        flatten = layers.Flatten(name="flatten")(maxpool2)
        dense = layers.Dense(100, activation="relu", name="dense1")(flatten)
        output_layer = layers.Dense(self.classes, activation="softmax", name="dense2")(dense)
        CH = keras.Model(inputs=input_layer, outputs=output_layer, name="CH")
        return CH


