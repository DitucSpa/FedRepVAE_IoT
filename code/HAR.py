import os
import json
import sys
import random
from scipy.ndimage import rotate
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
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
image_shape = (12,12)

class HAR:
    def __init__(self, path):
        self.path = os.path.normpath(path)
        self.classes = 6
    
    """
    user           --> the user to select from the dataset. If it isn't specified,
                       all the dataset is returned;
    ignore_classes --> a list of classes to ignore during the load of the dataset;
    test_size      --> the split of the test set (from 0. to 1.);
    shots          --> how many samples take from each class for the training set (>= 2).
                       In this case, the test_size param is ignored.
    """
    def get_dataset(self, user=None, ignore_classes = [], test_size = 0.4, shots = None):
        try:
                
            dataset = pd.read_csv(self.path)
            
            # delete all the rows that have the class specified in ignore_classes
            if ignore_classes: dataset = dataset[~dataset['Classes'].isin(ignore_classes)]
            
            # if user is defined, take the rows corresponding to this user
            if user: dataset = dataset[dataset['Subject'] == user]
            
            values = np.array(dataset.drop(columns=["Subject", "Classes"]))
            labels = np.array(dataset["Classes"])
            
            # if the dataset is empty, then there was something wrong
            if values.shape[0]==0 or labels.shape[0]==0: raise Exception("the dataset is empty.")
                
            # select only n-shots from each class
            if shots and shots >= 2:
                
                # for each class, save indices of that class in the dataset
                count_dict = {}
                for i, num in enumerate(labels):
                    if num in count_dict:
                        count_dict[num].append(i)
                    else:
                        count_dict[num] = [i]
                indices_for_classes = {key: value[:shots] for key, value in count_dict.items()}
                indices_selected = [item for sublist in indices_for_classes.values() for item in sublist]
                random.shuffle(indices_selected) # shuffle the dataset
                
                # x_train, y_train, x_test, y_test
                return (np.array(values)[indices_selected],
                        convert_to_categorical(np.array(labels)[indices_selected]-1, self.classes),
                        np.delete(values, indices_selected, 0),
                        convert_to_categorical(np.delete(labels, indices_selected)-1, self.classes))
                
                
            x_train, x_test, y_train, y_test = train_test_split(values, labels, test_size=test_size, 
                                                                stratify=labels, shuffle=True, 
                                                                random_state=seed)
            return x_train, convert_to_categorical(y_train-1, self.classes), x_test, convert_to_categorical(y_test-1, self.classes)

        except Exception as ex: 
            print("Something wrong:", ex)
            
    """
    Select only the first 144 features from the dataset with the highest variances (i.e. if the variance
    of a features is low, it means that this feature doesn't change along all the classes, so it's useless).
    Then, each sample is converted as an image 13x13x1.
    """
    def preprocessing(self, x_train, x_test):
        try:
            with open('HAR_selected_features.json', 'r') as f:
                selected_features = list(json.load(f))[:image_shape[0]*image_shape[1]]
            x_train = x_train[:, selected_features].reshape((-1, image_shape[0], image_shape[1], 1))
            x_test = x_test[:, selected_features].reshape((-1, image_shape[0], image_shape[1], 1))

            min_val = np.min(x_train)
            max_val = np.max(x_train)
            x_train = (x_train - min_val) / (max_val - min_val)
            x_test = (x_test - min_val) / (max_val - min_val)

            return x_train, x_test
        except Exception as ex: 
            print("Something wrong:", ex)   
            
    """
    dense_size                 --> number of neurons in the first dense layer (the others are divided by 2 * N_level);
    latent_dim                 --> latent dimension of the VAE;
    dropout                    --> dropout used in the VAE model only;
    regularizer                --> regularizer used in the VAE model only;
    classification_regularizer --> regularizer used in the Classification Head model only.
    """
    def create_model(self, dense_size=64, latent_dim=2, regularizer=0.1, classification_regularizer=0.5):
        # ====================== ENCODER ======================
        encoder_inputs = layers.Input(shape=(image_shape[0],image_shape[1],1), name="input")
        x = layers.Flatten()(encoder_inputs)
        x = layers.Dense(dense_size, activation='relu', kernel_regularizer=regularizers.L1(regularizer))(x)
        x = layers.Dense(dense_size//2, activation='relu', kernel_regularizer=regularizers.L1(regularizer))(x)
        z_mean = layers.Dense(latent_dim, name="z_mean", kernel_regularizer=regularizers.L1(regularizer))(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var", kernel_regularizer=regularizers.L1(regularizer))(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        # ====================== DECODER ======================
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(dense_size//4, activation='relu', 
                         kernel_regularizer=regularizers.L1(regularizer))(latent_inputs)
        x = layers.Dense(dense_size//2, activation='relu', 
                         kernel_regularizer=regularizers.L1(regularizer))(x)
        x = layers.Dense(dense_size, activation='relu', 
                         kernel_regularizer=regularizers.L1(regularizer))(x)
        x = layers.Dense(image_shape[0]*image_shape[1], activation='sigmoid', kernel_regularizer=regularizers.L1(regularizer))(x)
        x = layers.Reshape((image_shape[0], image_shape[1], 1))(x)
        decoder = keras.Model(latent_inputs, x, name="decoder")

        # ====================== CLASSIFICATION HEAD ======================
        classification_inputs = layers.Input(shape=(latent_dim,))
        x = layers.Dense(self.classes, activation="softmax", 
                         kernel_regularizer=regularizers.L1(classification_regularizer))(classification_inputs)
        class_model = keras.Model(classification_inputs, x, name="class_model")

        # ====================== CI-VAE ======================
        enc_inputs = layers.Input(shape=(image_shape[0], image_shape[1], 1))
        _, _, z = encoder(enc_inputs)
        x = decoder(z)
        class_out = class_model(z)
        CI_VAE = keras.Model(enc_inputs, [x, class_out], name='vae')
        return CI_VAE

    def create_model_CH(self, dense_size=64):
        inputs = keras.Input(shape=(image_shape[0], image_shape[1], 1), name = 'Input_Layer')
        flatten = layers.Flatten(name="flatten")(inputs)
        x = keras.layers.Dense(dense_size, activation='relu', name = '1_Dense')(flatten)
        x = keras.layers.Dense(dense_size//2, activation='relu', name = '2_Dense')(x)
        outputs = keras.layers.Dense(self.classes, activation='softmax', name = 'Output_Layer')(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name = 'CH')
        return model