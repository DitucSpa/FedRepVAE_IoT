import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import random
import cv2
seed = 42
random.seed(seed)
np.random.seed(seed)

def RestoreValues(image):
    image[image>1.] = 1.0
    image[image<0.] = 0.0
    return image

def ImageTranslation(image, border_value=1.0, min_value=-3, max_value=+3):
    tx = random.randint(min_value, max_value)
    ty = random.randint(min_value, max_value)
    return RestoreValues(cv2.warpAffine(image, 
                                        np.float32([[1, 0, tx], [0, 1, ty]]), 
                                        (image.shape[0], image.shape[1]), 
                                        borderValue=border_value))

def plot_images(images, labels, title, shapes=(28,28)):
    n_images = images.shape[0]
    rows = n_images//5 if n_images//5 < 5 else 5
    fig, axes = plt.subplots(5, rows, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.suptitle(title)
    for i, ax in enumerate(axes.flat):
        if i < n_images:
            image_data = images[i].reshape(shapes[0], shapes[1])
            ax.imshow(image_data, cmap='gray')
            ax.axis('off')
            ax.set_title(labels[i])
        else:
            fig.delaxes(ax)
    plt.show()
    return
    
def plot_distribution(labels, classes, title):
    plt.hist(labels, bins=np.arange(classes+1)-0.5, rwidth=0.8, align='mid')
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.xticks(range(classes))
    plt.title(title)
    plt.show()
    return

def convert_to_categorical(labels, classes):
    return to_categorical(labels, classes)

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), mean=0.0, stddev=1.0)
        return z_mean + 0.5 * tf.exp(0.5 * z_log_var) * epsilon

def GetMiniDataset(samples, labels, classes_choosen=3, support_size=20):
    classes = list(np.random.choice(list(set(labels)), classes_choosen, replace=False))
    df = pd.DataFrame({'samples': list(samples), 'labels': labels})
    df = df[df['labels'].isin(classes)]
    if df.shape[0] < support_size: df = df.sample(n=support_size, random_state=seed, replace=True)
    else: df = df.sample(n=support_size, random_state=seed, replace=False)
    support_indices = list(df.index)
    return samples[support_indices], labels[support_indices]


def GetMiniDatasetTest(samples, labels, classes_choosen=3, shots_per_class=4, classes=None):
    if not classes:
        if len(list(set(labels))) < classes_choosen: classes_choosen = len(list(set(labels)))
        classes = list(np.random.choice(list(set(labels)), classes_choosen, replace=False))
    df = pd.DataFrame({'samples': list(samples), 'labels': labels})
    df = df[df['labels'].isin(classes)]
    sample_result = df.groupby('labels').apply(lambda x: x.sample(n=min(shots_per_class, len(x))) if x.name in classes else pd.DataFrame())
    sample_result = sample_result.reset_index(drop=True, level='labels').sample(frac=1.0, random_state=42).index.values
    return samples[sample_result], labels[sample_result]


