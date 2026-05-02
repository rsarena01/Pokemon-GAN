from PIL import Image 
import numpy as np
import os 
import tensorflow as tf 

from data import load_image_paths

def preprocess_image(path, size=(64,64)):
    raw = tf.io.read_file(path)
    image = tf.image.decode_image(raw, channels=3)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, size)
    # normalize [-1, 1]
    image = (tf.cast(image, dtype=tf.float32) - 127.5)/127.5
    return image

def build_tf_dataset(img_paths, batch_size=128):
    """streams images from local storage to build dataset"""
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    dataset = dataset.shuffle(len(img_paths))
    dataset = dataset.map(preprocess_image, num_parallel_calls= tf.data.AUTOTUNE).batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset 
