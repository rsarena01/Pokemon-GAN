import matplotlib.pyplot as plt
import os
import json
import tensorflow as tf 
from tensorflow import keras
import numpy as np 
from datetime import datetime 

from data import load_image_paths
from preprocessing import build_tf_dataset
from models.architectures import (
    build_upsample_generator,
    build_transpose_generator,
    build_discriminator,
    build_critic,
)
from models.wgan_gp import WGAN_GP
from models.dcgan import DCGAN

class SaveImages(tf.keras.callbacks.Callback):
    def __init__(self, generator, latent_dim, image_dir, freq=10):
        # subclass from parent class
        super().__init__()
        self.generator = generator
        self.latent_dim = latent_dim
        
        self.image_dir = image_dir
        os.makedirs(self.image_dir, exist_ok=True)
        
        self.freq = freq
        # fixed noise for progress tracking 
        self.fixed_noise=tf.random.normal([16, latent_dim])

    def on_epoch_end(self, epoch, logs=None):
        # every freq num epochs
        if (epoch+1) % self.freq != 0:
            return 
            
        gen_images = self.generator(self.fixed_noise, training=False)
        # tensor -> np array
        gen_images = gen_images.numpy()
        # [-1,1] -> [0,1] for matplotlib
        gen_images = (gen_images+1)/2
        fig, axes = plt.subplots(4,4, figsize=(4,4))
        axes = axes.flatten()
        for i,ax in enumerate(axes):
            ax.imshow(gen_images[i])
            ax.axis('off')
            ax.grid(False)
        plt.tight_layout()

        save_path = os.path.join(self.image_dir, f'epoch{epoch+1}_images.png')
        plt.savefig(save_path)
        plt.close()

def save_metrics(history, save_path):
    hist_dict = history.history 
    with open(save_path, 'w') as f:
        json.dump(hist_dict, f, indent=4)

def run(model_type = 'wgan', generator_type='upsample', epochs=50, batch_size=128, latent_dim=100):
    # folder for all results 
    results_root = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_root, exist_ok=True)

    # save folder for individual run 
    run_name = f"{model_type}_{generator_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(results_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # raw images path
    raw_img_dir = '../data/raw'
  
    # load data
    img_paths = load_image_paths(raw_img_dir)
    dataset= build_tf_dataset(img_paths, batch_size=batch_size)

    # build models 
    if generator_type == 'upsample':
        generator = build_upsample_generator(latent_dim=latent_dim)
    else:
        generator = build_transpose_generator(latent_dim=latent_dim)

    if model_type == 'wgan':
        critic = build_critic()
        model = WGAN_GP(critic, generator, latent_dim)

        model.compile(
            c_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
            g_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        )

    else:
        discriminator = build_discriminator()
        model = DCGAN(discriminator, generator, latent_dim)

        model.compile(
            loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            d_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
            g_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        )

    #callbacks 
    image_dir = os.path.join(run_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    callbacks = [SaveImages(generator=generator, latent_dim=latent_dim, image_dir=image_dir)]

    # train
    history = model.fit(dataset, epochs=epochs, callbacks=callbacks)

    # save metrics
    save_metrics(history, os.path.join(run_dir, 'history.json'))

    # save model
    generator.save(os.path.join(run_dir, 'generator.keras'))

if __name__ == '__main__':
    run(model_type='wgan', generator_type='upsample', epochs=50)