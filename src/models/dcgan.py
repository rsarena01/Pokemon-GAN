import tensorflow as tf
from tensorflow import keras

class DCGAN(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        # inherent from parent class
        super().__init__()
        self.discriminator=discriminator
        self.generator=generator
        self.latent_dim=latent_dim

    # metrics must be an attribute to be handled correctly during fit i.e., reset 
    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.loss_fn = loss_fn
        # discriminator optimizer
        self.d_optimizer = d_optimizer
        # generator optimizer
        self.g_optimizer = g_optmizer
        # keras metrics aggregates over batches and resets after each epoch 
        self.d_loss_metric = tf.keras.metrics.Mean(name='d_loss')
        self.g_loss_metric = tf.keras.metrics.Mean(name='g_loss')
    
    def train_step(self, real_imgs):
        batch_size = tf.shape(real_imgs)[0]
       
        #######----- train discriminator -----#######
        latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        # fake images
        fake_imgs = self.generator(latent_vectors, training=True)
        
        # true labels + noise
        fake_labels = tf.random.uniform([batch_size,1], 0, 0.1)
        real_labels = tf.random.uniform([batch_size,1], 0.9, 1)
        
        with tf.GradientTape() as d_tape:
            real_preds = self.discriminator(real_imgs, training=True)
            fake_preds = self.discriminator(fake_imgs, training=True)
           
            # minimize loss from true labels and predictions on real and fake images using BCE
            d_real_loss = self.loss_fn(real_labels, real_preds)
            d_fake_loss = self.loss_fn(fake_labels, fake_preds)
            d_loss = (d_real_loss + d_fake_loss)/2
        
        # update 
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        ####### ----- train generator -----#######
        with tf.GradientTape() as g_tape:
            # new latent vectors/images
            latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            # generate and predict fake images 
            fake_imgs = self.generator(latent_vectors, training=True)
            fake_preds = self.discriminator(fake_imgs, training=True)
            
            # all images labeled real
            misleading_labels = tf.ones_like(fake_preds)
            
            # minimize loss from current discriminator predictions and predicting all real
            g_loss = self.loss_fn(misleading_labels, fake_preds)
       
        # update
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        #update metrics
        self.g_loss_metric.update_state(g_loss)
        self.d_loss_metric.update_state(d_loss)

        return {m.name: m.result() for m in self.metrics} # subclassing metrics 