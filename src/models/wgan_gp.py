import tensorflow as tf
from tensorflow import keras

class WGAN_GP(tf.keras.Model):
    def __init__(self, critic, generator, latent_dim):
        # inherent from parent class
        super().__init__()
        self.critic=critic
        self.generator=generator
        self.latent_dim=latent_dim

    # metrics must be an attribute to be handled correctly during fit i.e., reset 
    @property
    def metrics(self):
        return [self.c_loss_metric, self.c_wass_loss_metric, self.c_gp_metric, self.g_loss_metric]

    def compile(self, c_optimizer, g_optimizer, gp_weight = 10.0, n_critic=3):
        super().compile()
        # critic optimizer
        self.c_optimizer = c_optimizer
        # generator optimizer
        self.g_optimizer = g_optimizer
        # gradient penalty weight
        self.gp_weight=gp_weight
        # critic update iterations
        self.n_critic = n_critic
        
        
        # keras metrics aggregates over batches and resets after each epoch 
      
        # critic loss
        self.c_loss_metric = tf.keras.metrics.Mean(name='c_loss')
        # critic wassertein loss
        self.c_wass_loss_metric = tf.keras.metrics.Mean(name='c_wass_loss')
        # critic gradient penalty 
        self.c_gp_metric = tf.keras.metrics.Mean(name='c_gp')
        # generator loss
        self.g_loss_metric = tf.keras.metrics.Mean(name='g_loss')

    def gradient_penalty(self, real, fake):
        # batch size
        bsize=tf.shape(real)[0]
        # interpolation weights 
        alpha = tf.random.uniform([bsize, 1, 1, 1], 0.0, 1.0)
        diff = fake - real
        # points lie on straight line between real and fake samples
        interpolated = real + alpha*diff
       
        with tf.GradientTape() as gp_tape:
            # differentiate with respect to interpolated 
            gp_tape.watch(interpolated)
            # score interpolated images
            pred = self.critic(interpolated, training=True)
        grads = gp_tape.gradient(pred, interpolated)
       
        # distance using l2 norm plus epsilon  for stability 
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1,2,3]) + 1e-12)
       
        # average sq distance between l2 norm and 1
        # try to approximate normed gradient of critic to 1 (Lipschitz constraint)
        gp = tf.reduce_mean((norm-1.0)**2)
        return gp

    # WGAN-GP
    @tf.function
    def train_step(self, real_images):
        # batch size
        bsize = tf.shape(real_images)[0]

        #######----- train critic -----#######
        total_c_loss = tf.constant(0.0)
        total_c_wass = tf.constant(0.0)
        total_c_gp = tf.constant(0.0)
       
        # number of critic updates
        for i in range(self.n_critic):
            random_latent_vectors = tf.random.normal([bsize, self.latent_dim])
            with tf.GradientTape() as c_tape:
                fake_images = self.generator(random_latent_vectors, training=True)
                fake_preds = self.critic(fake_images, training=True)
                real_preds = self.critic(real_images, training=True)
                
                # average difference between fake and real 
                c_wass_loss = tf.reduce_mean(fake_preds) - tf.reduce_mean(real_preds)
                total_c_wass += c_wass_loss
                
                # gradient penalty
                c_gp = self.gradient_penalty(real_images, fake_images)
                total_c_gp += c_gp
                
                c_loss = c_wass_loss + c_gp*self.gp_weight
                total_c_loss += c_loss
                
            # calculate gradient and update weights 
            c_gradient = c_tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(c_gradient, self.critic.trainable_variables)) # grad, var pairs

        # update critic metrics with average across critic updates
        self.c_loss_metric.update_state(total_c_loss/self.n_critic)
        self.c_wass_loss_metric.update_state(total_c_wass/self.n_critic)
        self.c_gp_metric.update_state(total_c_gp/self.n_critic)
    
        #######----- train genrator -----#######
        random_latent_vectors = tf.random.normal([bsize, self.latent_dim])
       
        with tf.GradientTape() as g_tape:
            fake_images = self.generator(random_latent_vectors, training=True)
            fake_preds = self.critic(fake_images, training=True)
            g_loss = -tf.reduce_mean(fake_preds)
    
        # calculate gradients and update weights 
        gen_gradient = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
    
        # update gen loss
        self.g_loss_metric.update_state(g_loss)
        
        return {m.name: m.result() for m in self.metrics}