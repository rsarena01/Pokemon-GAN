import tensorflow as tf
from tensorflow import keras

def build_discriminator(leak=0.2, momentum=0.9, dropout=0.3):
    # discriminator input
    din = tf.keras.Input(shape=(64,64,3))
    # shape = 64 x 64
    
    x = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same', use_bias=False)(din)
    # no batch normalization first layer
    x = tf.keras.layers.LeakyReLU(leak)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    # shape = 32 x 32 -> stride on conv layer downsamples
    
    x = tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    # shape = 16 x 16
    
    x = tf.keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.LeakyReLU(leak)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    # shape = 8 x 8
    
    x = tf.keras.layers.Conv2D(512, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.LeakyReLU(leak)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    # shape = 4 x 4
    
    # convolutional layer, sigmoid activation handled by BCE loss from_logits=True
    x = tf.keras.layers.Conv2D(
        1,
        kernel_size=4,
        strides=1,
        padding='valid',
        use_bias=False,
    )(x)
    
    # shape = 1 x 1
    
    # dicriminator output
    dout = tf.keras.layers.Flatten()(x)
    
    # shape = (batch size, 1)
    
    # discriminator
    return tf.keras.models.Model(din, dout)

def build_critic(leak=0.2):
    # critic input
    cin = tf.keras.Input(shape=(64,64,3)) # RGB
    
    # shape = 64 x 64
    
    #convolutional layer
    x = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same', use_bias=False)(cin)
    
    # shape = 32 x 32 -> conv transpose upamples
    
    x = tf.keras.layers.LeakyReLU(leak)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    # shape = 16 x 16
    
    x = tf.keras.layers.LeakyReLU(leak)(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    # shape = 8 x 8
    
    x = tf.keras.layers.LeakyReLU(leak)(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    # shape = 4 x 4
    
    x = tf.keras.layers.LeakyReLU(leak)(x)
    x = tf.keras.layers.Conv2D(
        1,
        kernel_size=4,
        strides=1,
        padding='valid',
        use_bias=False,
    )(x)
    
    # shape = 1 x 1
    
    # critic output
    cout = tf.keras.layers.Flatten()(x)
    
    # shape = (batch size, 1)
    
    # critic
    return tf.keras.models.Model(cin, cout)

def build_upsample_generator(latent_dim=100, leak=0.2, momentum=0.9):
    layer = tf.keras.layers
    # gen input
    gin = layer.Input((latent_dim,))
    # project latent vector space to h * w * n filters dimensions
    x = layer.Dense(4*4*1024, use_bias=False)(gin)
    x = layer.Reshape((4,4,1024))(x)
    x = layer.BatchNormalization(momentum=momentum)(x)
    x = layer.LeakyReLU(0.2)(x)
    
    # shape = 4 x 4 x 1024
    
    #default upsamping size (2,2), interpolation=nearest
    x = layer.UpSampling2D()(x)
    x = layer.Conv2D(512, 3, padding='same', use_bias=False)(x)
    x = layer.BatchNormalization(momentum=momentum)(x)
    x = layer.LeakyReLU(leak)(x)
    # extra refinement conv
    x = layer.Conv2D(512, 3, padding='same', use_bias=False)(x)
    x = layer.BatchNormalization(momentum=momentum)(x)
    x = layer.LeakyReLU(leak)(x)
    # shape = 8 x 8 x 512
    
    x = layer.UpSampling2D()(x)
    x = layer.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layer.BatchNormalization(momentum=momentum)(x)
    x = layer.LeakyReLU(leak)(x)
    # extra refinement conv
    x = layer.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layer.BatchNormalization(momentum=momentum)(x)
    x = layer.LeakyReLU(leak)(x)
    # shape = 16 x 16 x 256
    
    x = layer.UpSampling2D()(x)
    x = layer.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layer.BatchNormalization(momentum=momentum)(x)
    x = layer.LeakyReLU(leak)(x)
    # extra refinement conv
    x = layer.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layer.BatchNormalization(momentum=momentum)(x)
    x = layer.LeakyReLU(leak)(x)
    # shape = 32 x 32 x 128
    
    x = layer.UpSampling2D()(x)
    x = layer.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layer.BatchNormalization(momentum=momentum)(x)
    x = layer.LeakyReLU(leak)(x)
    # extra refinement conv
    x = layer.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layer.BatchNormalization(momentum=momentum)(x)
    x = layer.LeakyReLU(leak)(x)
    # shape = 64 x 64 x 64
    
    #output
    gout = layer.Conv2D(3, 3, padding='same', activation='tanh')(x)
    
    return tf.keras.models.Model(gin, gout)

def build_transpose_generator(latent_dim=100, momentum=0.9):
    layer = tf.keras.layers
    # gen input 
    gin = layer.Input(shape=(latent_dim,))
    # latent vector space 
    x = layer.Reshape((1,1,latent_dim))(gin)
    
    # shape = 1 x 1 x latent_dim C
    
    x = layer.Conv2DTranspose(512, kernel_size=4, strides=1, padding='valid', use_bias=False)(x)
    # shape = 4 x 4 x 512 -> transpose conv strides up sample
    
    x = layer.BatchNormalization(momentum=momentum)(x)
    x = layer.LeakyReLU(0.2)(x)
    x = layer.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    # shape = 8 x 8 x 256
    
    x = layer.BatchNormalization(momentum=momentum)(x)
    x = layer.LeakyReLU(0.2)(x)
    x = layer.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    # shape = 16 x 16 x 128
    
    x = layer.BatchNormalization(momentum=momentum)(x)
    x = layer.LeakyReLU(0.2)(x)
    x = layer.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    # shape = 32 x 32 x 64
    
    x = layer.BatchNormalization(momentum=momentum)(x)
    x = layer.LeakyReLU(0.2)(x)
    gout = layer.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', use_bias=False, activation='tanh')(x)
    # shape = 64 x 64 x 3 -> same as real image 
    
    # generator
    return tf.keras.models.Model(gin, gout)