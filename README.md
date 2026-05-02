# Pokémon GAN Project

This project provides a modular framework for building and training Generative Adversarial Networks (GANs) to generate novel Pokémon images. It supports both:

- Deep Convolutional GAN (DCGAN)
- Wasserstein GAN with Gradient Penalty (WGAN-GP)

---

## Dataset

The model is trained on the:

**Complete Pokémon Image Dataset**  
https://www.kaggle.com/datasets/hlrhegemony/pokemon-image-dataset  

All raw images should be placed in:

```
data/raw
```

---

## Project Structure

```
pokemon-gan/
├── analysis.ipynb
├── data/
│   └── raw/
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── results/
│   └── models/
│       ├── architectures.py
│       ├── dcgan.py
│       └── wgan_gp.py
```

---

## Modules Overview

### src/models/architectures.py

Contains reusable model-building functions.

#### Generators

- **build_transpose_generator(latent_dim=100, momentum=0.9)**
  - Uses Conv2DTranspose  
  - Standard DCGAN-style upsampling  

- **build_upsample_generator(latent_dim=100, leak=0.2, momentum=0.9)**
  - Uses UpSampling2D + Conv2D  
  - Reduces checkerboard artifacts  

#### Discriminator (DCGAN)

- **build_discriminator(leak=0.2, momentum=0.9, dropout=0.3)**

#### Critic (WGAN-GP)

- **build_critic(leak=0.2)**

---

### src/models/dcgan.py

Defines the DCGAN training class.

**Class: DCGAN**

- compile() defaults:
  - loss: `BinaryCrossentropy(from_logits=True)`
  - optimizers: Adam (lr=2e-4, beta_1=0.5)

---

### src/models/wgan_gp.py

Defines the WGAN-GP training class.

**Class: WGAN_GP**

- compile() defaults:
  - c_optimizer: Adam (2e-4, beta_1=0.5)
  - g_optimizer: Adam (2e-4, beta_1=0.5)
  - gp_weight = 10.0
  - n_critic = 3

---

### src/data.py

Loads image paths from disk.

- **load_image_paths(dir_path)**

---

### src/preprocessing.py

Streams image data into TensorFlow.

- **load_and_preprocess_image(path)**
  - Resize: 64×64  
  - Normalize: [-1, 1]  

- **build_tf_dataset(img_paths, batch_size=128)**
  - Shuffling enabled  
  - Batching enabled  
  - Streaming (no full RAM load)  

---

### src/train.py

Main training pipeline.

**Function:**

```
run(...)
```

**Default parameters:**

```
model_type = 'wgan'
generator_type = 'upsample'
epochs = 50
batch_size = 128
latent_dim = 100
```

---

### SaveImages Callback

**Class: SaveImages**

**Default parameters:**

```
freq = 10        # save every 10 epochs
latent_dim = 100
```

---

## Results Output

All results are saved in:

```
src/results/{run_dir}/
```

Where:

```
run_dir = {model_type}_{generator_type}_{timestamp}
```

Example:

```
wgan_upsample_20260501_123045/
```

---

### Each run directory contains:

```
run_dir/
├── history.json          # training metrics
├── generator.keras       # trained generator
└── images/
    ├── epoch10_images.png
    ├── epoch20_images.png
    └── ...
```

---

## Analysis Notebook

The `analysis.ipynb` notebook demonstrates:

- Viewing raw dataset images  
- Inspecting image dimensions and statistics  
- Visualizing TensorFlow dataset samples  
- Plotting training metrics from `history.json`  

---

## Key Design Features

- Modular architecture design (swap generators/discriminators easily)  
- Streaming dataset pipeline (memory efficient)  
- Support for multiple GAN variants  
- Automatic experiment tracking via run directories  
- Image logging for training visualization  
