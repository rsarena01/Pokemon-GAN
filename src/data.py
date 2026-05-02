import os
from PIL import Image
import numpy as np
import pandas as pd

def load_image_paths(dir_path):
    img_paths = []
    for root, _, files in os.walk(dir_path):
        for im in files:
            if not im.startswith('.'):
                img_paths.append(os.path.join(root, im))
    return img_paths

def compute_img_stats(img_paths):
    disk_size = 0
    pix_total = []
    pix_height = []
    pix_width = []
    for im in img_paths:
        # image size 
        disk_size += os.path.getsize(im)
        # pixel width and height
        with Image.open(im) as img:
            w, h = img.size
        pix_total.append(h*w)
        pix_height.append(h)
        pix_width.append(w)
    # list -> array
    pix_arr = np.array(pix_total)
    h_arr = np.array(pix_height)
    w_arr = np.array(pix_width)
    #bytes -> mb
    size_mb = disk_size/(1024**2)
    # total images in dataset
    num_im = len(img_paths)
    stats = {
        'total images': num_im,
        'disk size (MB)': size_mb,
        'mean (height)': h_arr.mean(),
        'mean (width)': w_arr.mean(),
        'mean (total pixels)': pix_arr.mean(),
        'std (total pixels)': pix_arr.std(),
        'min (total pixels)': pix_arr.min(),
        'max (total pixels)': pix_arr.max()
    }

    df = pd.DataFrame.from_dict(stats, orient='index', columns=['value'])
    df['value'] = df['value'].round(2)

    return df

