import imageio
import pathlib
import numpy as np

def load_image():
    paths = pathlib.Path('./data/trainA').glob('/*.jpg')
    paths_sorted = sorted([x for x in paths])
    im_path = paths_sorted[45]
    im = imageio.imread(str(im_path))
