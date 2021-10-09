# https://www.tensorflow.org/datasets/overview
# https://www.tensorflow.org/datasets/catalog/gigaword

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

DIR = os.path.abspath(os.path.dirname(__file__))

builder = tfds.builder('gigaword')
builder.download_and_prepare(download_dir=os.path.join(DIR, 'data'))

