import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

idx2char = ['h', 'i', 'e', 'l', 'o']

x_data= [[0, 1, 0, 2, 3, 3]] # hihell
x_one_hot = [[[1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0]]]

y_data = [[1, 0, 2, 3, 3, 4]] # ihello


