# Run some setup code for this notebook.
from __future__ import print_function  

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

plt.show()