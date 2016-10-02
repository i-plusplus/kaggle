import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def draw(t, name):
        t = np.split(t,28)
        plt.imsave(name, np.asarray(t), cmap = plt.get_cmap('gray'))

